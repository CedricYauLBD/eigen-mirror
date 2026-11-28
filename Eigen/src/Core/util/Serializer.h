// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Team
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SERIALIZER_H
#define EIGEN_SERIALIZER_H

#include <type_traits>

// The Serializer class encodes data into a memory buffer so it can be later
// reconstructed. This is mainly used to send objects back-and-forth between
// the CPU and GPU.

namespace Eigen {

/**
 * Serializes an object to a memory buffer.
 *
 * Useful for transferring data (e.g. back-and-forth to a device).
 */
template <typename T, typename EnableIf = void>
class Serializer;

namespace internal {

// Zeroes the bytes of serialized Scalars that hold no part of their value. Of the built-in scalars only the padded x87
// extended precision long double has such bytes: a store writes 10 of its 12 or 16, and the others keep what the
// memory held before, which must not be serialized along with the value. A long double that is a member of Scalar is
// not cleared.
template <typename Scalar>
EIGEN_DEVICE_FUNC void clear_serialized_padding(uint8_t* dest, size_t bytes) {
#if EIGEN_ARCH_i386_OR_x86_64 && !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__)
  constexpr size_t value_bytes = 10;
  constexpr bool has_padding =
      (std::is_same<Scalar, long double>::value || std::is_same<Scalar, std::complex<long double>>::value) &&
      std::numeric_limits<long double>::digits == 64 && sizeof(long double) > value_bytes;
  constexpr size_t padding_bytes = has_padding ? sizeof(long double) - value_bytes : 0;
  EIGEN_IF_CONSTEXPR (has_padding) {
    EIGEN_USING_STD(memset)
    for (size_t i = 0; i + sizeof(long double) <= bytes; i += sizeof(long double))
      memset(dest + i + value_bytes, 0, padding_bytes);
  }
#else
  EIGEN_UNUSED_VARIABLE(dest);
  EIGEN_UNUSED_VARIABLE(bytes);
#endif
}

// deserialize() copies untrusted bytes into objects of type T, and reading a bool that holds a byte other than 0 or 1
// is undefined; the other built-in scalars can be read whatever their bytes. Returns whether the count objects stored
// at src can be read. A bool that is a member of T is not checked.
template <typename T>
EIGEN_DEVICE_FUNC bool valid_serialized_values(const uint8_t* src, Index count) {
  EIGEN_IF_CONSTEXPR (is_same<T, bool>::value) {
    for (Index i = 0; i < count; ++i) {
      if (EIGEN_PREDICT_FALSE(src[i] > 1)) return false;
    }
  }
  return true;
}

}  // namespace internal

// Specialization for POD types.
template <typename T>
class Serializer<T, std::enable_if_t<std::is_trivially_copyable<T>::value && std::is_standard_layout<T>::value>> {
 public:
  /**
   * Determines the required size of the serialization buffer for a value.
   *
   * \param value the value to serialize.
   * \return the required size.
   */
  EIGEN_DEVICE_FUNC size_t size(const T& value) const { return sizeof(value); }

  /**
   * Serializes a value to a byte buffer.
   * \param dest the destination buffer; if this is nullptr, does nothing.
   * \param end the end of the destination buffer.
   * \param value the value to serialize.
   * \return the next memory address past the end of the serialized data.
   */
  EIGEN_DEVICE_FUNC uint8_t* serialize(uint8_t* dest, uint8_t* end, const T& value) const {
    if (EIGEN_PREDICT_FALSE(dest == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(dest + sizeof(value) > end)) return nullptr;
    EIGEN_USING_STD(memcpy)
    memcpy(dest, &value, sizeof(value));
    internal::clear_serialized_padding<T>(dest, sizeof(value));
    return dest + sizeof(value);
  }

  /**
   * Deserializes a value from a byte buffer.
   * \param src the source buffer; if this is nullptr, does nothing.
   * \param end the end of the source buffer.
   * \param value the value to populate.
   * \return the next unprocessed memory address; nullptr if parsing errors are detected.
   */
  EIGEN_DEVICE_FUNC const uint8_t* deserialize(const uint8_t* src, const uint8_t* end, T& value) const {
    if (EIGEN_PREDICT_FALSE(src == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(src + sizeof(value) > end)) return nullptr;
    if (EIGEN_PREDICT_FALSE(!internal::valid_serialized_values<T>(src, 1))) return nullptr;
    EIGEN_USING_STD(memcpy)
    memcpy(&value, src, sizeof(value));
    return src + sizeof(value);
  }
};

// Specialization for DenseBase.
// Serializes [rows, cols, data...].
template <typename Derived>
class Serializer<DenseBase<Derived>, void> {
 public:
  using Scalar = typename Derived::Scalar;

  struct Header {
    typename Derived::Index rows;
    typename Derived::Index cols;
  };

  EIGEN_DEVICE_FUNC size_t size(const Derived& value) const { return sizeof(Header) + sizeof(Scalar) * value.size(); }

  EIGEN_DEVICE_FUNC uint8_t* serialize(uint8_t* dest, uint8_t* end, const Derived& value) const {
    if (EIGEN_PREDICT_FALSE(dest == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(dest + size(value) > end)) return nullptr;
    const size_t header_bytes = sizeof(Header);
    const size_t data_bytes = sizeof(Scalar) * value.size();
    Header header = {value.rows(), value.cols()};
    EIGEN_USING_STD(memcpy)
    memcpy(dest, &header, header_bytes);
    dest += header_bytes;
    // An empty object may have a null data(), which memcpy does not accept even for zero bytes.
    if (data_bytes > 0) memcpy(dest, value.data(), data_bytes);
    internal::clear_serialized_padding<Scalar>(dest, data_bytes);
    return dest + data_bytes;
  }

  EIGEN_DEVICE_FUNC const uint8_t* deserialize(const uint8_t* src, const uint8_t* end, Derived& value) const {
    if (EIGEN_PREDICT_FALSE(src == nullptr)) return nullptr;
    if (EIGEN_PREDICT_FALSE(src + sizeof(Header) > end)) return nullptr;
    const size_t header_bytes = sizeof(Header);
    Header header;
    EIGEN_USING_STD(memcpy)
    memcpy(&header, src, header_bytes);
    src += header_bytes;
    // The header is untrusted: reject sizes that are negative, that value cannot take, or that overrun the buffer.
    if (EIGEN_PREDICT_FALSE(header.rows < 0 || header.cols < 0)) return nullptr;
    if (EIGEN_PREDICT_FALSE((Derived::RowsAtCompileTime != Dynamic && header.rows != Derived::RowsAtCompileTime) ||
                            (Derived::ColsAtCompileTime != Dynamic && header.cols != Derived::ColsAtCompileTime) ||
                            (Derived::MaxRowsAtCompileTime != Dynamic && header.rows > Derived::MaxRowsAtCompileTime) ||
                            (Derived::MaxColsAtCompileTime != Dynamic && header.cols > Derived::MaxColsAtCompileTime)))
      return nullptr;
    // Only a plain object can be resized: for a Map, Ref or Block, resize() just asserts that the sizes are unchanged.
    if (EIGEN_PREDICT_FALSE(!Derived::IsPlainObjectBase &&
                            (header.rows != value.rows() || header.cols != value.cols())))
      return nullptr;
    // resize() forms rows * cols in Index, so bound the product by the largest Index as well as by the buffer.
    using Count = numext::uint64_t;
    const Count buffer_coeffs = static_cast<Count>(end - src) / sizeof(Scalar);
    const Count index_max = static_cast<Count>(NumTraits<typename Derived::Index>::highest());
    const Count max_coeffs = buffer_coeffs < index_max ? buffer_coeffs : index_max;
    if (EIGEN_PREDICT_FALSE(header.cols != 0 &&
                            static_cast<Count>(header.rows) > max_coeffs / static_cast<Count>(header.cols)))
      return nullptr;
    const size_t data_bytes = sizeof(Scalar) * header.rows * header.cols;
    if (EIGEN_PREDICT_FALSE(!internal::valid_serialized_values<Scalar>(src, header.rows * header.cols))) return nullptr;
    value.resize(header.rows, header.cols);
    EIGEN_IF_CONSTEXPR (Derived::IsPlainObjectBase) {
      if (data_bytes > 0) memcpy(value.data(), src, data_bytes);
    } else {
      // A Map, Ref or Block can have strides, so the coefficients are not one contiguous run: store them one by one,
      // in the storage order that serialize() wrote.
      const Index outer_size = Derived::IsRowMajor ? header.rows : header.cols;
      const Index inner_size = Derived::IsRowMajor ? header.cols : header.rows;
      const uint8_t* coeff = src;
      for (Index o = 0; o < outer_size; ++o) {
        for (Index i = 0; i < inner_size; ++i, coeff += sizeof(Scalar)) {
          memcpy(&value.coeffRef(Derived::IsRowMajor ? o : i, Derived::IsRowMajor ? i : o), coeff, sizeof(Scalar));
        }
      }
    }
    return src + data_bytes;
  }
};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class Serializer<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : public Serializer<DenseBase<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>> {};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class Serializer<Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
    : public Serializer<DenseBase<Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>> {};

namespace internal {

template <typename Arg>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t serialize_size_one(const Arg& arg) {
  Serializer<std::decay_t<Arg>> serializer;
  return serializer.size(arg);
}

template <typename Arg>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t* serialize_one(uint8_t* dest, uint8_t* end, const Arg& arg) {
  Serializer<std::decay_t<Arg>> serializer;
  return serializer.serialize(dest, end, arg);
}

template <typename Arg>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const uint8_t* deserialize_one(const uint8_t* src, const uint8_t* end, Arg& arg) {
  Serializer<std::decay_t<Arg>> serializer;
  return serializer.deserialize(src, end, arg);
}

}  // namespace internal

/**
 * Determine the buffer size required to serialize a set of values.
 *
 * \param args ... arguments to serialize in sequence.
 * \return the total size of the required buffer.
 */
template <typename... Args>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t serialize_size(const Args&... args) {
  size_t size = 0;
  int unused[] = {0, (size += internal::serialize_size_one(args), 0)...};
  EIGEN_UNUSED_VARIABLE(unused);
  return size;
}

/**
 * Serialize a set of values to the byte buffer.
 *
 * \param dest output byte buffer; if this is nullptr, does nothing.
 * \param end the end of the output byte buffer.
 * \param args ... arguments to serialize in sequence.
 * \return the next address after all serialized values.
 */
template <typename... Args>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint8_t* serialize(uint8_t* dest, uint8_t* end, const Args&... args) {
  EIGEN_UNUSED_VARIABLE(end);
  int unused[] = {0, (dest = internal::serialize_one(dest, end, args), 0)...};
  EIGEN_UNUSED_VARIABLE(unused);
  return dest;
}

/**
 * Deserialize a set of values from the byte buffer.
 *
 * \param src input byte buffer; if this is nullptr, does nothing.
 * \param end the end of input byte buffer.
 * \param args ... arguments to deserialize in sequence.
 * \return the next address after all parsed values; nullptr if parsing errors are detected.
 */
template <typename... Args>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const uint8_t* deserialize(const uint8_t* src, const uint8_t* end,
                                                                 Args&... args) {
  EIGEN_UNUSED_VARIABLE(end);
  int unused[] = {0, (src = internal::deserialize_one(src, end, args), 0)...};
  EIGEN_UNUSED_VARIABLE(unused);
  return src;
}

}  // namespace Eigen

#endif  // EIGEN_SERIALIZER_H

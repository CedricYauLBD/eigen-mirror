// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 The Eigen Team
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#if defined(EIGEN_TEST_PART_2)
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#endif

#include "main.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>

#if defined(__unix__)
#include <sys/mman.h>
#endif

template <typename T>
struct RandomImpl {
  static auto Create(Eigen::Index rows, Eigen::Index cols) { return T::Random(rows, cols); }
};

template <typename Scalar, int Options, typename DenseIndex>
struct RandomImpl<Eigen::SparseMatrix<Scalar, Options, DenseIndex>> {
  using T = Eigen::SparseMatrix<Scalar, Options, DenseIndex>;

  static auto Create(Eigen::Index rows, Eigen::Index cols) {
    Eigen::SparseMatrix<Scalar, Options, DenseIndex> M(rows, cols);
    M.setZero();
    double density = 0.1;

    // Reserve some space along each inner dim.
    int nnz = static_cast<int>(std::ceil(density * 1.5 * M.innerSize()));
    M.reserve(Eigen::VectorXi::Constant(M.outerSize(), nnz));

    for (int j = 0; j < M.outerSize(); j++) {
      for (int i = 0; i < M.innerSize(); i++) {
        bool zero = (Eigen::internal::random<double>(0, 1) > density);
        if (!zero) {
          M.insertByOuterInner(j, i) = internal::random<Scalar>();
        }
      }
    }

    // 50-50 whether to compress or not.
    if (Eigen::internal::random<double>(0, 1) >= 0.5) {
      M.makeCompressed();
    }

    return M;
  }
};

template <typename Scalar, int Options, typename DenseIndex>
struct RandomImpl<Eigen::SparseVector<Scalar, Options, DenseIndex>> {
  using T = Eigen::SparseVector<Scalar, Options, DenseIndex>;

  static auto Create(Eigen::Index rows, Eigen::Index cols) {
    Eigen::SparseVector<Scalar, Options, DenseIndex> M(rows, cols);
    M.setZero();
    double density = 0.1;

    // Reserve some space along each inner dim.
    int nnz = static_cast<int>(density * 1.5 * M.innerSize());
    M.reserve(nnz);

    for (int i = 0; i < M.innerSize(); i++) {
      bool zero = (Eigen::internal::random<double>(0, 1) > density);
      if (!zero) {
        M.insert(i) = internal::random<Scalar>();
      }
    }

    return M;
  }
};

struct MyPodType {
  double x;
  int y;
  float z;
};

// Plain-old-data serialization.
void test_pod_type() {
  MyPodType initial = {1.3, 17, 1.9f};
  MyPodType clone = {-1, -1, -1};

  Eigen::Serializer<MyPodType> serializer;

  // Determine required size.
  size_t buffer_size = serializer.size(initial);
  VERIFY_IS_EQUAL(buffer_size, sizeof(MyPodType));

  // Serialize.
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* begin = buffer.data();
  uint8_t* end = buffer.data() + buffer.size();
  uint8_t* dest = serializer.serialize(begin, end, initial);
  VERIFY(dest != nullptr);
  VERIFY_IS_EQUAL(dest - begin, buffer_size);

  // Deserialize.
  const uint8_t* src = serializer.deserialize(begin, end, clone);
  VERIFY(src != nullptr);
  VERIFY_IS_EQUAL(src - begin, buffer_size);
  VERIFY_IS_EQUAL(clone.x, initial.x);
  VERIFY_IS_EQUAL(clone.y, initial.y);
  VERIFY_IS_EQUAL(clone.z, initial.z);

  // Serialize with bounds checking errors.
  dest = serializer.serialize(begin, end - 1, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(begin, begin, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(nullptr, nullptr, initial);
  VERIFY(dest == nullptr);

  // Deserialize with bounds checking errors.
  src = serializer.deserialize(begin, end - 1, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(begin, begin, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(nullptr, nullptr, clone);
  VERIFY(src == nullptr);
}

// Matrix, Vector, Array
template <typename T>
void test_eigen_type(const T& type) {
  const Index rows = type.rows();
  const Index cols = type.cols();

  const T initial = RandomImpl<T>::Create(rows, cols);

  // Serialize.
  Eigen::Serializer<T> serializer;
  size_t buffer_size = serializer.size(initial);
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* begin = buffer.data();
  uint8_t* end = buffer.data() + buffer.size();
  uint8_t* dest = serializer.serialize(begin, end, initial);
  VERIFY(dest != nullptr);
  VERIFY_IS_EQUAL(dest - begin, buffer_size);

  // Deserialize.
  T clone;
  const uint8_t* src = serializer.deserialize(begin, end, clone);
  VERIFY(src != nullptr);
  VERIFY_IS_EQUAL(src - begin, buffer_size);
  VERIFY_IS_CWISE_EQUAL(clone, initial);

  // Serialize with bounds checking errors.
  dest = serializer.serialize(begin, end - 1, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(begin, begin, initial);
  VERIFY(dest == nullptr);
  dest = serializer.serialize(nullptr, nullptr, initial);
  VERIFY(dest == nullptr);

  // Deserialize with bounds checking errors.
  src = serializer.deserialize(begin, end - 1, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(begin, begin, clone);
  VERIFY(src == nullptr);
  src = serializer.deserialize(nullptr, nullptr, clone);
  VERIFY(src == nullptr);
}

// A dense header whose sizes are negative, wrong for the type, or larger than the buffer is a parsing error.
template <typename T>
void test_malformed_dense_header(Index rows, Index cols) {
  struct Header {
    Index rows;
    Index cols;
  };
  std::vector<uint8_t> buffer(sizeof(Header) + 16 * sizeof(typename T::Scalar));
  const Header header = {rows, cols};
  std::memcpy(buffer.data(), &header, sizeof(header));
  T value;
  VERIFY(Eigen::Serializer<T>().deserialize(buffer.data(), buffer.data() + buffer.size(), value) == nullptr);
}

void test_malformed_dense_headers() {
  test_malformed_dense_header<Matrix2d>(4, 4);
  test_malformed_dense_header<Matrix2d>(1, 4);
  test_malformed_dense_header<Matrix<double, Dynamic, Dynamic, 0, 2, 2>>(4, 1);
  test_malformed_dense_header<MatrixXd>(-1, -4);
  test_malformed_dense_header<MatrixXd>(5, 4);
  test_malformed_dense_header<MatrixXd>(Index(1) << (8 * sizeof(Index) - 2), 4);
  test_malformed_dense_header<VectorXd>(4, 4);
}

// A Map, Ref or Block cannot be resized, so a header with other sizes than the destination's is a parsing error.
// `dest` views the leading coefficients of `storage`: a rejected buffer must leave all of `storage` as it was.
template <typename Dest>
void test_dense_view_destination(VectorXd& storage, Dest dest) {
  Serializer<DenseBase<Dest>> serializer;
  for (Index rows_delta = -1; rows_delta <= 1; ++rows_delta) {
    for (Index cols_delta = -1; cols_delta <= 1; ++cols_delta) {
      const MatrixXd sent = MatrixXd::Random(dest.rows() + rows_delta, dest.cols() + cols_delta);
      std::vector<uint8_t> buffer(Eigen::serialize_size(sent));
      uint8_t* begin = buffer.data();
      uint8_t* end = begin + buffer.size();
      VERIFY(Eigen::serialize(begin, end, sent) == end);
      const bool accepted = rows_delta == 0 && cols_delta == 0;
      VectorXd expected = storage;
      if (accepted) expected.head(sent.size()) = Map<const VectorXd>(sent.data(), sent.size());
      VERIFY(serializer.deserialize(begin, end, dest) == (accepted ? end : nullptr));
      VERIFY_IS_CWISE_EQUAL(storage, expected);
    }
  }
}

void test_dense_view_destinations() {
  VectorXd storage = VectorXd::Random(12);
  test_dense_view_destination(storage, Map<VectorXd>(storage.data(), 4));
  test_dense_view_destination(storage, Map<MatrixXd>(storage.data(), 2, 3));
  test_dense_view_destination(storage, Ref<VectorXd>(storage.head(4)));
  test_dense_view_destination(storage, storage.head(4));
}

// A view with strides is not one contiguous run of coefficients: the payload must land in the view, and no coefficient
// of the backing matrix outside it may change.
template <int Options>
void test_strided_dense_view_destinations() {
  using Backing = Matrix<double, Dynamic, Dynamic, Options>;
  const Backing original = Backing::Random(5, 6);
  const Matrix<double, Dynamic, Dynamic, Options> sent = Matrix<double, Dynamic, Dynamic, Options>::Random(2, 3);
  std::vector<uint8_t> buffer(Eigen::serialize_size(sent));
  uint8_t* begin = buffer.data();
  uint8_t* end = begin + buffer.size();
  VERIFY(Eigen::serialize(begin, end, sent) == end);

  Backing backing = original;
  auto block = backing.block(1, 2, 2, 3);
  Backing expected = original;
  expected.block(1, 2, 2, 3) = sent;
  VERIFY(Serializer<DenseBase<decltype(block)>>().deserialize(begin, end, block) == end);
  VERIFY_IS_CWISE_EQUAL(backing, expected);

  backing = original;
  Map<Backing, 0, Stride<Dynamic, Dynamic>> map(backing.data() + 1, 2, 3, Stride<Dynamic, Dynamic>(10, 3));
  Backing expected_map = original;
  Map<Backing, 0, Stride<Dynamic, Dynamic>>(expected_map.data() + 1, 2, 3, Stride<Dynamic, Dynamic>(10, 3)) = sent;
  VERIFY(Serializer<DenseBase<decltype(map)>>().deserialize(begin, end, map) == end);
  VERIFY_IS_CWISE_EQUAL(backing, expected_map);

  // A rejected buffer leaves the whole backing matrix as it was.
  backing = original;
  auto wrong = backing.block(1, 2, 3, 2);
  VERIFY(Serializer<DenseBase<decltype(wrong)>>().deserialize(begin, end, wrong) == nullptr);
  VERIFY_IS_CWISE_EQUAL(backing, original);
}

// Serializes `valid` into a buffer with room to spare, applies `corrupt` to the header and payload, and checks that
// deserializing the result is a parsing error that leaves no entries in the destination.
template <typename T, typename Corrupt>
void verify_rejected(const T& valid, Corrupt corrupt) {
  using Header = typename Serializer<T>::Header;
  Serializer<T> serializer;
  std::vector<uint8_t> buffer(serializer.size(valid) + 256);
  uint8_t* begin = buffer.data();
  uint8_t* end = begin + buffer.size();
  VERIFY(serializer.serialize(begin, end, valid) != nullptr);
  Header header;
  std::memcpy(&header, begin, sizeof(Header));
  corrupt(header, begin + sizeof(Header));
  std::memcpy(begin, &header, sizeof(Header));
  T value;
  VERIFY(serializer.deserialize(begin, end, value) == nullptr);
  VERIFY_IS_EQUAL(value.nonZeros(), 0);
}

template <typename StorageIndex>
void set_index(uint8_t* array, Index i, StorageIndex v) {
  std::memcpy(array + i * sizeof(StorageIndex), &v, sizeof(v));
}

// Every header field and stored index of a sparse matrix decides where deserialize() writes or what later reads
// address, so a buffer that does not describe a valid matrix must be rejected.
template <int Options>
void test_malformed_sparse_matrix() {
  using T = SparseMatrix<float, Options, int>;
  using Header = typename Serializer<T>::Header;
  // static: MSVC 19.29 wants a captureless lambda to capture even a constexpr local.
  static constexpr bool kRowMajor = T::IsRowMajor;
  // 3 x 2 with entries (0, 0), (2, 0), (1, 1): compressed, the payload is outer[3], inner[3], values[3] in column-major
  // order, and outer[4], inner[3], values[3] in row-major order.
  T m(3, 2);
  m.insert(0, 0) = 1;
  m.insert(2, 0) = 2;
  m.insert(1, 1) = 3;
  m.makeCompressed();
  const Index outer_size = m.outerSize();
  const auto inner = [outer_size](uint8_t* payload) { return payload + (outer_size + 1) * sizeof(int); };

  verify_rejected(T(1, 1), [](Header& h, uint8_t*) { h.outer_size = 10; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.outer_size = kRowMajor ? h.cols : h.rows; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.rows = -1; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.cols = -1; });
  // No array backs the inner size, so only rows * cols bounds it: a product that wraps would be the size of a small
  // dense matrix. The StorageIndex bounds the stored indices only, so the largest shape that does not wrap is accepted.
  if (sizeof(Index) == 8) {
    const Index max_inner = NumTraits<Index>::highest() / outer_size;
    verify_rejected(m, [max_inner](Header& h, uint8_t*) { (kRowMajor ? h.cols : h.rows) = max_inner + 1; });
    Serializer<T> tall_serializer;
    std::vector<uint8_t> tall_buffer(tall_serializer.size(m));
    uint8_t* tall_begin = tall_buffer.data();
    VERIFY(tall_serializer.serialize(tall_begin, tall_begin + tall_buffer.size(), m) != nullptr);
    Header tall_header;
    std::memcpy(&tall_header, tall_begin, sizeof(Header));
    (kRowMajor ? tall_header.cols : tall_header.rows) = max_inner;
    std::memcpy(tall_begin, &tall_header, sizeof(Header));
    T tall;
    VERIFY(tall_serializer.deserialize(tall_begin, tall_begin + tall_buffer.size(), tall) != nullptr);
    VERIFY_IS_EQUAL(tall.innerSize(), max_inner);
    VERIFY_IS_EQUAL(tall.outerSize(), outer_size);
    VERIFY_IS_EQUAL(tall.nonZeros(), m.nonZeros());
  }
  verify_rejected(m, [](Header& h, uint8_t*) { h.inner_buffer_size = -1; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.inner_buffer_size = 4; });
  verify_rejected(m, [](Header& h, uint8_t*) { h.inner_buffer_size = Index(1) << 30; });
  verify_rejected(m, [](Header& h, uint8_t*) {
    const unsigned char two = 2;
    std::memcpy(&h.compressed, &two, 1);
  });
  verify_rejected(m, [](Header&, uint8_t* p) { set_index(p, 0, 1); });
  verify_rejected(m, [](Header&, uint8_t* p) { set_index(p, 1, 4); });
  verify_rejected(m, [&](Header&, uint8_t* p) { set_index(p, outer_size, 2); });
  verify_rejected(m, [&](Header&, uint8_t* p) { set_index(inner(p), 1, kRowMajor ? 2 : 3); });
  verify_rejected(m, [&](Header&, uint8_t* p) { set_index(inner(p), 2, -1); });

  // Uncompressed, the payload starts with innerNonZeros[outer_size], and the unused entries are not checked.
  T u = m;
  u.reserve(Eigen::VectorXi::Constant(outer_size, 3));
  VERIFY(!u.isCompressed());
  const int capacity0 = u.outerIndexPtr()[1] - u.outerIndexPtr()[0];
  verify_rejected(u, [&](Header&, uint8_t* p) { set_index(p, 0, capacity0 + 1); });
  verify_rejected(u, [&](Header&, uint8_t* p) { set_index(p, 0, -1); });
  Serializer<T> serializer;
  std::vector<uint8_t> buffer(serializer.size(u));
  VERIFY(serializer.serialize(buffer.data(), buffer.data() + buffer.size(), u) != nullptr);
  uint8_t* payload = buffer.data() + sizeof(Header);
  const int* outer = u.outerIndexPtr();
  const int* inner_non_zeros = u.innerNonZeroPtr();
  for (Index j = 0; j < outer_size; ++j) {
    for (Index k = outer[j] + inner_non_zeros[j]; k < outer[j + 1]; ++k)
      set_index(payload + (2 * outer_size + 1) * sizeof(int), k, -7);
  }
  T clone;
  VERIFY(serializer.deserialize(buffer.data(), buffer.data() + buffer.size(), clone) != nullptr);
  VERIFY_IS_CWISE_EQUAL(MatrixXf(clone), MatrixXf(m));
}

// The rest of the API relies on inner vectors that are sorted and free of duplicates: an unsorted one is sorted on
// deserialization, a repeated inner index is a parsing error.
template <int Options>
void test_unsorted_sparse_matrix() {
  using T = SparseMatrix<float, Options, int>;
  using Header = typename Serializer<T>::Header;
  const Index n = 4;
  const Index outer = 1;
  // Inner vector 1 stores the values 1, 2, 3 at the inner indices 0, 1, 2.
  T m(n, n);
  for (Index i = 0; i < 3; ++i) m.insert(T::IsRowMajor ? outer : i, T::IsRowMajor ? i : outer) = float(i + 1);
  for (bool compressed : {true, false}) {
    if (compressed)
      m.makeCompressed();
    else
      m.reserve(VectorXi::Constant(n, 2));
    VERIFY_IS_EQUAL(m.isCompressed(), compressed);
    // The position of the first inner index of that vector in the payload, which starts with the outer index arrays.
    const Index first = (compressed ? 0 : n) + (n + 1) + m.outerIndexPtr()[outer];

    Serializer<T> serializer;
    std::vector<uint8_t> buffer(serializer.size(m));
    VERIFY(serializer.serialize(buffer.data(), buffer.data() + buffer.size(), m) != nullptr);
    uint8_t* payload = buffer.data() + sizeof(Header);
    for (int k = 0; k < 3; ++k) set_index(payload, first + k, 2 - k);
    T clone;
    VERIFY(serializer.deserialize(buffer.data(), buffer.data() + buffer.size(), clone) != nullptr);
    VERIFY_IS_EQUAL(clone.isCompressed(), compressed);
    VERIFY_IS_EQUAL(clone.nonZeros(), Index(3));
    VERIFY_IS_EQUAL(clone.innerIndicesAreSorted(), clone.outerSize());
    for (Index i = 0; i < 3; ++i) {
      VERIFY_IS_EQUAL(clone.coeff(T::IsRowMajor ? outer : i, T::IsRowMajor ? i : outer), float(3 - i));
    }
    // A block that starts at the last inner index holds the value that was stored first and nothing else.
    const MatrixXf last = T::IsRowMajor ? MatrixXf(clone.middleCols(2, 1)) : MatrixXf(clone.middleRows(2, 1));
    VERIFY_IS_EQUAL(last.sum(), 1.f);

    verify_rejected(m, [first](Header&, uint8_t* p) { set_index(p, first + 1, 0); });
    verify_rejected(m, [first](Header&, uint8_t* p) { set_index(p, first + 2, 0); });
    verify_rejected(m, [first](Header&, uint8_t* p) { set_index(p, first, 2); });
  }
}

void test_malformed_sparse_vector() {
  using T = SparseVector<float, 0, int>;
  using Header = Serializer<T>::Header;
  T v(4);
  v.insert(1) = 1;
  v.insert(3) = 2;
  verify_rejected(v, [](Header& h, uint8_t*) { h.size = -1; });
  verify_rejected(v, [](Header& h, uint8_t*) { h.num_non_zeros = -1; });
  verify_rejected(v, [](Header& h, uint8_t*) { h.num_non_zeros = Index(1) << 30; });
  verify_rejected(v, [](Header&, uint8_t* p) { set_index(p, 1, 4); });
  verify_rejected(v, [](Header&, uint8_t* p) { set_index(p, 0, -1); });
}

void test_unsorted_sparse_vector() {
  using T = SparseVector<float, 0, int>;
  using Header = Serializer<T>::Header;
  T v(4);
  v.insert(0) = 1;
  v.insert(1) = 2;
  v.insert(3) = 3;

  Serializer<T> serializer;
  std::vector<uint8_t> buffer(serializer.size(v));
  VERIFY(serializer.serialize(buffer.data(), buffer.data() + buffer.size(), v) != nullptr);
  uint8_t* payload = buffer.data() + sizeof(Header);
  set_index(payload, 0, 3);
  set_index(payload, 2, 0);
  T clone;
  VERIFY(serializer.deserialize(buffer.data(), buffer.data() + buffer.size(), clone) != nullptr);
  VERIFY_IS_EQUAL(clone.nonZeros(), Index(3));
  VERIFY_IS_EQUAL(clone.innerIndicesAreSorted(), Index(1));
  VERIFY_IS_CWISE_EQUAL(VectorXf(clone), Vector4f(3, 2, 0, 1));
  VERIFY_IS_CWISE_EQUAL(VectorXf(clone.tail(1)), VectorXf::Constant(1, 1.f));

  verify_rejected(v, [](Header&, uint8_t* p) { set_index(p, 1, 0); });
  verify_rejected(v, [](Header&, uint8_t* p) { set_index(p, 0, 3); });
}

// Reading a bool that holds a byte other than 0 or 1 is undefined, so such a byte is a parsing error wherever
// deserialize() would store it in a coefficient that can be read.
void test_malformed_bool() {
  const uint8_t bytes[] = {0, 1, 2};
  bool flag = false;
  VERIFY(Eigen::deserialize(bytes + 1, bytes + 2, flag) == bytes + 2);
  VERIFY(flag);
  VERIFY(Eigen::deserialize(bytes + 2, bytes + 3, flag) == nullptr);

  using Mask = Array<bool, Dynamic, 1>;
  Mask mask(8);
  mask << true, false, false, true, true, false, true, false;
  Mask mask_clone;
  std::vector<uint8_t> mask_buffer(Eigen::serialize_size(mask));
  uint8_t* mask_end = mask_buffer.data() + mask_buffer.size();
  VERIFY(Eigen::serialize(mask_buffer.data(), mask_end, mask) == mask_end);
  VERIFY(Eigen::deserialize(mask_buffer.data(), mask_end, mask_clone) == mask_end);
  VERIFY_IS_CWISE_EQUAL(mask_clone, mask);
  uint8_t* coeffs = mask_buffer.data() + sizeof(Serializer<Mask>::Header);
  coeffs[0] = 0xFF;
  VERIFY(Eigen::deserialize(mask_buffer.data(), mask_end, mask_clone) == nullptr);
  coeffs[0] = 1;
  coeffs[7] = 2;
  VERIFY(Eigen::deserialize(mask_buffer.data(), mask_end, mask_clone) == nullptr);
  VERIFY_IS_CWISE_EQUAL(mask_clone, mask);

  // The payload is inner[2], values[2].
  using Vector = SparseVector<bool, 0, int>;
  Vector v(4);
  v.insert(1) = true;
  v.insert(3) = true;
  verify_rejected(v, [](Serializer<Vector>::Header&, uint8_t* p) { p[2 * sizeof(int) + 1] = 2; });

  // 3 x 3 with entries (0, 0) and (2, 1): compressed, the payload is outer[4], inner[2], values[2].
  using T = SparseMatrix<bool, ColMajor, int>;
  using Header = Serializer<T>::Header;
  T m(3, 3);
  m.insert(0, 0) = true;
  m.insert(2, 1) = true;
  m.makeCompressed();
  verify_rejected(m, [](Header&, uint8_t* p) { p[6 * sizeof(int) + 1] = 2; });

  // Uncompressed, the payload is innerNonZeros[3], outer[4], inner[capacity], values[capacity]. The unused entries are
  // zeros that nothing has written, but a sender other than serialize() may put any byte there.
  T u = m;
  u.reserve(Eigen::VectorXi::Constant(3, 2));
  VERIFY(!u.isCompressed());
  const int* outer = u.outerIndexPtr();
  const int* inner_non_zeros = u.innerNonZeroPtr();
  const std::size_t values = (7 + outer[3]) * sizeof(int);
  const int used = outer[1];
  VERIFY_IS_EQUAL(inner_non_zeros[1], 1);
  verify_rejected(u, [=](Header&, uint8_t* p) { p[values + used] = 2; });
  Serializer<T> serializer;
  std::vector<uint8_t> buffer(serializer.size(u));
  VERIFY(serializer.serialize(buffer.data(), buffer.data() + buffer.size(), u) != nullptr);
  uint8_t* payload = buffer.data() + sizeof(Header);
  int unused = 0;
  for (int j = 0; j < 3; ++j) {
    for (int k = outer[j] + inner_non_zeros[j]; k < outer[j + 1]; ++k, ++unused) payload[values + k] = 2;
  }
  VERIFY(unused > 0);
  T clone;
  VERIFY(serializer.deserialize(buffer.data(), buffer.data() + buffer.size(), clone) != nullptr);
  VERIFY_IS_CWISE_EQUAL(clone.toDense(), m.toDense());
}

// serialize() writes the reserved slots of an uncompressed matrix as zeros: it copies only the live entries, so the
// bytes the memory held before do not reach the buffer.
template <int Options>
void test_uncompressed_unused_slots() {
  using T = SparseMatrix<float, Options, int>;
  T m(4, 4);
  m.insert(0, 0) = 1;
  m.insert(2, 0) = 2;
  m.insert(1, 1) = 3;
  m.insert(3, 3) = 4;
  m.makeCompressed();
  T u = m;
  u.reserve(Eigen::VectorXi::Constant(4, 3));
  VERIFY(!u.isCompressed());
  // Poison the reserved slots.
  const int* outer = u.outerIndexPtr();
  const int* inner_non_zeros = u.innerNonZeroPtr();
  for (int j = 0; j < 4; ++j) {
    for (int k = outer[j] + inner_non_zeros[j]; k < outer[j + 1]; ++k) {
      u.innerIndexPtr()[k] = 0x5A5A5A5A;
      u.valuePtr()[k] = 12345.f;
    }
  }
  Serializer<T> serializer;
  std::vector<uint8_t> buffer(serializer.size(u));
  VERIFY(serializer.serialize(buffer.data(), buffer.data() + buffer.size(), u) != nullptr);
  const std::size_t capacity = outer[4];
  const uint8_t* payload = buffer.data() + sizeof(typename Serializer<T>::Header);
  const uint8_t* inner_bytes = payload + (4 + 5) * sizeof(int);
  const uint8_t* value_bytes = inner_bytes + capacity * sizeof(int);
  int unused = 0;
  for (int j = 0; j < 4; ++j) {
    for (int k = outer[j]; k < outer[j + 1]; ++k) {
      int index;
      float value;
      std::memcpy(&index, inner_bytes + k * sizeof(int), sizeof(int));
      std::memcpy(&value, value_bytes + k * sizeof(float), sizeof(float));
      if (k < outer[j] + inner_non_zeros[j]) {
        VERIFY_IS_EQUAL(index, u.innerIndexPtr()[k]);
        VERIFY_IS_EQUAL(value, u.valuePtr()[k]);
      } else {
        VERIFY_IS_EQUAL(index, 0);
        VERIFY_IS_EQUAL(value, 0.f);
        ++unused;
      }
    }
  }
  VERIFY(unused > 0);
  T clone;
  VERIFY(serializer.deserialize(buffer.data(), buffer.data() + buffer.size(), clone) != nullptr);
  VERIFY_IS_CWISE_EQUAL(clone.toDense(), m.toDense());
}

#if defined(EIGEN_TEST_PART_2) && defined(__unix__) && defined(MAP_NORESERVE) && SIZE_MAX > 0xFFFFFFFFu
// 65536 x 65536 chars fit in a 4 GiB buffer, but resize() forms rows * cols in the 32-bit Index, where it overflows.
// The mapping only reserves address space: the header must be rejected before the payload is read.
void test_dense_header_index_overflow() {
  STATIC_CHECK(sizeof(Index) == 4);
  using T = Matrix<char, Dynamic, Dynamic>;
  using Header = Serializer<T>::Header;
  const std::size_t bytes = sizeof(Header) + (std::size_t(1) << 32);
  void* map = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  VERIFY(map != MAP_FAILED);
  const Header header = {65536, 65536};
  std::memcpy(map, &header, sizeof(header));
  const uint8_t* begin = static_cast<const uint8_t*>(map);
  T value;
  VERIFY(Serializer<T>().deserialize(begin, begin + bytes, value) == nullptr);
  VERIFY_IS_EQUAL(munmap(map, bytes), 0);
}
#endif

// Fills the scalars with a byte pattern and then assigns each of them: what an assignment does not write, such as the
// padding of an x87 long double, keeps the pattern.
template <typename Scalar>
void assign_over_pattern(Scalar* data, Index size, int pattern) {
  std::memset(static_cast<void*>(data), pattern, sizeof(Scalar) * size);
  for (Index i = 0; i < size; ++i) data[i] = Scalar(static_cast<typename NumTraits<Scalar>::Real>(i + 1));
}

// Serializes into a buffer that holds the pattern as well.
template <typename T>
std::vector<uint8_t> serialize_over_pattern(const T& value, int pattern) {
  std::vector<uint8_t> buffer(Eigen::serialize_size(value), static_cast<uint8_t>(pattern));
  VERIFY(Eigen::serialize(buffer.data(), buffer.data() + buffer.size(), value) == buffer.data() + buffer.size());
  return buffer;
}

// Equal values serialize to equal bytes: the padding of the scalars and of the headers must not carry what the memory
// of the value, the stack or the buffer held before.
template <typename Scalar>
void test_padding() {
  std::vector<uint8_t> pod[2], dense[2], sparse_matrix[2], sparse_vector[2];
  for (int k = 0; k < 2; ++k) {
    const int pattern = k == 0 ? 0x55 : 0xaa;

    Scalar x;
    assign_over_pattern(&x, 1, pattern);
    pod[k] = serialize_over_pattern(x, pattern);

    Matrix<Scalar, Dynamic, Dynamic> m(4, 2);
    assign_over_pattern(m.data(), m.size(), pattern);
    dense[k] = serialize_over_pattern(m, pattern);

    SparseMatrix<Scalar> s(4, 4);
    for (Index i = 0; i < 4; ++i) s.insert(i, (i + 1) % 4) = Scalar(1);
    s.makeCompressed();
    assign_over_pattern(s.valuePtr(), s.nonZeros(), pattern);
    sparse_matrix[k] = serialize_over_pattern(s, pattern);

    SparseVector<Scalar> v(8);
    for (Index i = 0; i < 8; i += 2) v.insert(i) = Scalar(1);
    assign_over_pattern(v.valuePtr(), v.nonZeros(), pattern);
    sparse_vector[k] = serialize_over_pattern(v, pattern);
  }
  VERIFY(pod[0] == pod[1]);
  VERIFY(dense[0] == dense[1]);
  VERIFY(sparse_matrix[0] == sparse_matrix[1]);
  VERIFY(sparse_vector[0] == sparse_vector[1]);
}

// Test a collection of dense types.
template <typename T1, typename T2, typename T3>
void test_dense_types(const T1& type1, const T2& type2, const T3& type3) {
  // Make random inputs.
  const T1 x1 = T1::Random(type1.rows(), type1.cols());
  const T2 x2 = T2::Random(type2.rows(), type2.cols());
  const T3 x3 = T3::Random(type3.rows(), type3.cols());

  // Allocate buffer and serialize.
  size_t buffer_size = Eigen::serialize_size(x1, x2, x3);
  std::vector<uint8_t> buffer(buffer_size);
  uint8_t* begin = buffer.data();
  uint8_t* end = buffer.data() + buffer.size();
  uint8_t* dest = Eigen::serialize(begin, end, x1, x2, x3);
  VERIFY(dest != nullptr);

  // Clone everything.
  T1 y1;
  T2 y2;
  T3 y3;
  const uint8_t* src = Eigen::deserialize(begin, end, y1, y2, y3);
  VERIFY(src != nullptr);

  // Verify they equal.
  VERIFY_IS_CWISE_EQUAL(y1, x1);
  VERIFY_IS_CWISE_EQUAL(y2, x2);
  VERIFY_IS_CWISE_EQUAL(y3, x3);

  // Serialize everything with bounds checking errors.
  dest = Eigen::serialize(begin, end - 1, y1, y2, y3);
  VERIFY(dest == nullptr);
  dest = Eigen::serialize(begin, begin, y1, y2, y3);
  VERIFY(dest == nullptr);
  dest = Eigen::serialize(nullptr, nullptr, y1, y2, y3);
  VERIFY(dest == nullptr);

  // Deserialize everything with bounds checking errors.
  src = Eigen::deserialize(begin, end - 1, y1, y2, y3);
  VERIFY(src == nullptr);
  src = Eigen::deserialize(begin, begin, y1, y2, y3);
  VERIFY(src == nullptr);
  src = Eigen::deserialize(nullptr, nullptr, y1, y2, y3);
  VERIFY(src == nullptr);
}

EIGEN_DECLARE_TEST(serializer) {
  CALL_SUBTEST_1(test_pod_type());
  CALL_SUBTEST_1(test_padding<double>());
  CALL_SUBTEST_1(test_padding<long double>());
  CALL_SUBTEST_1(test_padding<std::complex<long double>>());
  CALL_SUBTEST_1(test_malformed_dense_headers());
  CALL_SUBTEST_1(test_dense_view_destinations());
  CALL_SUBTEST_1(test_strided_dense_view_destinations<ColMajor>());
  CALL_SUBTEST_1(test_strided_dense_view_destinations<RowMajor>());
  CALL_SUBTEST_1(test_uncompressed_unused_slots<ColMajor>());
  CALL_SUBTEST_1(test_uncompressed_unused_slots<RowMajor>());
  CALL_SUBTEST_1(test_malformed_sparse_matrix<ColMajor>());
  CALL_SUBTEST_1(test_malformed_sparse_matrix<RowMajor>());
  CALL_SUBTEST_1(test_unsorted_sparse_matrix<ColMajor>());
  CALL_SUBTEST_1(test_unsorted_sparse_matrix<RowMajor>());
  CALL_SUBTEST_1(test_malformed_sparse_vector());
  CALL_SUBTEST_1(test_unsorted_sparse_vector());
  CALL_SUBTEST_1(test_malformed_bool());

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(test_eigen_type(Eigen::Array33f()));
    CALL_SUBTEST_1(test_eigen_type(Eigen::ArrayXd(10)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::Vector3f()));
    CALL_SUBTEST_1(test_eigen_type(Eigen::Matrix4d()));
    CALL_SUBTEST_1(test_eigen_type(Eigen::MatrixXd(15, 17)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::SparseMatrix<float>(13, 12)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::SparseMatrix<float>(0, 3)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::SparseVector<float>(17)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>(7, 5)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::SparseMatrix<std::complex<long double>>(13, 12)));
    CALL_SUBTEST_1(test_eigen_type(Eigen::SparseVector<long double>(17)));

    CALL_SUBTEST_1(test_dense_types(Eigen::Array33f(), Eigen::ArrayXd(10), Eigen::MatrixXd(15, 17)));
  }

  CALL_SUBTEST_2(test_malformed_dense_headers());
  CALL_SUBTEST_2(test_dense_view_destinations());
#if defined(EIGEN_TEST_PART_2) && defined(__unix__) && defined(MAP_NORESERVE) && SIZE_MAX > 0xFFFFFFFFu
  CALL_SUBTEST_2(test_dense_header_index_overflow());
#endif
}

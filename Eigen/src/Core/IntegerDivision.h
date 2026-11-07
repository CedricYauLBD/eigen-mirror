// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Charlie Schlosser <cs.schlosser@gmail.com>
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_INTEGER_DIVISION_H
#define EIGEN_INTEGER_DIVISION_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// Division of k-bit unsigned integers by an invariant divisor 1 <= d < 2^k, after Granlund and Montgomery, "Division
// by invariant integers using multiplication", PLDI 1994, Fig. 4.1. With
//
//   l = ceil(log2(d)),   m = floor(2^k (2^l - d) / d) + 1,   1 <= m < 2^k,
//
// every 0 <= n < 2^k satisfies, with t = floor(m n / 2^k),
//
//   floor(n / d) = (t + n) >> l = (t + ((n - t) >> 1)) >> (l - 1).
//
// The first form needs k + 1 bits; the second never overflows (t <= n) but needs l >= 1. Splitting the shift as
// s1 = min(l, 1), s2 = l - s1 covers d = 1 too (l = 0, m = 1, t = 0), so one branch-free sequence serves every d.
// A divisor d >= 2^k is stored as m = 1, l = k, for which both forms return 0.
namespace int_div {

template <typename T>
EIGEN_DEVICE_FUNC constexpr bool is_negative(const T& x, std::true_type) {
  return x < 0;
}
template <typename T>
EIGEN_DEVICE_FUNC constexpr bool is_negative(const T&, std::false_type) {
  return false;
}
template <typename T>
EIGEN_DEVICE_FUNC constexpr bool is_negative(const T& x) {
  return is_negative(x, std::is_signed<T>());
}

// |x| as the unsigned type of the same width; exact for lowest().
template <typename T>
EIGEN_DEVICE_FUNC constexpr std::make_unsigned_t<T> unsigned_abs(const T& x) {
  using Unsigned = std::make_unsigned_t<T>;
  return is_negative(x) ? static_cast<Unsigned>(Unsigned(0) - static_cast<Unsigned>(x)) : static_cast<Unsigned>(x);
}

// floor(a b / 2^k).
template <typename Unsigned, std::enable_if_t<(sizeof(Unsigned) < 8), int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned umulhi(Unsigned a, Unsigned b) {
  return static_cast<Unsigned>((static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) >> (CHAR_BIT * sizeof(Unsigned)));
}
template <typename Unsigned, std::enable_if_t<sizeof(Unsigned) == 8, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned umulhi(Unsigned a, Unsigned b) {
#if defined(EIGEN_GPU_COMPILE_PHASE)
  return __umul64hi(a, b);
#elif defined(EIGEN_SYCL_DEVICE_ONLY)
  return cl::sycl::mul_hi(static_cast<uint64_t>(a), static_cast<uint64_t>(b));
#elif EIGEN_COMP_MSVC && (EIGEN_ARCH_x86_64 || EIGEN_ARCH_ARM64)
  return __umulh(a, b);
#elif EIGEN_HAS_BUILTIN_INT128
  return static_cast<Unsigned>((static_cast<__uint128_t>(a) * b) >> 64);
#else
  // Hacker's Delight, 2nd ed., Fig. 8-2: with 32-bit digits, u and w stay below 2^64.
  const Unsigned a0 = a & 0xffffffffu, a1 = a >> 32, b0 = b & 0xffffffffu, b1 = b >> 32;
  const Unsigned u = a1 * b0 + ((a0 * b0) >> 32);
  const Unsigned w = a0 * b1 + (u & 0xffffffffu);
  return a1 * b1 + (u >> 32) + (w >> 32);
#endif
}

// floor(r 2^k / d) for r < d, which fits in k bits.
template <typename Unsigned, std::enable_if_t<(sizeof(Unsigned) < 8), int> = 0>
EIGEN_DEVICE_FUNC Unsigned div_shifted(Unsigned r, Unsigned d) {
  return static_cast<Unsigned>((static_cast<uint64_t>(r) << (CHAR_BIT * sizeof(Unsigned))) / d);
}
template <typename Unsigned, std::enable_if_t<sizeof(Unsigned) == 8, int> = 0>
EIGEN_DEVICE_FUNC Unsigned div_shifted(Unsigned r, Unsigned d) {
#if EIGEN_HAS_BUILTIN_INT128 && !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(__SYCL_DEVICE_ONLY__)
  return static_cast<Unsigned>((static_cast<__uint128_t>(r) << 64) / d);
#else
  // Restoring division, one quotient bit per step, keeping r < d.
  Unsigned q = 0;
  for (int i = 0; i < 64; ++i) {
    const bool carry = (r >> 63) != 0;
    r <<= 1;
    q <<= 1;
    if (carry || r >= d) {
      r -= d;
      q |= 1;
    }
  }
  return q;
#endif
}

// floor(a / d) for the (magic, shift) = (m, l) of d: the first form in 64-bit arithmetic, else the second.
template <typename Unsigned, std::enable_if_t<(sizeof(Unsigned) < 8), int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned udiv(Unsigned a, Unsigned magic, int shift) {
  return static_cast<Unsigned>((static_cast<uint64_t>(umulhi(a, magic)) + a) >> shift);
}
template <typename Unsigned, std::enable_if_t<sizeof(Unsigned) == 8, int> = 0>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Unsigned udiv(Unsigned a, Unsigned magic, int shift) {
  const Unsigned t = umulhi(a, magic);
  const int shift1 = shift > 0 ? 1 : 0;
  return (t + ((a - t) >> shift1)) >> (shift - shift1);
}

template <typename Unsigned>
struct params {
  static_assert(std::is_unsigned<Unsigned>::value, "params takes the unsigned type of the numerators");
  static constexpr int kBits = CHAR_BIT * sizeof(Unsigned);

  template <typename UnsignedDivisor>
  EIGEN_DEVICE_FUNC explicit params(UnsignedDivisor d) : magic(1), shift(kBits) {
    eigen_assert(d != 0 && "IntDivider: division by zero");
    // A wider divisor type can hold d >= 2^k, which keeps the defaults above.
    if (d <= static_cast<UnsignedDivisor>(NumTraits<Unsigned>::highest())) {
      const Unsigned du = static_cast<Unsigned>(d);
      shift = log_2_impl<Unsigned>::run_ceil(du);
      // r = 2^l - d < d, reduced mod 2^k for l = k.
      const Unsigned r =
          shift == kBits ? static_cast<Unsigned>(Unsigned(0) - du) : static_cast<Unsigned>((Unsigned(1) << shift) - du);
      magic = static_cast<Unsigned>(div_shifted(r, du) + 1);
    }
  }

  Unsigned magic;
  int shift;
};

}  // namespace int_div

/** \internal \returns floor(a / d) lane-wise, the lanes of \a a read as unsigned integers, for the \a magic and
 * \a shift of d computed by int_div::params. This lane-by-lane fallback keeps every packet type correct; backends
 * specialize it where the instructions exist. */
template <typename Packet>
EIGEN_DEVICE_FUNC inline Packet pudiv_magic(const Packet& a,
                                            std::make_unsigned_t<typename unpacket_traits<Packet>::type> magic,
                                            int shift) {
  using Scalar = typename unpacket_traits<Packet>::type;
  using Unsigned = std::make_unsigned_t<Scalar>;
  constexpr int kSize = unpacket_traits<Packet>::size;
  Scalar lanes[kSize];
  pstoreu(lanes, a);
  for (int i = 0; i < kSize; ++i) {
    lanes[i] = static_cast<Scalar>(int_div::udiv(static_cast<Unsigned>(lanes[i]), magic, shift));
  }
  return ploadu<Packet>(lanes);
}

/** \internal Truncating signed division: |q| = floor(|a| / |d|) with the sign of a * d. \a divisor_sign holds -1 in
 * every lane for d < 0, else 0. lowest() / -1 wraps to lowest(). */
template <typename Packet>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet
psdiv_magic(const Packet& a, std::make_unsigned_t<typename unpacket_traits<Packet>::type> magic, int shift,
            const Packet& divisor_sign) {
  // pabs(lowest()) wraps to lowest(), whose unsigned reading is |lowest()|.
  const Packet abs_quotient = pudiv_magic(pabs(a), magic, shift);
  const Packet quotient_sign = pxor(psignbit(a), divisor_sign);
  return psub(pxor(abs_quotient, quotient_sign), quotient_sign);
}

template <typename Scalar, bool IsSigned = std::is_signed<Scalar>::value>
struct scalar_int_div_op : int_div::params<Scalar> {
  using Base = int_div::params<Scalar>;
  using result_type = Scalar;

  template <typename Divisor>
  EIGEN_DEVICE_FUNC explicit scalar_int_div_op(const Divisor& d) : Base(int_div::unsigned_abs(d)) {
    eigen_assert(!int_div::is_negative(d) && "IntDivider: an unsigned numerator needs a positive divisor");
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    return int_div::udiv(a, this->magic, this->shift);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return pudiv_magic(a, this->magic, this->shift);
  }
};

template <typename Scalar>
struct scalar_int_div_op<Scalar, true> : int_div::params<std::make_unsigned_t<Scalar>> {
  using Unsigned = std::make_unsigned_t<Scalar>;
  using Base = int_div::params<Unsigned>;
  using result_type = Scalar;

  template <typename Divisor>
  EIGEN_DEVICE_FUNC explicit scalar_int_div_op(const Divisor& d)
      : Base(int_div::unsigned_abs(d)), divisor_sign(int_div::is_negative(d) ? Scalar(-1) : Scalar(0)) {}

  // q = (|q| ^ s) - s for the mask s = -[q < 0]: a select here compiles to a mispredicted branch with some
  // compilers (GCC 12 for AArch64) when numerator signs are random. The conversion back to Scalar is modular; only
  // lowest() / -1 wraps.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const {
    const Unsigned abs_quotient = int_div::udiv(int_div::unsigned_abs(a), this->magic, this->shift);
    const Unsigned quotient_sign = static_cast<Unsigned>(
        static_cast<Unsigned>(Unsigned(0) - static_cast<Unsigned>(a < 0)) ^ static_cast<Unsigned>(divisor_sign));
    return static_cast<Scalar>(static_cast<Unsigned>((abs_quotient ^ quotient_sign) - quotient_sign));
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return psdiv_magic(a, this->magic, this->shift, pset1<Packet>(divisor_sign));
  }

  Scalar divisor_sign;
};

template <typename Scalar, bool IsSigned>
struct functor_traits<scalar_int_div_op<Scalar, IsSigned>> {
  static constexpr int Cost = 2 * NumTraits<Scalar>::MulCost + 5 * NumTraits<Scalar>::AddCost;
  static constexpr bool PacketAccess = packet_traits<Scalar>::HasFastIntDiv;
};

}  // namespace internal

/** \class IntDivider
 * \ingroup Core_Module
 *
 * \brief Divides integers by a divisor that is fixed at run time, with multiplications and shifts in place of
 * division instructions.
 *
 * \tparam Scalar_ the type of the numerators and quotients: a built-in integer type other than \c bool.
 *
 * The constructor precomputes a multiplier and a shift count from the divisor. Each quotient then costs a
 * double-width multiplication, an addition, a subtraction and two shifts, which on most hardware is several times
 * cheaper than an integer division, and vectorizes where hardware integer division does not exist. The quotients
 * equal those of the built-in \c / operator, truncated toward zero:
 *
 * \code
 * Eigen::IntDivider<int> by(d);           // d is known only at run time
 * int q = n / by;                         // == n / d
 * Eigen::ArrayXi Q = A / by;              // == A / d, coefficient-wise
 * \endcode
 *
 * The divisor may have any built-in integer type, and its magnitude may exceed the range of \c Scalar_. It must be
 * nonzero, and it must be positive when \c Scalar_ is unsigned. Unlike the built-in operator, which has undefined
 * behavior there, \c lowest() divided by -1 wraps to \c lowest().
 *
 * Use \c IntDivider when one divisor that is not a compile-time constant divides many numerators: constructing it costs
 * about as much as a few divisions. For a constant divisor, write the constant; the compiler then derives a similar
 * sequence itself.
 */
template <typename Scalar_>
class IntDivider {
  EIGEN_STATIC_ASSERT((std::is_integral<Scalar_>::value && !std::is_same<Scalar_, bool>::value),
                      THE_NUMERATOR_TYPE_MUST_BE_A_BUILT_IN_INTEGER)

 public:
  using Scalar = Scalar_;
  using Functor = internal::scalar_int_div_op<Scalar>;

  /** Precomputes the division by \a divisor. */
  template <typename Divisor>
  EIGEN_DEVICE_FUNC explicit IntDivider(const Divisor& divisor) : m_functor(divisor) {
    EIGEN_STATIC_ASSERT((std::is_integral<Divisor>::value && !std::is_same<Divisor, bool>::value),
                        THE_DIVISOR_TYPE_MUST_BE_A_BUILT_IN_INTEGER)
  }

  /** \returns \a numerator divided by the divisor, truncated toward zero. */
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar divide(const Scalar& numerator) const { return m_functor(numerator); }

  /** \returns an expression of the coefficient-wise quotients of \a numerators by the divisor. */
  template <typename Derived>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CwiseUnaryOp<Functor, const Derived> divide(
      const DenseBase<Derived>& numerators) const {
    EIGEN_STATIC_ASSERT(
        (std::is_same<typename Derived::Scalar, Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
    return CwiseUnaryOp<Functor, const Derived>(numerators.derived(), m_functor);
  }

  /** \returns the functor that divides one numerator. */
  EIGEN_DEVICE_FUNC const Functor& functor() const { return m_functor; }

 private:
  Functor m_functor;
};

/** \relates IntDivider \returns \a numerator divided by the divisor of \a divider, truncated toward zero. */
template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator/(const typename IntDivider<Scalar>::Scalar& numerator,
                                                       const IntDivider<Scalar>& divider) {
  return divider.divide(numerator);
}

/** \relates IntDivider \returns an expression of the coefficient-wise quotients of \a numerators by the divisor of
 * \a divider. */
template <typename Derived, typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE CwiseUnaryOp<internal::scalar_int_div_op<Scalar>, const Derived> operator/(
    const DenseBase<Derived>& numerators, const IntDivider<Scalar>& divider) {
  return divider.divide(numerators);
}

}  // namespace Eigen

#endif  // EIGEN_INTEGER_DIVISION_H

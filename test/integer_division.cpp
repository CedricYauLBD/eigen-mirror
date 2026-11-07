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

#include "main.h"

#include <vector>

// Truncating n / d by hardware division of the magnitudes. lowest() / -1 wraps to lowest(), as IntDivider documents.
template <typename Scalar, typename Divisor>
Scalar reference_quotient(Scalar n, Divisor d) {
  const uint64_t abs_n = internal::int_div::unsigned_abs(n);
  const uint64_t abs_d = internal::int_div::unsigned_abs(d);
  const uint64_t abs_q = abs_n / abs_d;
  const bool negative = internal::int_div::is_negative(n) != internal::int_div::is_negative(d);
  return static_cast<Scalar>(negative ? uint64_t(0) - abs_q : abs_q);
}

template <typename Scalar, typename Divisor>
void verify_quotients(const Matrix<Scalar, Dynamic, 1>& numerators, Divisor d) {
  const IntDivider<Scalar> by(d);
  // The expression takes the packet path where packet_traits<Scalar>::HasFastIntDiv; n / by takes the scalar path.
  const Matrix<Scalar, Dynamic, 1> quotients = numerators / by;
  for (Index i = 0; i < numerators.size(); ++i) {
    const Scalar n = numerators(i);
    const Scalar expected = reference_quotient(n, d);
    VERIFY_IS_EQUAL(quotients(i), expected);
    VERIFY_IS_EQUAL(n / by, expected);
  }
}

template <typename T>
std::vector<T> special_values() {
  using Unsigned = std::make_unsigned_t<T>;
  constexpr int kBits = CHAR_BIT * sizeof(T);
  std::vector<T> values;
  for (int e = 0; e < kBits; ++e) {
    const Unsigned p = static_cast<Unsigned>(Unsigned(1) << e);
    for (Unsigned v : {p, static_cast<Unsigned>(p - 1), static_cast<Unsigned>(p + 1)}) {
      values.push_back(static_cast<T>(v));
      if (std::is_signed<T>::value) values.push_back(static_cast<T>(Unsigned(0) - v));
    }
  }
  values.insert(values.end(),
                {T(3), T(5), T(6), T(7), T(10), T(641), NumTraits<T>::highest(), NumTraits<T>::lowest(),
                 static_cast<T>(NumTraits<T>::highest() - 1), static_cast<T>(NumTraits<T>::lowest() + 1)});
  return values;
}

// Uniform in bit length rather than in value, so that small divisors and numerators occur.
template <typename T>
T random_magnitude() {
  using Unsigned = std::make_unsigned_t<T>;
  constexpr int kBits = CHAR_BIT * sizeof(T);
  const Unsigned bits = internal::random<Unsigned>(0, NumTraits<Unsigned>::highest());
  return static_cast<T>(bits >> internal::random<int>(0, kBits - 1));
}

// Numerators on both sides of each quotient boundary q * d, and the extremes.
template <typename Scalar, typename Divisor>
Matrix<Scalar, Dynamic, 1> test_numerators(Divisor d, Index count) {
  using Unsigned = std::make_unsigned_t<Scalar>;
  Matrix<Scalar, Dynamic, 1> numerators(count);
  const std::vector<Scalar> special = special_values<Scalar>();
  Index i = 0;
  for (; i < Index(special.size()) && i < count; ++i) numerators(i) = special[i];
  for (; i < count; ++i) {
    Scalar n = random_magnitude<Scalar>();
    if (i % 2 == 0) {
      // n = q * d + {-1, 0, 1} mod 2^k, in uint64_t: 16-bit operands would promote to int, where q * d can overflow.
      const uint64_t q = static_cast<uint64_t>(random_magnitude<Scalar>());
      const uint64_t offset = static_cast<uint64_t>(internal::random<int>(-1, 1));
      n = static_cast<Scalar>(static_cast<Unsigned>(q * static_cast<uint64_t>(d) + offset));
    }
    // -n mod 2^k, since -lowest() overflows.
    if (std::is_signed<Scalar>::value && internal::random<bool>()) {
      n = static_cast<Scalar>(Unsigned(0) - static_cast<Unsigned>(n));
    }
    numerators(i) = n;
  }
  return numerators;
}

template <typename Scalar, typename Divisor>
void test_divisor_type() {
  // Odd count: a scalar tail after the packets.
  constexpr Index kCount = 1027;
  for (Divisor d : special_values<Divisor>()) {
    if (d == 0 || (!std::is_signed<Scalar>::value && internal::int_div::is_negative(d))) continue;
    verify_quotients(test_numerators<Scalar>(d, kCount), d);
  }
  for (int repeat = 0; repeat < g_repeat; ++repeat) {
    Divisor d = random_magnitude<Divisor>();
    if (!std::is_signed<Scalar>::value && internal::int_div::is_negative(d)) d = static_cast<Divisor>(~d);
    if (d == 0) d = 1;
    verify_quotients(test_numerators<Scalar>(d, kCount), d);
  }
}

template <typename Scalar>
void test_random_division() {
  test_divisor_type<Scalar, uint8_t>();
  test_divisor_type<Scalar, uint16_t>();
  test_divisor_type<Scalar, uint32_t>();
  test_divisor_type<Scalar, uint64_t>();
  // Positive signed divisors are valid for unsigned numerators too.
  test_divisor_type<Scalar, int8_t>();
  test_divisor_type<Scalar, int16_t>();
  test_divisor_type<Scalar, int32_t>();
  test_divisor_type<Scalar, int64_t>();
}

// Every numerator against every divisor of Divisor.
template <typename Scalar, typename Divisor>
void test_exhaustive_division() {
  using Unsigned = std::make_unsigned_t<Scalar>;
  const Index count = Index(NumTraits<Unsigned>::highest()) + 1;
  Matrix<Scalar, Dynamic, 1> numerators(count);
  for (Index i = 0; i < count; ++i) numerators(i) = static_cast<Scalar>(NumTraits<Scalar>::lowest() + i);
  constexpr int kDivisorBits = CHAR_BIT * sizeof(Divisor);
  constexpr bool kSignedDivisor = std::is_signed<Divisor>::value;
  const int64_t lowest = kSignedDivisor ? -(int64_t(1) << (kDivisorBits - 1)) : 0;
  const int64_t highest = (int64_t(1) << (kDivisorBits - int(kSignedDivisor))) - 1;
  for (int64_t d = lowest; d <= highest; ++d) {
    if (d == 0 || (!std::is_signed<Scalar>::value && d < 0)) continue;
    verify_quotients(numerators, static_cast<Divisor>(d));
  }
}

// One packet of numerators through the kernel. The extremes and the quotient boundaries are where a multiplier or an
// addend that is off by one shows.
template <typename Scalar, typename Packet, typename Divisor>
void verify_packet_kernel(Divisor d) {
  using Unsigned = std::make_unsigned_t<Scalar>;
  constexpr int kSize = internal::unpacket_traits<Packet>::size;
  const IntDivider<Scalar> by(d);
  const std::vector<Scalar> special = special_values<Scalar>();
  Scalar lanes[kSize];
  Scalar result[kSize];
  for (size_t next = 0; next < special.size(); next += kSize) {
    for (int i = 0; i < kSize; ++i) {
      const size_t index = next + size_t(i);
      lanes[i] = index < special.size() ? special[index] : random_magnitude<Scalar>();
    }
    // n = q * d + {-1, 0, 1} mod 2^k, in uint64_t: see test_numerators.
    const uint64_t q = static_cast<uint64_t>(random_magnitude<Scalar>());
    const uint64_t offset = static_cast<uint64_t>(internal::random<int>(-1, 1));
    lanes[kSize - 1] = static_cast<Scalar>(static_cast<Unsigned>(q * static_cast<uint64_t>(d) + offset));
    internal::pstoreu(result, by.functor().packetOp(internal::ploadu<Packet>(lanes)));
    for (int i = 0; i < kSize; ++i) VERIFY_IS_EQUAL(result[i], reference_quotient(lanes[i], d));
  }
}

// The packet and half-packet kernels directly: an expression need not reach the half packet.
template <typename Scalar, typename Packet>
void test_packet_kernel() {
  // Without a packet type the "packet" is a plain Scalar: packetOp is unreachable, since PacketAccess needs a kernel,
  // and pabs(lowest()) would overflow where a real packet wraps.
  if (std::is_same<Packet, Scalar>::value) return;
  for (Scalar d : special_values<Scalar>()) {
    if (d == 0) continue;
    verify_packet_kernel<Scalar, Packet>(d);
  }
  for (int repeat = 0; repeat < 10 * g_repeat; ++repeat) {
    Scalar d = random_magnitude<Scalar>();
    if (d == 0) d = 1;
    verify_packet_kernel<Scalar, Packet>(d);
  }
  // Divisors whose magnitude exceeds the numerators, where every quotient is 0.
  for (uint64_t d : {uint64_t(1) << 32, (uint64_t(1) << 32) + 1, ~uint64_t(0), uint64_t(1) << 63}) {
    if (sizeof(Scalar) < 8) verify_packet_kernel<Scalar, Packet>(d);
  }
  if (sizeof(Scalar) < 4) {
    verify_packet_kernel<Scalar, Packet>(static_cast<uint32_t>(uint32_t(1) << (CHAR_BIT * sizeof(Scalar))));
  }
}

template <typename Scalar>
void test_packet_kernels() {
  test_packet_kernel<Scalar, typename internal::packet_traits<Scalar>::type>();
  test_packet_kernel<Scalar, typename internal::packet_traits<Scalar>::half>();
}

template <typename Numerators, typename Quotients, typename Divisor>
void verify_coefficients(const DenseBase<Numerators>& numerators, const DenseBase<Quotients>& quotients, Divisor d) {
  for (Index j = 0; j < numerators.cols(); ++j) {
    for (Index i = 0; i < numerators.rows(); ++i) {
      VERIFY_IS_EQUAL(quotients(i, j), reference_quotient(numerators(i, j), d));
    }
  }
}

// The quotient expression through dense types other than a plain vector.
template <typename Scalar>
void test_expressions() {
  const Scalar d = static_cast<Scalar>(random_magnitude<Scalar>() | 1);
  const IntDivider<Scalar> by(d);

  using Array = Eigen::Array<Scalar, Dynamic, Dynamic>;
  const Array a = Array::Random(37, 29);
  const Array q = a / by;
  const Array q_block = by.divide(a.block(1, 2, 33, 21));
  verify_coefficients(a, q, d);
  verify_coefficients(a.block(1, 2, 33, 21), q_block, d);

  using Matrix3 = Matrix<Scalar, 3, 3, RowMajor>;
  const Matrix3 m = Matrix3::Random();
  const Matrix3 qm = m.transpose() / by;
  verify_coefficients(m.transpose(), qm, d);

  // Fixed sizes whose best packet is a half packet or none.
  const Matrix<Scalar, 2, 1> v2 = Matrix<Scalar, 2, 1>::Random();
  const Matrix<Scalar, 2, 1> q2 = v2 / by;
  const Matrix<Scalar, 5, 1> v5 = Matrix<Scalar, 5, 1>::Random();
  const Matrix<Scalar, 5, 1> q5 = v5 / by;
  verify_coefficients(v2, q2, d);
  verify_coefficients(v5, q5, d);
}

void test_traits() {
  // Types without a kernel stay on the scalar path; a vectorized expression would compute lane-wise fallbacks.
  STATIC_CHECK(!(internal::functor_traits<internal::scalar_int_div_op<int16_t>>::PacketAccess));
  STATIC_CHECK(!(internal::functor_traits<internal::scalar_int_div_op<uint8_t>>::PacketAccess));
#if defined(EIGEN_VECTORIZE_SSE2) || defined(EIGEN_VECTORIZE_NEON)
  STATIC_CHECK((internal::functor_traits<internal::scalar_int_div_op<int32_t>>::PacketAccess));
#endif
#if defined(EIGEN_VECTORIZE_AVX2) || defined(EIGEN_VECTORIZE_AVX512)
  STATIC_CHECK((internal::functor_traits<internal::scalar_int_div_op<int64_t>>::PacketAccess));
#endif
  // AVX-512 leaves the unsigned types unvectorized.
#if (defined(EIGEN_VECTORIZE_SSE2) && !defined(EIGEN_VECTORIZE_AVX512)) || defined(EIGEN_VECTORIZE_NEON)
  STATIC_CHECK((internal::functor_traits<internal::scalar_int_div_op<uint32_t>>::PacketAccess));
#endif
#if defined(EIGEN_VECTORIZE_AVX2) && !defined(EIGEN_VECTORIZE_AVX512)
  STATIC_CHECK((internal::functor_traits<internal::scalar_int_div_op<uint64_t>>::PacketAccess));
#endif
}

EIGEN_DECLARE_TEST(integer_division) {
  CALL_SUBTEST_1(test_traits());
  CALL_SUBTEST_1((test_exhaustive_division<uint8_t, uint8_t>()));
  CALL_SUBTEST_1((test_exhaustive_division<int8_t, int8_t>()));
  CALL_SUBTEST_1((test_exhaustive_division<uint8_t, uint16_t>()));
  CALL_SUBTEST_1((test_exhaustive_division<int8_t, int16_t>()));
  CALL_SUBTEST_2((test_exhaustive_division<uint16_t, uint8_t>()));
  CALL_SUBTEST_2((test_exhaustive_division<int16_t, int8_t>()));
  CALL_SUBTEST_3((test_random_division<uint8_t>()));
  CALL_SUBTEST_3((test_random_division<int8_t>()));
  CALL_SUBTEST_3((test_random_division<uint16_t>()));
  CALL_SUBTEST_3((test_random_division<int16_t>()));
  CALL_SUBTEST_3((test_packet_kernels<uint8_t>()));
  CALL_SUBTEST_3((test_packet_kernels<int8_t>()));
  CALL_SUBTEST_3((test_packet_kernels<uint16_t>()));
  CALL_SUBTEST_3((test_packet_kernels<int16_t>()));
  CALL_SUBTEST_4((test_random_division<uint32_t>()));
  CALL_SUBTEST_4((test_random_division<int32_t>()));
  CALL_SUBTEST_4((test_packet_kernels<uint32_t>()));
  CALL_SUBTEST_4((test_packet_kernels<int32_t>()));
  CALL_SUBTEST_4((test_expressions<int32_t>()));
  CALL_SUBTEST_5((test_random_division<uint64_t>()));
  CALL_SUBTEST_5((test_random_division<int64_t>()));
  CALL_SUBTEST_5((test_random_division<long long>()));
  CALL_SUBTEST_5((test_packet_kernels<uint64_t>()));
  CALL_SUBTEST_5((test_packet_kernels<int64_t>()));
  CALL_SUBTEST_5((test_expressions<int64_t>()));
  CALL_SUBTEST_5((test_expressions<uint32_t>()));
}

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Pedro Gonnet (pedro.gonnet@gmail.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_
#define THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_FLOAT(Packet16f)
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_DOUBLE(Packet8d)

template <>
EIGEN_STRONG_INLINE Packet16h pfrexp(const Packet16h& a, Packet16h& exponent) {
  Packet16f fexponent;
  const Packet16h out = float2half(pfrexp<Packet16f>(half2float(a), fexponent));
  exponent = float2half(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet16h pldexp(const Packet16h& a, const Packet16h& exponent) {
  return float2half(pldexp<Packet16f>(half2float(a), half2float(exponent)));
}

template <>
EIGEN_STRONG_INLINE Packet16bf pfrexp(const Packet16bf& a, Packet16bf& exponent) {
  // Both results are exact: the mantissa keeps the input's significand, and the exponent is a
  // small integer.
  Packet16f fexponent;
  const Packet16bf out = F32ToBf16Truncate(pfrexp<Packet16f>(Bf16ToF32(a), fexponent));
  exponent = F32ToBf16Truncate(fexponent);
  return out;
}

template <>
EIGEN_STRONG_INLINE Packet16bf pldexp(const Packet16bf& a, const Packet16bf& exponent) {
  return F32ToBf16(pldexp<Packet16f>(Bf16ToF32(a), Bf16ToF32(exponent)));
}

#if EIGEN_FAST_MATH
// psqrt(x) = x * rsqrt(x) from an approximation of rsqrt(x), as generic_sqrt_newton_step computes it, except that the
// lanes to return unchanged (x = +-0 and +inf) are found from the bits of the approximation, rsqrt14/rsqrt28 being
//   +-inf for +-0 and for a subnormal x flushed by DAZ, 0 for +inf, and never subnormal.
// A compare of x with zero would be wrong under DAZ: the compiler may assume no flush happens, test the bits of x, miss
// the flushed lane and return x * inf = NaN. Integer compares of the approximation also survive -ffinite-math-only.
template <typename Packet, int Steps>
EIGEN_STRONG_INLINE Packet rsqrt_newton_refine(const Packet& x, const Packet& approx_rsqrt) {
  using Scalar = typename unpacket_traits<Packet>::type;
  const Packet one_point_five = pset1<Packet>(Scalar(1.5));
  const Packet minus_half = pset1<Packet>(Scalar(-0.5));
  Packet rsqrt = approx_rsqrt;
  for (int step = 0; step < Steps; ++step) {
    rsqrt = pmul(rsqrt, pmadd(pmul(minus_half, rsqrt), pmul(x, rsqrt), one_point_five));
  }
  return rsqrt;
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet16f psqrt<Packet16f>(const Packet16f& x) {
  const Packet16f approx = _mm512_rsqrt14_ps(x);
  const __m512i abs_bits = _mm512_and_si512(_mm512_castps_si512(approx), _mm512_set1_epi32(0x7fffffff));
  const __mmask16 return_x = _mm512_cmpeq_epi32_mask(abs_bits, _mm512_setzero_si512()) |
                             _mm512_cmpeq_epi32_mask(abs_bits, _mm512_set1_epi32(0x7f800000));
  return _mm512_mask_blend_ps(return_x, pmul(x, rsqrt_newton_refine<Packet16f, 1>(x, approx)), x);
}

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet8d psqrt<Packet8d>(const Packet8d& x) {
#ifdef EIGEN_VECTORIZE_AVX512ER
  const Packet8d approx = _mm512_rsqrt28_pd(x);
  constexpr int kSteps = 1;
#else
  const Packet8d approx = _mm512_rsqrt14_pd(x);
  constexpr int kSteps = 2;
#endif
  const __m512i abs_bits = _mm512_and_si512(_mm512_castpd_si512(approx), _mm512_set1_epi64(0x7fffffffffffffffLL));
  const __mmask8 return_x = _mm512_cmpeq_epi64_mask(abs_bits, _mm512_setzero_si512()) |
                            _mm512_cmpeq_epi64_mask(abs_bits, _mm512_set1_epi64(0x7ff0000000000000LL));
  return _mm512_mask_blend_pd(return_x, pmul(x, rsqrt_newton_refine<Packet8d, kSteps>(x, approx)), x);
}
#else
template <>
EIGEN_STRONG_INLINE Packet16f psqrt<Packet16f>(const Packet16f& x) {
  return _mm512_sqrt_ps(x);
}

template <>
EIGEN_STRONG_INLINE Packet8d psqrt<Packet8d>(const Packet8d& x) {
  return _mm512_sqrt_pd(x);
}
#endif

// prsqrt for float.
#if defined(EIGEN_VECTORIZE_AVX512ER)
template <>
EIGEN_STRONG_INLINE Packet16f prsqrt<Packet16f>(const Packet16f& x) {
  return _mm512_rsqrt28_ps(x);
}
#elif EIGEN_FAST_MATH

template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet16f prsqrt<Packet16f>(const Packet16f& x) {
  return generic_rsqrt_newton_step<Packet16f, /*Steps=*/1>::run(x, _mm512_rsqrt14_ps(x));
}
#endif

// prsqrt for double.
#if EIGEN_FAST_MATH
template <>
EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS Packet8d prsqrt<Packet8d>(const Packet8d& x) {
#ifdef EIGEN_VECTORIZE_AVX512ER
  return generic_rsqrt_newton_step<Packet8d, /*Steps=*/1>::run(x, _mm512_rsqrt28_pd(x));
#else
  return generic_rsqrt_newton_step<Packet8d, /*Steps=*/2>::run(x, _mm512_rsqrt14_pd(x));
#endif
}

template <>
EIGEN_STRONG_INLINE Packet16f preciprocal<Packet16f>(const Packet16f& a) {
#ifdef EIGEN_VECTORIZE_AVX512ER
  return _mm512_rcp28_ps(a);
#else
  // generic_reciprocal_newton_step, with its NaN-or-zero test as one compare into a mask register (r == 0 or
  // unordered), as for AVX: GCC compiles the generic predux_any(pcmp_lt_or_nan(...)) to a compare mask converted to a
  // vector and back.
  const Packet16f one = pset1<Packet16f>(1.0f);
  const Packet16f x = _mm512_rcp14_ps(a);
  const Packet16f refined = pmadd(x, pnmadd(a, x, one), x);
  const __mmask16 redo = _mm512_cmp_ps_mask(refined, _mm512_setzero_ps(), _CMP_EQ_UQ);
  return redo == 0 ? refined : _mm512_mask_div_ps(refined, redo, one, a);
#endif
}
#endif

EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_BF16(Packet16f, Packet16bf)
EIGEN_INSTANTIATE_GENERIC_SQRT_FUNCS_BF16(Packet16f, Packet16bf)

#ifndef EIGEN_VECTORIZE_AVX512FP16
EIGEN_INSTANTIATE_GENERIC_MATH_FUNCS_F16(Packet16f, Packet16h)
#endif  // EIGEN_VECTORIZE_AVX512FP16

}  // end namespace internal

}  // end namespace Eigen

#endif  // THIRD_PARTY_EIGEN3_EIGEN_SRC_CORE_ARCH_AVX512_MATHFUNCTIONS_H_

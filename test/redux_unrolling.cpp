// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Reductions of a runtime size must not unroll, however large the packet-scaled unrolling budget grows. With this
// limit the budget EIGEN_UNROLLING_LIMIT * PacketSize reaches HugeCost from a packet of two scalars on, as the default
// limit of 110 does from 91 scalars on (RVV at VLEN 1024 and LMUL 4). The limit itself stays below HugeCost, so the
// reductions without the packet factor are not affected.
#define EIGEN_UNROLLING_LIMIT 5000
#include "main.h"
#include "random_for_arithmetic.h"

template <typename Xpr>
typename Xpr::Scalar sum_by_coeff(const Xpr& x) {
  typename Xpr::Scalar sum(0);
  for (Index j = 0; j < x.cols(); ++j)
    for (Index i = 0; i < x.rows(); ++i) sum += x.coeff(i, j);
  return sum;
}

template <typename Xpr>
constexpr int sum_unrolling() {
  using Scalar = typename Xpr::Scalar;
  return internal::redux_traits<internal::scalar_sum_op<Scalar, Scalar>, internal::redux_evaluator<Xpr>>::Unrolling;
}

template <typename Scalar>
void redux_runtime_size() {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  STATIC_CHECK(sum_unrolling<Vec>() == NoUnrolling);
  STATIC_CHECK(sum_unrolling<Mat>() == NoUnrolling);
  STATIC_CHECK(sum_unrolling<Block<Mat>>() == NoUnrolling);

  // Linear traversal for the vector, slice traversal for the block.
  const Vec v = random_for_arithmetic<Vec>(37);
  const Mat m = random_for_arithmetic<Mat>(23, 11);
  const Block<const Mat> block = m.block(1, 2, 15, 7);
  VERIFY_IS_APPROX(v.sum(), sum_by_coeff(v));
  VERIFY_IS_APPROX(m.sum(), sum_by_coeff(m));
  VERIFY_IS_APPROX(block.sum(), sum_by_coeff(block));
  // For a complex operand the budget scales with the real packet size, even where the reduction reads no packets.
  VERIFY_IS_APPROX(v.squaredNorm(), sum_by_coeff(v.cwiseAbs2()));
  VERIFY_IS_APPROX(m.squaredNorm(), sum_by_coeff(m.cwiseAbs2()));
  VERIFY_IS_APPROX(m.cwiseAbs2().sum(), sum_by_coeff(m.cwiseAbs2()));
}

template <typename Scalar>
void redux_fixed_size() {
  using Vec = Matrix<Scalar, 4, 1>;
  STATIC_CHECK(sum_unrolling<Vec>() == CompleteUnrolling);
  const Vec v = Vec::Random();
  VERIFY_IS_APPROX(v.sum(), sum_by_coeff(v));
}

EIGEN_DECLARE_TEST(redux_unrolling) {
  CALL_SUBTEST(redux_runtime_size<float>());
  CALL_SUBTEST(redux_runtime_size<double>());
  CALL_SUBTEST(redux_runtime_size<std::complex<float>>());
  CALL_SUBTEST(redux_runtime_size<std::complex<double>>());
  CALL_SUBTEST(redux_runtime_size<int>());
  CALL_SUBTEST(redux_fixed_size<float>());
  CALL_SUBTEST(redux_fixed_size<std::complex<double>>());
}

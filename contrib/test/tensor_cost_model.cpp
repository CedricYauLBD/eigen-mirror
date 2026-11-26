// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 The Eigen Authors.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/Tensor>

using Eigen::DefaultDevice;
using Eigen::Tensor;
using Eigen::TensorEvaluator;

// Regression: TensorCwiseNullaryOp must report bytes_loaded == 0.
// NullaryOps (constants, Zero, Identity, Random, sequence generators)
// produce values from registers or minimal state without loading from
// memory. If they reported nonzero bytes_loaded, expressions dominated
// by constants (e.g. Horner-form polynomials) would be misclassified as
// memory-bound and the threadpool cost model would over-restrict
// parallelism. See TensorEvaluator.h, the CwiseNullaryOp specialization
// of costPerCoeff().
template <typename Scalar>
static void test_nullary_zero_bytes_loaded() {
  Tensor<Scalar, 1> shape(/*firstDimension=*/16);
  auto zeros = shape.constant(Scalar(0));
  auto sevens = shape.constant(Scalar(7));

  using ZeroEval = TensorEvaluator<const decltype(zeros), DefaultDevice>;
  using ConstEval = TensorEvaluator<const decltype(sevens), DefaultDevice>;

  DefaultDevice device;
  ZeroEval zero_eval(zeros, device);
  ConstEval const_eval(sevens, device);

  for (bool vectorized : {false, true}) {
    const auto zero_cost = zero_eval.costPerCoeff(vectorized);
    const auto const_cost = const_eval.costPerCoeff(vectorized);
    VERIFY_IS_EQUAL(zero_cost.bytes_loaded(), 0.0);
    VERIFY_IS_EQUAL(zero_cost.bytes_stored(), 0.0);
    VERIFY_IS_EQUAL(const_cost.bytes_loaded(), 0.0);
    VERIFY_IS_EQUAL(const_cost.bytes_stored(), 0.0);
  }
}

static void test_cost_model_constants_odr_use() {
  using Cost = Eigen::TensorCostModel<DefaultDevice>;
  VERIFY_IS_EQUAL(read_through_address(Cost::kDeviceCyclesPerComputeCycle), 1);
  VERIFY_IS_EQUAL(read_through_address(Cost::kStartupCycles), 100000);
  VERIFY_IS_EQUAL(read_through_address(Cost::kPerThreadCycles), 50000);
  VERIFY_IS_EQUAL(read_through_address(Cost::kTaskSize), 40000);
  VERIFY_IS_EQUAL(read_through_address(Cost::kMemBandwidthSaturationThreads), 4);
  VERIFY_IS_EQUAL(read_through_address(Cost::kMemBoundThreshold), 2.0);
  VERIFY_IS_EQUAL(read_through_address(Cost::kDramThresholdBytes), 1024.0 * 1024.0);
}

template <typename Scalar>
struct scalar_only_unary_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& x) const { return -x; }
};

template <typename Scalar>
struct scalar_only_binary_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& x, const Scalar& y) const { return x + y; }
};

template <typename Scalar>
struct scalar_only_ternary_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& x, const Scalar& y, const Scalar& z) const {
    return x + y + z;
  }
};

namespace Eigen {
namespace internal {
template <typename Scalar>
struct functor_traits<scalar_only_unary_op<Scalar>> {
  enum { Cost = 16, PacketAccess = false };
};
template <typename Scalar>
struct functor_traits<scalar_only_binary_op<Scalar>> {
  enum { Cost = 16, PacketAccess = false };
};
template <typename Scalar>
struct functor_traits<scalar_only_ternary_op<Scalar>> {
  enum { Cost = 16, PacketAccess = false };
};
}  // namespace internal
}  // namespace Eigen

template <typename Scalar>
static void test_unary_binary_ternary_block_resource_cost() {
  Tensor<Scalar, 2> a(64, 64);
  Tensor<Scalar, 2> b(64, 64);
  Tensor<Scalar, 2> c(64, 64);
  a.setRandom();
  b.setRandom();
  c.setRandom();

  DefaultDevice device;

  auto unary_expr = a.unaryExpr(scalar_only_unary_op<Scalar>());
  using UnaryEval = TensorEvaluator<const decltype(unary_expr), DefaultDevice>;
  STATIC_CHECK(!UnaryEval::PacketAccess);
  UnaryEval unary_eval(unary_expr, device);
  VERIFY_IS_EQUAL(unary_eval.getResourceRequirements().cost_per_coeff.compute_cycles(), 16.0);

  auto binary_expr = a.binaryExpr(b, scalar_only_binary_op<Scalar>());
  using BinaryEval = TensorEvaluator<const decltype(binary_expr), DefaultDevice>;
  STATIC_CHECK(!BinaryEval::PacketAccess);
  BinaryEval binary_eval(binary_expr, device);
  VERIFY_IS_EQUAL(binary_eval.getResourceRequirements().cost_per_coeff.compute_cycles(), 16.0);

  Eigen::TensorCwiseTernaryOp<scalar_only_ternary_op<Scalar>, const decltype(a), const decltype(b), const decltype(c)>
      ternary_expr(a, b, c, scalar_only_ternary_op<Scalar>());
  using TernaryEval = TensorEvaluator<const decltype(ternary_expr), DefaultDevice>;
  STATIC_CHECK(!TernaryEval::PacketAccess);
  TernaryEval ternary_eval(ternary_expr, device);
  VERIFY_IS_EQUAL(ternary_eval.getResourceRequirements().cost_per_coeff.compute_cycles(), 16.0);
}

template <typename Scalar>
static void test_shuffling_cost_and_block_requirements() {
  Tensor<Scalar, 2> a(256, 256);
  a.setRandom();

  Eigen::array<Eigen::Index, 2> perm = {1, 0};
  auto shuffle_expr = a.shuffle(perm);
  using ShuffleEval = TensorEvaluator<const decltype(shuffle_expr), DefaultDevice>;

  DefaultDevice device;
  ShuffleEval eval(shuffle_expr, device);

  const auto scalar_cost = eval.costPerCoeff(/*vectorized=*/false);
  const auto vec_cost = eval.costPerCoeff(/*vectorized=*/true);
  if (ShuffleEval::PacketAccess && ShuffleEval::PacketSize > 1) {
    VERIFY(vec_cost.compute_cycles() < scalar_cost.compute_cycles());
  }

  const auto req = eval.getResourceRequirements();
  VERIFY(req.shape_type == Eigen::internal::TensorBlockShapeType::kUniformAllDims);
  VERIFY_IS_EQUAL(req.cost_per_coeff.bytes_loaded(), static_cast<double>(sizeof(Scalar)));
  VERIFY_IS_EQUAL(req.cost_per_coeff.bytes_stored(), static_cast<double>(sizeof(Scalar)));
  VERIFY_IS_EQUAL(req.cost_per_coeff.compute_cycles(), 1.0);
  // Small 256x256 2D transpose should stay single-threaded rather than paying
  // thread-pool dispatch and barrier overhead.
  const int threads_256 =
      Eigen::TensorCostModel<DefaultDevice>::numThreads(256.0 * 256.0, req.cost_per_coeff, /*max_threads=*/8);
  VERIFY_IS_EQUAL(threads_256, 1);

  // Larger 1024x1024 2D transpose should parallelize across threads.
  const int threads_1024 =
      Eigen::TensorCostModel<DefaultDevice>::numThreads(1024.0 * 1024.0, req.cost_per_coeff, /*max_threads=*/8);
  VERIFY(threads_1024 > 1);
}

// Regression: TensorConcatenationOp weights the cost of each operand by its
// share of the output, which was 0 / 0 when both operands are empty. The NaN
// went unnoticed until an enclosing expression with a non-empty output, such
// as a reduction over the concatenation axis, handed it to numThreads(), which
// converted a NaN thread count to int.
template <int DataLayout>
static void test_empty_concatenation_cost() {
  Tensor<float, 2, DataLayout> left(0, 64);
  Tensor<float, 2, DataLayout> right(0, 64);
  auto concatenation = left.concatenate(right, 0);
  Eigen::array<int, 1> concatenation_axis = {0};
  auto column_sums = concatenation.sum(concatenation_axis);

  DefaultDevice device;
  TensorEvaluator<const decltype(concatenation), DefaultDevice> eval(concatenation, device);
  TensorEvaluator<const decltype(column_sums), DefaultDevice> sum_eval(column_sums, device);
  sum_eval.evalSubExprsIfNeeded(nullptr);
  VERIFY_IS_EQUAL(sum_eval.dimensions().TotalSize(), 64);

  for (bool vectorized : {false, true}) {
    const auto cost = eval.costPerCoeff(vectorized);
    VERIFY((numext::isfinite)(cost.bytes_loaded()));
    VERIFY((numext::isfinite)(cost.bytes_stored()));
    VERIFY((numext::isfinite)(cost.compute_cycles()));

    const auto sum_cost = sum_eval.costPerCoeff(vectorized);
    VERIFY((numext::isfinite)(sum_cost.total_bytes()));
    VERIFY((numext::isfinite)(sum_cost.compute_cycles()));
    VERIFY_IS_EQUAL(Eigen::TensorCostModel<DefaultDevice>::numThreads(64.0, sum_cost, /*max_threads=*/8), 1);
  }
  sum_eval.cleanup();
}

// Regression: numThreads() must not convert a NaN thread count to int if an
// evaluator ever reports a NaN cost.
static void test_num_threads_nan_cost() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Eigen::TensorOpCost nan_cost(nan, nan, nan);
  VERIFY_IS_EQUAL(Eigen::TensorCostModel<DefaultDevice>::numThreads(1e9, nan_cost, /*max_threads=*/8), 1);
}

EIGEN_DECLARE_TEST(tensor_cost_model) {
  CALL_SUBTEST(test_cost_model_constants_odr_use());
  CALL_SUBTEST(test_nullary_zero_bytes_loaded<float>());
  CALL_SUBTEST(test_nullary_zero_bytes_loaded<double>());
  CALL_SUBTEST(test_nullary_zero_bytes_loaded<int>());
  CALL_SUBTEST(test_unary_binary_ternary_block_resource_cost<float>());
  CALL_SUBTEST(test_unary_binary_ternary_block_resource_cost<double>());
  CALL_SUBTEST(test_shuffling_cost_and_block_requirements<float>());
  CALL_SUBTEST(test_shuffling_cost_and_block_requirements<double>());
  CALL_SUBTEST(test_empty_concatenation_cost<Eigen::ColMajor>());
  CALL_SUBTEST(test_empty_concatenation_cost<Eigen::RowMajor>());
  CALL_SUBTEST(test_num_threads_nan_cost());
}

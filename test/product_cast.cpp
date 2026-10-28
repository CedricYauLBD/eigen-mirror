// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define TEST_ENABLE_TEMPORARY_TRACKING
#define TEST_IGNORE_STACK_ALLOCATED_TEMPORARY

#include "main.h"

// GEMM operands of the form A.cast<T>() are packed straight from A.

// Temporaries per cast operand: none, or one with EIGEN_USE_BLAS, which materializes the cast for the external ?gemm.
#ifdef EIGEN_USE_BLAS
constexpr int kCastTemporaries = 1;
#else
constexpr int kCastTemporaries = 0;
#endif

template <typename Src, int Order>
Matrix<Src, Dynamic, Dynamic, Order> random_small_integers(Index rows, Index cols, int bound) {
  Matrix<Src, Dynamic, Dynamic, Order> m(rows, cols);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) m(i, j) = Src(internal::random<int>(-bound, bound));
  return m;
}

// The lazily cast operands pack to the same blocks as the materialized casts, so the products agree bit for bit.
template <typename Src, typename Dst, int LhsOrder, int RhsOrder, int DstOrder>
void lazy_cast_product(Index rows, Index cols, Index depth) {
  using DstMat = Matrix<Dst, Dynamic, Dynamic, DstOrder>;
  using PlainDst = Matrix<Dst, Dynamic, Dynamic>;
  Matrix<Src, Dynamic, Dynamic, LhsOrder> a = random_small_integers<Src, LhsOrder>(rows + 3, depth + 2, 8);
  Matrix<Src, Dynamic, Dynamic, RhsOrder> b = random_small_integers<Src, RhsOrder>(depth + 1, cols + 4, 8);
  // Strided operands: blocks of larger matrices.
  auto A = a.block(1, 2, rows, depth);
  auto B = b.block(1, 3, depth, cols);
  Matrix<Dst, Dynamic, Dynamic, LhsOrder> Ad = A.template cast<Dst>();
  Matrix<Dst, Dynamic, Dynamic, RhsOrder> Bd = B.template cast<Dst>();

  DstMat ref(rows, cols), dst(rows, cols);
  ref.noalias() = Ad * Bd;
  dst.setRandom();
  VERIFY_EVALUATION_COUNT(dst.noalias() = A.template cast<Dst>() * B.template cast<Dst>(), 2 * kCastTemporaries);
  VERIFY_IS_EQUAL(dst, ref);
  VERIFY_EVALUATION_COUNT(dst.noalias() += A.template cast<Dst>() * B.template cast<Dst>(), 2 * kCastTemporaries);
  VERIFY_IS_EQUAL(dst, (ref + ref).eval());
  VERIFY_EVALUATION_COUNT(dst.noalias() -= A.template cast<Dst>() * B.template cast<Dst>(), 2 * kCastTemporaries);
  VERIFY_IS_EQUAL(dst, ref);

  // One side already in the kernel's type.
  ref.noalias() = Ad * Bd;
  VERIFY_EVALUATION_COUNT(dst.noalias() = A.template cast<Dst>() * Bd, kCastTemporaries);
  VERIFY_IS_EQUAL(dst, ref);
  VERIFY_EVALUATION_COUNT(dst.noalias() = Ad * B.template cast<Dst>(), kCastTemporaries);
  VERIFY_IS_EQUAL(dst, ref);
  // A transposed operand below the cast.
  DstMat dst_t(cols, rows), ref_t(cols, rows);
  ref_t.noalias() = Bd.transpose() * Ad.transpose();
  VERIFY_EVALUATION_COUNT(dst_t.noalias() = B.transpose().template cast<Dst>() * A.transpose().template cast<Dst>(),
                          2 * kCastTemporaries);
  VERIFY_IS_EQUAL(dst_t, ref_t);

  // A scalar factor below the cast is not hoisted out of it, and the operand is materialized as before.
  ref.noalias() = (Src(2) * A).template cast<Dst>().eval() * Bd;
  dst.noalias() = (Src(2) * A).template cast<Dst>() * B.template cast<Dst>();
  VERIFY_IS_EQUAL(dst, ref);
  // A scalar factor above the cast multiplies the result.
  ref.noalias() = (Dst(3) * Ad) * Bd;
  dst.noalias() = (Dst(3) * A.template cast<Dst>()) * B.template cast<Dst>();
  VERIFY_IS_EQUAL(dst, ref);
  // Not a GEMM: the GEMV fallback for a single result column.
  PlainDst col(rows, 1);
  col.noalias() = A.template cast<Dst>() * B.col(0).template cast<Dst>();
  VERIFY_IS_EQUAL(col, (Ad * Bd.col(0)).eval());
}

template <typename Src, typename Dst>
void lazy_cast_products() {
  const Index rows = internal::random<Index>(20, 300);
  const Index cols = internal::random<Index>(20, 300);
  const Index depth = internal::random<Index>(20, 300);
  CALL_SUBTEST((lazy_cast_product<Src, Dst, ColMajor, ColMajor, ColMajor>(rows, cols, depth)));
  CALL_SUBTEST((lazy_cast_product<Src, Dst, RowMajor, ColMajor, ColMajor>(rows, cols, depth)));
  CALL_SUBTEST((lazy_cast_product<Src, Dst, ColMajor, RowMajor, RowMajor>(rows, cols, depth)));
  CALL_SUBTEST((lazy_cast_product<Src, Dst, RowMajor, RowMajor, RowMajor>(rows, cols, depth)));
  // Sizes that are not a multiple of the packet or the register block.
  CALL_SUBTEST((lazy_cast_product<Src, Dst, ColMajor, RowMajor, ColMajor>(17, 13, 9)));
  CALL_SUBTEST((lazy_cast_product<Src, Dst, RowMajor, ColMajor, RowMajor>(31, 7, 23)));
}

EIGEN_DECLARE_TEST(product_cast) {
  using MatrixXhalf = Matrix<half, Dynamic, Dynamic>;
  using internal::gemm_operand_traits;
#ifndef EIGEN_USE_BLAS
  STATIC_CHECK((std::is_same<gemm_operand_traits<decltype(MatrixXhalf().cast<float>())>::StorageScalar, half>::value));
  STATIC_CHECK((
      std::is_same<gemm_operand_traits<decltype(MatrixXf().transpose().cast<double>())>::StorageScalar, float>::value));
#endif
  STATIC_CHECK(
      (std::is_same<gemm_operand_traits<decltype((MatrixXhalf() + MatrixXhalf()).cast<float>())>::StorageScalar,
                    float>::value));
  STATIC_CHECK((std::is_same<gemm_operand_traits<decltype((half(2) * MatrixXhalf()).cast<float>())>::StorageScalar,
                             float>::value));
  STATIC_CHECK((std::is_same<gemm_operand_traits<decltype(MatrixXf().cast<std::complex<float>>())>::StorageScalar,
                             std::complex<float>>::value));
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((lazy_cast_products<half, float>()));
    CALL_SUBTEST_2((lazy_cast_products<bfloat16, float>()));
    CALL_SUBTEST_3((lazy_cast_products<float, double>()));
    CALL_SUBTEST_3((lazy_cast_products<int, float>()));
    CALL_SUBTEST_4((lazy_cast_products<int, double>()));
    CALL_SUBTEST_4((lazy_cast_products<half, double>()));
  }
}

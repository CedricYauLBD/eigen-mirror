// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "sparse_solver.h"

#include <Eigen/CholmodSupport>

// A matrix that never stored an entry has no index and value arrays. CHOLMOD must still be able to analyze it, and the
// solver must remain usable after the numerical factorization of this zero matrix failed.
template <typename Solver>
void check_cholmod_matrix_without_entries(Solver& solver) {
  using Mat = typename Solver::MatrixType;
  using Scalar = typename Mat::Scalar;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using SpMat = SparseMatrix<Scalar, ColMajor, typename Mat::StorageIndex>;

  Mat A(3, 3);
  solver.analyzePattern(A);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_EQUAL(solver.rows(), 3);
  VERIFY_IS_EQUAL(solver.cols(), 3);
  solver.factorize(A);
  VERIFY_IS_EQUAL(solver.info(), NumericalIssue);
  DenseVector b = DenseVector::Ones(3);
  DenseVector x = solver.solve(b);
  VERIFY_IS_EQUAL(x.size(), 3);
  VERIFY_IS_EQUAL(solver.info(), NumericalIssue);

  // A right-hand side without entries is viewed in the same way.
  A.setIdentity();
  solver.compute(A);
  VERIFY_IS_EQUAL(solver.info(), Success);
  SpMat spB(3, 2), spX;
  spX = solver.solve(spB);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_EQUAL(spX.rows(), 3);
  VERIFY_IS_EQUAL(spX.cols(), 2);
  VERIFY_IS_EQUAL(spX.nonZeros(), 0);

  Mat empty(0, 0);
  solver.compute(empty);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_EQUAL(solver.rows(), 0);
  VERIFY_IS_EQUAL(solver.cols(), 0);
}

template <typename SparseType>
void test_cholmod_ST() {
  CholmodDecomposition<SparseType, Lower> g_chol_colmajor_lower;
  g_chol_colmajor_lower.setMode(CholmodSupernodalLLt);
  CholmodDecomposition<SparseType, Upper> g_chol_colmajor_upper;
  g_chol_colmajor_upper.setMode(CholmodSupernodalLLt);
  CholmodDecomposition<SparseType, Lower> g_llt_colmajor_lower;
  g_llt_colmajor_lower.setMode(CholmodSimplicialLLt);
  CholmodDecomposition<SparseType, Upper> g_llt_colmajor_upper;
  g_llt_colmajor_upper.setMode(CholmodSimplicialLLt);
  CholmodDecomposition<SparseType, Lower> g_ldlt_colmajor_lower;
  g_ldlt_colmajor_lower.setMode(CholmodLDLt);
  CholmodDecomposition<SparseType, Upper> g_ldlt_colmajor_upper;
  g_ldlt_colmajor_upper.setMode(CholmodLDLt);

  CholmodSupernodalLLT<SparseType, Lower> chol_colmajor_lower;
  CholmodSupernodalLLT<SparseType, Upper> chol_colmajor_upper;
  CholmodSimplicialLLT<SparseType, Lower> llt_colmajor_lower;
  CholmodSimplicialLLT<SparseType, Upper> llt_colmajor_upper;
  CholmodSimplicialLDLT<SparseType, Lower> ldlt_colmajor_lower;
  CholmodSimplicialLDLT<SparseType, Upper> ldlt_colmajor_upper;

  check_sparse_spd_solving(g_chol_colmajor_lower);
  check_sparse_spd_solving(g_chol_colmajor_upper);
  check_sparse_spd_solving(g_llt_colmajor_lower);
  check_sparse_spd_solving(g_llt_colmajor_upper);
  check_sparse_spd_solving(g_ldlt_colmajor_lower);
  check_sparse_spd_solving(g_ldlt_colmajor_upper);

  check_sparse_spd_solving(chol_colmajor_lower);
  check_sparse_spd_solving(chol_colmajor_upper);
  check_sparse_spd_solving(llt_colmajor_lower);
  check_sparse_spd_solving(llt_colmajor_upper);
  check_sparse_spd_solving(ldlt_colmajor_lower);
  check_sparse_spd_solving(ldlt_colmajor_upper);

  check_sparse_spd_determinant(chol_colmajor_lower);
  check_sparse_spd_determinant(chol_colmajor_upper);
  check_sparse_spd_determinant(llt_colmajor_lower);
  check_sparse_spd_determinant(llt_colmajor_upper);
  check_sparse_spd_determinant(ldlt_colmajor_lower);
  check_sparse_spd_determinant(ldlt_colmajor_upper);

  check_sparse_zero_matrix(chol_colmajor_lower);
  check_sparse_zero_matrix(chol_colmajor_upper);
  check_sparse_zero_matrix(llt_colmajor_lower);
  check_sparse_zero_matrix(llt_colmajor_upper);
  check_sparse_zero_matrix(ldlt_colmajor_lower);
  check_sparse_zero_matrix(ldlt_colmajor_upper);

  check_cholmod_matrix_without_entries(g_chol_colmajor_lower);
  check_cholmod_matrix_without_entries(chol_colmajor_lower);
  check_cholmod_matrix_without_entries(chol_colmajor_upper);
  check_cholmod_matrix_without_entries(llt_colmajor_lower);
  check_cholmod_matrix_without_entries(llt_colmajor_upper);
  check_cholmod_matrix_without_entries(ldlt_colmajor_lower);
  check_cholmod_matrix_without_entries(ldlt_colmajor_upper);
}

template <typename T, int flags, typename IdxType>
void test_cholmod_T() {
  test_cholmod_ST<SparseMatrix<T, flags, IdxType> >();
}

EIGEN_DECLARE_TEST(cholmod_support) {
  CALL_SUBTEST_11((test_cholmod_T<double, ColMajor, int>()));
  CALL_SUBTEST_12((test_cholmod_T<double, ColMajor, long>()));
  CALL_SUBTEST_13((test_cholmod_T<double, RowMajor, int>()));
  CALL_SUBTEST_14((test_cholmod_T<double, RowMajor, long>()));
  CALL_SUBTEST_21((test_cholmod_T<std::complex<double>, ColMajor, int>()));
  CALL_SUBTEST_22((test_cholmod_T<std::complex<double>, ColMajor, long>()));
  // TODO complex row-major matrices do not work at the moment:
}

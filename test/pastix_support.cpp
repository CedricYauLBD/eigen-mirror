// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2012 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "sparse_solver.h"
#include <Eigen/PaStiXSupport>
#include <contrib/Eigen/SparseExtra>

template <typename Scalar>
void test_pastix_reuse() {
  using Sparse = SparseMatrix<Scalar, RowMajor, long long>;
  using Dense = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  Sparse matrix(5, 5);
  for (Index i = 0; i < matrix.rows(); ++i) {
    matrix.insert(i, i) = Scalar(4);
    if (i > 0) matrix.insert(i, i - 1) = Scalar(-1);
    if (i + 1 < matrix.rows()) matrix.insert(i, i + 1) = Scalar(1);
  }
  Dense expected(5, 2);
  for (Index i = 0; i < expected.rows(); ++i) {
    expected(i, 0) = Scalar(i + 1);
    expected(i, 1) = Scalar(2 - i);
  }
  PastixLU<Sparse> solver;
  solver.iparm(IPARM_THREAD_NBR) = 1;
  solver.analyzePattern(matrix);
  VERIFY_IS_EQUAL(solver.info(), Success);
  for (int iteration = 0; iteration < 2; ++iteration) {
    for (Index i = 0; i < matrix.rows(); ++i) matrix.coeffRef(i, i) += Scalar(1);
    solver.factorize(matrix);
    VERIFY_IS_EQUAL(solver.info(), Success);
    const Dense rhs = matrix * expected;
    const Dense result = solver.solve(rhs);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY((result - expected).norm() <=
           128 * NumTraits<typename NumTraits<Scalar>::Real>::epsilon() * expected.norm());
  }
  solver.compute(matrix);
  VERIFY_IS_EQUAL(solver.info(), Success);
  const Dense rhs = matrix * expected;
  const Dense result = solver.solve(rhs);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_APPROX(result, expected);

  Sparse diagonal(2, 2);
  diagonal.insert(0, 0) = Scalar(2);
  diagonal.insert(1, 1) = Scalar(3);
  solver.compute(diagonal);
  VERIFY_IS_EQUAL(solver.info(), Success);
  const Matrix<Scalar, 2, 1> diagonalRhs(Scalar(2), Scalar(3));
  const Matrix<Scalar, 2, 1> diagonalResult = solver.solve(diagonalRhs);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_APPROX(diagonalResult, (Matrix<Scalar, 2, 1>::Ones()));
}

template <typename T>
void test_pastix_T() {
  PastixLLT<SparseMatrix<T, ColMajor>, Eigen::Lower> pastix_llt_lower;
  PastixLDLT<SparseMatrix<T, ColMajor>, Eigen::Lower> pastix_ldlt_lower;
  PastixLLT<SparseMatrix<T, ColMajor>, Eigen::Upper> pastix_llt_upper;
  PastixLDLT<SparseMatrix<T, ColMajor>, Eigen::Upper> pastix_ldlt_upper;
  PastixLU<SparseMatrix<T, ColMajor> > pastix_lu;

  check_sparse_spd_solving(pastix_llt_lower);
  check_sparse_spd_solving(pastix_ldlt_lower);
  check_sparse_spd_solving(pastix_llt_upper);
  check_sparse_spd_solving(pastix_ldlt_upper);
  check_sparse_square_solving(pastix_lu);

  // Some compilation check:
  pastix_llt_lower.iparm();
  pastix_llt_lower.dparm();
  pastix_ldlt_lower.iparm();
  pastix_ldlt_lower.dparm();
  pastix_lu.iparm();
  pastix_lu.dparm();
}

// There is no support for selfadjoint matrices with PaStiX.
// Complex symmetric matrices should pass though
template <typename T>
void test_pastix_T_LU() {
  PastixLU<SparseMatrix<T, ColMajor> > pastix_lu;
  check_sparse_square_solving(pastix_lu);
}

EIGEN_DECLARE_TEST(pastix_support) {
  CALL_SUBTEST_1(test_pastix_T<float>());
  CALL_SUBTEST_2(test_pastix_T<double>());
  CALL_SUBTEST_3((test_pastix_T_LU<std::complex<float> >()));
  CALL_SUBTEST_4(test_pastix_T_LU<std::complex<double>>());
  CALL_SUBTEST_1(test_pastix_reuse<float>());
  CALL_SUBTEST_2(test_pastix_reuse<double>());
  CALL_SUBTEST_3(test_pastix_reuse<std::complex<float>>());
  CALL_SUBTEST_4(test_pastix_reuse<std::complex<double>>());
}

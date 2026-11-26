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

#include <Eigen/SuperLUSupport>

// The last pivot of this rank-2 matrix is exactly zero; determinant() used to skip it and return 1.
void test_superlu_singular_determinant() {
  SparseMatrix<double> A(3, 3);
  A.insert(0, 0) = 1;
  A.insert(0, 1) = 2;
  A.insert(1, 1) = 1;
  A.insert(1, 2) = 1;
  A.insert(2, 0) = 1;
  A.insert(2, 1) = 3;
  A.insert(2, 2) = 1;
  A.makeCompressed();
  SuperLU<SparseMatrix<double> > lu(A);
  VERIFY_IS_EQUAL(lu.determinant(), 0.0);
}

// det(A) = -15 for this triangular matrix, while the pivots, or the equilibration scales, of its first three columns
// overflow together.
void test_superlu_scaled_determinant() {
  const double big = std::ldexp(1.0, 400);
  const double diagonal[6] = {big, 3 * big, big, 1 / big, -5 / big, 1 / big};
  SparseMatrix<double> A(6, 6);
  for (int j = 0; j < 6; ++j) {
    A.insert(j, j) = diagonal[j];
    if (j > 0) A.insert(j - 1, j) = diagonal[j] / 4;
  }
  A.makeCompressed();
  SuperLU<SparseMatrix<double> > lu(A);
  VERIFY_IS_EQUAL(lu.info(), Success);
  // One rounding for each of the 6 pivots and the 12 equilibration scales.
  VERIFY(numext::abs(lu.determinant() + 15.0) <= 4 * 18 * NumTraits<double>::epsilon() * 15.0);
}

// Each scalar type has its own SuperLU routines, and a row-major matrix is factorized as its transpose.
template <typename Scalar, int Options>
void test_superlu_factorization() {
  using DenseMatrix = Matrix<Scalar, 3, 3>;
  using DenseVector = Matrix<Scalar, 3, 1>;
  DenseMatrix dense;
  dense << 1, 2, 0, 0, 1, 1, 1, 3, 1;  // The third row is the sum of the other two.
  SparseMatrix<Scalar, Options> A = dense.sparseView();
  SuperLU<SparseMatrix<Scalar, Options> > lu(A);
  // Under LeakSanitizer this also checks that a factorization that stops at a zero pivot frees what it allocated.
  VERIFY_IS_EQUAL(lu.info(), NumericalIssue);

  dense(2, 2) = 3;
  A = dense.sparseView();
  lu.compute(A);
  VERIFY_IS_EQUAL(lu.info(), Success);
  const DenseVector x = DenseVector::Random();
  const DenseVector b = dense * x;
  VERIFY_IS_APPROX(lu.solve(b).eval(), x);
}

#ifdef EIGEN_SUPERLU_HAS_FACTORIZATION_STEPS
// SuperLU::factorize() always runs with ConditionNumber == NO, so this calls the factorization itself.
int superlu_factorize_info(yes_no_t condition_number) {
  SparseMatrix<double> A(2, 2);
  A.insert(0, 0) = 1;
  A.insert(0, 1) = 1;
  A.insert(1, 1) = 1e-30;
  A.makeCompressed();
  SluMatrix sluA = internal::asSluMatrix(A);

  superlu_options_t options;
  set_default_options(&options);
  options.Equil = NO;  // Scaling the second row would make A well conditioned.
  options.ConditionNumber = condition_number;
  int perm_c[2], perm_r[2], etree[2];
  double R[2], C[2];
  char equed;
  SuperMatrix L, U;
  SuperLUStat_t stats;
  int info = -1;
  StatInit(&stats);
  internal::superlu_factorize<double>(&options, &sluA, perm_c, perm_r, etree, &equed, R, C, &L, &U, &stats, &info);
  StatFree(&stats);
  Destroy_SuperNode_Matrix(&L);
  Destroy_CompCol_Matrix(&U);
  return info;
}

void test_superlu_condition_number() {
  VERIFY_IS_EQUAL(superlu_factorize_info(NO), 0);
  // info == n + 1 reports a matrix that is singular to working precision.
  VERIFY_IS_EQUAL(superlu_factorize_info(YES), 3);
}
#endif

EIGEN_DECLARE_TEST(superlu_support) {
  SuperLU<SparseMatrix<double> > superlu_double_colmajor;
  SuperLU<SparseMatrix<std::complex<double> > > superlu_cplxdouble_colmajor;
  CALL_SUBTEST_1(check_sparse_square_solving(superlu_double_colmajor));
  CALL_SUBTEST_2(check_sparse_square_solving(superlu_cplxdouble_colmajor));
  CALL_SUBTEST_1(check_sparse_square_determinant(superlu_double_colmajor));
  CALL_SUBTEST_2(check_sparse_square_determinant(superlu_cplxdouble_colmajor));
  CALL_SUBTEST_1(test_superlu_singular_determinant());
  CALL_SUBTEST_1(test_superlu_scaled_determinant());
  CALL_SUBTEST_1((test_superlu_factorization<double, ColMajor>()));
  CALL_SUBTEST_1((test_superlu_factorization<double, RowMajor>()));
  CALL_SUBTEST_1((test_superlu_factorization<float, ColMajor>()));
  CALL_SUBTEST_2((test_superlu_factorization<std::complex<double>, RowMajor>()));
  CALL_SUBTEST_2((test_superlu_factorization<std::complex<float>, ColMajor>()));
#ifdef EIGEN_SUPERLU_HAS_FACTORIZATION_STEPS
  CALL_SUBTEST_1(test_superlu_condition_number());
#endif
}

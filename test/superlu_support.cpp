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

// SuperLU equilibrates a matrix with badly scaled rows or tiny entries, and then scales a copy of the right-hand side,
// which it reads as column-major. The copy used to keep the storage order of the right-hand side.
template <typename Solver>
void test_superlu_row_major_rhs_equilibrated(Solver& solver) {
  using RowMajorMatrix = Matrix<double, Dynamic, Dynamic, RowMajor>;
  SparseMatrix<double> A(2, 2);
  A.insert(0, 0) = 2e-8;
  A.insert(1, 0) = 1;
  A.insert(1, 1) = 2;
  A.makeCompressed();
  solver.compute(A);
  VERIFY_IS_EQUAL(solver.info(), Success);
  for (Index cols : {1, 2, 40}) {
    const RowMajorMatrix B = RowMajorMatrix::Random(2, cols);
    const RowMajorMatrix X = solver.solve(B);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_IS_APPROX(A * X, B);
  }

  // A row vector is row-major as well.
  SparseMatrix<double> tiny(1, 1);
  tiny.insert(0, 0) = 1e-300;
  tiny.makeCompressed();
  solver.compute(tiny);
  VERIFY_IS_EQUAL(solver.info(), Success);
  const RowVectorXd b = RowVectorXd::Random(40);
  const RowVectorXd x = solver.solve(b);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_APPROX(tiny * x, b);
}

// SuperLU finds no pivot for the empty last row and leaves the row permutation and the last supernode incomplete.
// determinant() and solve() used to go on with them and write out of bounds, and solve() reset info() to Success.
void test_superlu_failed_factorization() {
  SparseMatrix<double> A(4, 4);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) A.insert(i, j) = (i == j) ? i + 2 : 1;
  A.makeCompressed();
  SuperLU<SparseMatrix<double> > lu(A);
  VERIFY_IS_EQUAL(lu.info(), NumericalIssue);
  VERIFY_IS_EQUAL(lu.determinant(), 0.0);
  VectorXd b = VectorXd::Ones(4);
  VectorXd x = lu.solve(b);
  VERIFY_IS_EQUAL(lu.info(), NumericalIssue);
  VERIFY((x.array() == 0.0).all());

  // A later successful factorization makes the solver usable again.
  for (int j = 0; j < 4; ++j) A.insert(3, j) = (j == 3) ? 5 : 1;
  A.makeCompressed();
  lu.compute(A);
  VERIFY_IS_EQUAL(lu.info(), Success);
  x = lu.solve(b);
  VERIFY_IS_EQUAL(lu.info(), Success);
  VERIFY_IS_APPROX(A * x, b);
}

// SuperLU is handed one error bound per right-hand side and one scaling factor per row and column: none of these
// arrays has a first element when the right-hand side has no column or the matrix is 0 x 0.
template <typename Scalar>
void test_superlu_empty() {
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  SparseMatrix<Scalar> A(3, 3);
  A.setIdentity();
  SuperLU<SparseMatrix<Scalar> > lu(A);
  VERIFY_IS_EQUAL(lu.info(), Success);
  DenseMatrix X = lu.solve(DenseMatrix(3, 0));
  VERIFY_IS_EQUAL(lu.info(), Success);
  VERIFY_IS_EQUAL(X.rows(), Index(3));
  VERIFY_IS_EQUAL(X.cols(), Index(0));

  SparseMatrix<Scalar> A0(0, 0);
  SuperLU<SparseMatrix<Scalar> > lu0(A0);
  VERIFY_IS_EQUAL(lu0.info(), Success);
  DenseMatrix X0 = lu0.solve(DenseMatrix(0, 2));
  VERIFY_IS_EQUAL(lu0.info(), Success);
  VERIFY_IS_EQUAL(X0.rows(), Index(0));
  VERIFY_IS_EQUAL(X0.cols(), Index(2));
}

EIGEN_DECLARE_TEST(superlu_support) {
  SuperLU<SparseMatrix<double> > superlu_double_colmajor;
  SuperLU<SparseMatrix<std::complex<double> > > superlu_cplxdouble_colmajor;
  CALL_SUBTEST_1(check_sparse_square_solving(superlu_double_colmajor));
  CALL_SUBTEST_2(check_sparse_square_solving(superlu_cplxdouble_colmajor));
  CALL_SUBTEST_1(check_sparse_square_determinant(superlu_double_colmajor));
  CALL_SUBTEST_2(check_sparse_square_determinant(superlu_cplxdouble_colmajor));
  CALL_SUBTEST_1(test_superlu_singular_determinant());
  CALL_SUBTEST_1(test_superlu_scaled_determinant());
  CALL_SUBTEST_1(test_superlu_row_major_rhs_equilibrated(superlu_double_colmajor));
#ifdef EIGEN_SUPERLU_HAS_ILU
  SuperILU<SparseMatrix<double> > superilu_double_colmajor;
  // The default row permutation needs MC64, and a SuperLU built without it aborts.
  superilu_double_colmajor.options().RowPerm = NOROWPERM;
  CALL_SUBTEST_1(test_superlu_row_major_rhs_equilibrated(superilu_double_colmajor));
#endif
  CALL_SUBTEST_1(test_superlu_failed_factorization());
  CALL_SUBTEST_1(test_superlu_empty<double>());
  CALL_SUBTEST_2(test_superlu_empty<std::complex<double> >());
}

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2012 Kolja Brix <brix@igpm.rwth-aaachen.de>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template <typename T>
void test_idrs_T() {
  IDRS<SparseMatrix<T>, DiagonalPreconditioner<T> > idrs_colmajor_diag;
  IDRS<SparseMatrix<T>, IncompleteLUT<T> > idrs_colmajor_ilut;

  CALL_SUBTEST(check_sparse_square_solving(idrs_colmajor_diag));
  CALL_SUBTEST(check_sparse_square_solving(idrs_colmajor_ilut));
}

// Residual smoothing used to scale its line search by |t| instead of |t|^2 and diverged.
template <typename T>
void test_idrs_smoothing() {
  using RealScalar = typename NumTraits<T>::Real;
  const Index n = 100;
  SparseMatrix<T> A(n, n);
  for (Index i = 0; i < n; ++i) {
    A.insert(i, i) = T(4);
    if (i > 0) A.insert(i, i - 1) = T(-1);
  }
  Matrix<T, Dynamic, 1> b = Matrix<T, Dynamic, 1>::Random(n);
  IDRS<SparseMatrix<T> > solver(A);
  solver.setSmoothing(true);
  solver.setTolerance(RealScalar(1e-10));
  Matrix<T, Dynamic, 1> x = solver.solve(b);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_APPROX(A * x, b);
}

// In exact arithmetic IDR(s) reaches the solution of an n x n system within n / s cycles. The bi-orthogonalization used
// to skip the latest direction, which lost that property: these systems took 7 or more cycles, and for s = n often did
// not converge.
template <typename T>
void test_idrs_finite_termination() {
  using RealScalar = typename NumTraits<T>::Real;
  using MatrixType = Matrix<T, Dynamic, Dynamic>;
  using VectorType = Matrix<T, Dynamic, 1>;
  const Index n = 8;
  // Singular values 1, ..., n and random singular vectors, so no early convergence hides a lost termination property.
  const MatrixType q1 = HouseholderQR<MatrixType>(MatrixType::Random(n, n)).householderQ();
  const MatrixType q2 = HouseholderQR<MatrixType>(MatrixType::Random(n, n)).householderQ();
  const MatrixType A =
      q1 * Matrix<RealScalar, Dynamic, 1>::LinSpaced(n, RealScalar(1), RealScalar(n)).asDiagonal() * q2.adjoint();
  const VectorType b = VectorType::Random(n);
  for (Index s : {Index(4), n}) {
    IDRS<MatrixType, IdentityPreconditioner> solver(A);
    solver.setS(s);
    solver.setTolerance(RealScalar(1e-13));
    solver.setResidualUpdate(true);
    const VectorType x = solver.solve(b);
    VERIFY_IS_EQUAL(solver.info(), Success);
    // Rounding costs an occasional cycle: over 10^6 seeds no solve took more than n / s + 1 of them.
    VERIFY_LE(solver.iterations(), n / s + 2);
    // The recurrence residual drifts from b - A x when the random shadow space is nearly degenerate.
    VERIFY_LE((A * x - b).norm(), RealScalar(1e-6) * b.norm());
  }
}

EIGEN_DECLARE_TEST(idrs) {
  CALL_SUBTEST_1(test_idrs_T<double>());
  CALL_SUBTEST_2(test_idrs_T<std::complex<double> >());
  CALL_SUBTEST_1(test_idrs_smoothing<double>());
  CALL_SUBTEST_2(test_idrs_smoothing<std::complex<double> >());
  CALL_SUBTEST_1(test_idrs_finite_termination<double>());
  CALL_SUBTEST_2(test_idrs_finite_termination<std::complex<double> >());
}

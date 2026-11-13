// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Giacomo Po <gpo@ucla.edu>
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0
#include <cmath>

#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template <typename T>
void test_minres_T() {
  // Identity preconditioner
  MINRES<SparseMatrix<T>, Lower, IdentityPreconditioner> minres_colmajor_lower_I;
  MINRES<SparseMatrix<T>, Upper, IdentityPreconditioner> minres_colmajor_upper_I;

  // Diagonal preconditioner
  MINRES<SparseMatrix<T>, Lower, DiagonalPreconditioner<T> > minres_colmajor_lower_diag;
  MINRES<SparseMatrix<T>, Upper, DiagonalPreconditioner<T> > minres_colmajor_upper_diag;
  MINRES<SparseMatrix<T>, Lower | Upper, DiagonalPreconditioner<T> > minres_colmajor_uplo_diag;

  // call tests for SPD matrix
  CALL_SUBTEST(check_sparse_spd_solving(minres_colmajor_lower_I));
  CALL_SUBTEST(check_sparse_spd_solving(minres_colmajor_upper_I));

  CALL_SUBTEST(check_sparse_spd_solving(minres_colmajor_lower_diag));
  CALL_SUBTEST(check_sparse_spd_solving(minres_colmajor_upper_diag));
  CALL_SUBTEST(check_sparse_spd_solving(minres_colmajor_uplo_diag));

  CALL_SUBTEST(check_sparse_zero_tolerance(minres_colmajor_uplo_diag));
  CALL_SUBTEST(check_sparse_zero_tolerance(minres_colmajor_lower_I));

  // TO DO: symmetric semi-definite matrix
  // TO DO: symmetric indefinite matrix
}

void test_minres_extreme_rhs() {
  const Matrix2d mat = Matrix2d::Identity();
  const Vector2d direction = (Vector2d() << 1, -1).finished();
  MINRES<Matrix2d, Lower | Upper, IdentityPreconditioner> solver(mat);
  solver.setTolerance(1e-12);

  for (double scale : {1e-200, 1e200}) {
    const Vector2d rhs = scale * direction;
    const Vector2d guess = 0.5 * rhs;
    Vector2d x = solver.solve(rhs);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(x.allFinite());
    VERIFY_IS_APPROX(x / scale, direction);

    x = solver.solveWithGuess(rhs, guess);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(x.allFinite());
    VERIFY_IS_APPROX(x / scale, direction);
  }
}

// With a preconditioner M, MINRES minimizes the M^{-1}-norm of the residual, and the product of its Givens sines
// reduces that norm, not the 2-norm that error() documents. MINRES used to scale the initial 2-norm by that product, so
// on a symmetrically scaled system with a diagonal preconditioner it reported Success and an error() of 3e-9 for a
// tolerance of 1e-8 while |b - A x| / |b| was 2e-6. Success now implies that b - A x meets the tolerance, and error()
// is its norm.
template <typename = void>
void test_minres_true_residual() {
  using SpMat = SparseMatrix<double>;
  const Index n = 120;
  const double eps = NumTraits<double>::epsilon();
  const double tol = 1e-8;
  for (double spread : {1e2, 1e4}) {
    // D T D for the tridiagonal T = tridiag(-1, 4, -1) and a diagonal D with entries in [spread^-1/2, spread^1/2].
    VectorXd d(n);
    for (Index i = 0; i < n; ++i) d(i) = std::pow(spread, double((i * 7) % 9) / 8.0 - 0.5);
    std::vector<Triplet<double> > triplets;
    for (Index i = 0; i < n; ++i) {
      triplets.emplace_back(i, i, d(i) * d(i) * 4);
      if (i + 1 < n) {
        triplets.emplace_back(i, i + 1, -d(i) * d(i + 1));
        triplets.emplace_back(i + 1, i, -d(i) * d(i + 1));
      }
    }
    SpMat A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    const MatrixXd Aabs = A.toDense().cwiseAbs();
    for (int trial = 0; trial < 5; ++trial) {
      const VectorXd b = VectorXd::Random(n);
      MINRES<SpMat, Lower | Upper, DiagonalPreconditioner<double> > solver;
      solver.setTolerance(tol);
      solver.setMaxIterations(2000);
      solver.compute(A);
      const VectorXd x = solver.solve(b);
      VERIFY(x.allFinite());
      VERIFY_LE(solver.iterations(), Index(2000));
      const double res = (b - A * x).norm() / b.norm();
      // Evaluating b - A x in another order can change each component by about eps (|b| + |A| |x|).
      const double rounding = 10 * eps * (b.norm() + (Aabs * x.cwiseAbs()).norm()) / b.norm();
      VERIFY_IS_EQUAL(solver.info(), Success);
      VERIFY_LE(res, tol + rounding);
      VERIFY_LE(numext::abs(solver.error() - res), res / 2 + rounding);
    }
  }
}

// The 5-point Laplacian on an m x m grid, for which the CG and MINRES iteration counts are about 4.5 m.
template <typename SpMat>
SpMat minres_poisson2d(Index m) {
  std::vector<Triplet<double> > triplets;
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < m; ++j) {
      const Index k = i * m + j;
      triplets.emplace_back(k, k, 4.0);
      if (j + 1 < m) triplets.emplace_back(k, k + 1, -1.0);
      if (j > 0) triplets.emplace_back(k, k - 1, -1.0);
      if (i + 1 < m) triplets.emplace_back(k, k + m, -1.0);
      if (i > 0) triplets.emplace_back(k, k - m, -1.0);
    }
  }
  SpMat A(m * m, m * m);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

// A tolerance of eps is below the rounding error of b - A x, which MINRES evaluates to confirm convergence. It used to
// keep iterating until maxIterations() = 2 n, 1800 iterations for m = 30 instead of the 136 that stop on the recurrence
// residual. It now stops when a failed confirmation comes 5 iterations after |b - A x| last halved. The default
// tolerance is 8 sqrt(n) eps, which MINRES can confirm; the confirmation of the former default eps failed.
template <typename Preconditioner>
void test_minres_poisson() {
  using SpMat = SparseMatrix<double>;
  const double eps = NumTraits<double>::epsilon();
  for (Index m : {25, 30}) {
    const SpMat A = minres_poisson2d<SpMat>(m);
    const Index n = A.rows();
    const VectorXd b = VectorXd::Random(n);
    MINRES<SpMat, Lower | Upper, Preconditioner> solver(A);

    // The default tolerance, 8 sqrt(n) eps, can be confirmed: MINRES returns Success after about 4.5 m iterations.
    VectorXd x = solver.solve(b);
    VERIFY_IS_EQUAL(solver.tolerance(), 8 * numext::sqrt(double(n)) * eps);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_LE(solver.iterations(), 5 * m);
    VERIFY_LE((b - A * x).norm(), 2 * solver.tolerance() * b.norm());
    const Index defaultIterations = solver.iterations();

    solver.setTolerance(eps);
    x = solver.solve(b);
    VERIFY(x.allFinite());
    VERIFY(solver.info() == Success || solver.info() == NoConvergence);
    // The recurrence reaches eps after about 4.5 m iterations (136 for m = 30), and stagnation is detected within 5
    // more iterations plus the spacing of the confirmations.
    VERIFY_LE(solver.iterations(), 5 * m + 10);
    VERIFY_LE(solver.iterations(), 2 * defaultIterations);
    const double res = (b - A * x).norm() / b.norm();
    VERIFY_LE(res, double(n) * eps);
    VERIFY_LE(numext::abs(solver.error() - res), res / 2 + eps);
  }
}

// See check_sparse_small_rhs: the solvers returned NoConvergence or an inaccurate x for small |b|.
template <typename = void>
void test_minres_small_rhs() {
  MINRES<SparseMatrix<double>, Lower | Upper, DiagonalPreconditioner<double> > solver_double;
  MINRES<SparseMatrix<float>, Lower | Upper, DiagonalPreconditioner<float> > solver_float;
  check_sparse_small_rhs(solver_double);
  check_sparse_small_rhs(solver_float);
}

EIGEN_DECLARE_TEST(minres) {
  CALL_SUBTEST_1(test_minres_T<double>());
  CALL_SUBTEST_3(test_minres_extreme_rhs());
  CALL_SUBTEST_4(test_minres_true_residual<>());
  CALL_SUBTEST_4(test_minres_poisson<IdentityPreconditioner>());
  CALL_SUBTEST_4(test_minres_poisson<DiagonalPreconditioner<double> >());
  CALL_SUBTEST_4(test_minres_small_rhs<>());
}

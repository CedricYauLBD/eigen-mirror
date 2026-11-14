// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_solver.h"
#include <Eigen/IterativeLinearSolvers>

template <typename T>
void test_least_square_diagonal_preconditioner_zero_columns() {
  SparseMatrix<T, RowMajor> mat(3, 3);
  mat.insert(0, 0) = T(2);
  mat.insert(2, 2) = T(4);
  mat.makeCompressed();

  LeastSquareDiagonalPreconditioner<T> precond(mat);
  Matrix<T, 3, 1> rhs = Matrix<T, 3, 1>::Ones();
  Matrix<T, 3, 1> expected;
  expected << T(0.25), T(1), T(0.0625);

  VERIFY_IS_APPROX(precond.solve(rhs), expected);
}

template <typename T>
void test_lscg_T() {
  LeastSquaresConjugateGradient<SparseMatrix<T> > lscg_colmajor_diag;
  LeastSquaresConjugateGradient<SparseMatrix<T>, IdentityPreconditioner> lscg_colmajor_I;
  LeastSquaresConjugateGradient<SparseMatrix<T, RowMajor> > lscg_rowmajor_diag;
  LeastSquaresConjugateGradient<SparseMatrix<T, RowMajor>, IdentityPreconditioner> lscg_rowmajor_I;

  CALL_SUBTEST(check_sparse_square_solving(lscg_colmajor_diag));
  CALL_SUBTEST(check_sparse_square_solving(lscg_colmajor_I));

  CALL_SUBTEST(check_sparse_leastsquare_solving(lscg_colmajor_diag));
  CALL_SUBTEST(check_sparse_leastsquare_solving(lscg_colmajor_I));

  CALL_SUBTEST(check_sparse_square_solving(lscg_rowmajor_diag));
  CALL_SUBTEST(check_sparse_square_solving(lscg_rowmajor_I));

  CALL_SUBTEST(check_sparse_leastsquare_solving(lscg_rowmajor_diag));
  CALL_SUBTEST(check_sparse_leastsquare_solving(lscg_rowmajor_I));

  CALL_SUBTEST(check_sparse_zero_tolerance(lscg_colmajor_diag));
  CALL_SUBTEST(check_sparse_zero_tolerance(lscg_rowmajor_I));

  CALL_SUBTEST(check_sparse_default_tolerance(lscg_colmajor_diag));
}

void test_lscg_extreme_rhs() {
  const Matrix2d mat = Matrix2d::Identity();
  const Vector2d direction = (Vector2d() << 1, -1).finished();
  LeastSquaresConjugateGradient<Matrix2d, IdentityPreconditioner> solver(mat);
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

// Checks a least-squares solve at the default tolerance eps, which is below the rounding error of the normal-equation
// residual A'(b - A x) for these problems, against the dense minimum-norm solution. Past the rounding floor
// eps |A| |r| of that residual, LeastSquaresConjugateGradient used to keep iterating on rounding noise and diverge: the
// solution became non-finite or 1e57 times too large within 100 iterations, with both preconditioners, and info() was
// NoConvergence. It now stops there and reports Success.
template <typename Preconditioner>
void check_lscg_rounding_floor(const MatrixXd& A, const VectorXd& b) {
  const double eps = NumTraits<double>::epsilon();
  const VectorXd ref = A.completeOrthogonalDecomposition().solve(b);
  const VectorXd sv = JacobiSVD<MatrixXd>(A).singularValues();
  const double normA = sv(0);
  double smin = sv(0);
  for (Index i = 0; i < sv.size(); ++i)
    if (sv(i) > 1e-12 * normA) smin = sv(i);
  const double cond = normA / smin;
  // The forward error of a stable least-squares solve is about eps cond (1 + cond |r| / (|A| |x|)).
  const double bound = 100 * eps * cond * (1 + cond * (b - A * ref).norm() / (normA * ref.norm())) * ref.norm();

  LeastSquaresConjugateGradient<MatrixXd, Preconditioner> solver(A);
  solver.setMaxIterations(100);
  for (int guess = 0; guess < 2; ++guess) {
    // A guess in the row space keeps the solution the minimum-norm one.
    const VectorXd x = guess ? VectorXd(solver.solveWithGuess(b, VectorXd(0.5 * ref))) : VectorXd(solver.solve(b));
    VERIFY(x.allFinite());
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY(solver.iterations() < 100);
    VERIFY_LE((x - ref).norm(), bound);
  }
}

template <typename = void>
void test_lscg_rounding_floor() {
  // A 10 x 2 sparse problem with condition number 1.35, in which the least-squares solution is small compared with b.
  MatrixXd A = MatrixXd::Zero(10, 2);
  A(0, 0) = 5.0181430310712383;
  A(1, 0) = -0.75470211299957013;
  A(1, 1) = 6.7666925831808298;
  A(2, 0) = 0.59203245963326534;
  A(2, 1) = 0.64548853512802173;
  A(6, 0) = 0.20796378805872262;
  A(7, 0) = 0.83399352431452956;
  A(9, 0) = 0.44383881999646535;
  VectorXd b = VectorXd::Zero(10);
  b(2) = -0.31680664972850403;
  b(6) = 5.1520653518359332;
  b(8) = 0.57896647511858745;
  check_lscg_rounding_floor<LeastSquareDiagonalPreconditioner<double> >(A, b);
  check_lscg_rounding_floor<IdentityPreconditioner>(A, b);

  // A zero column.
  MatrixXd Z(9, 2);
  Z << -2, 0, 3, 0, -2, 0, -2, 0, 3, 0, -2, 0, 0, 0, 3, 0, 1, 0;
  VectorXd bz(9);
  bz << 2, -2, -3, 3, 3, 4, 2, 4, -2;
  check_lscg_rounding_floor<LeastSquareDiagonalPreconditioner<double> >(Z, bz);
  check_lscg_rounding_floor<IdentityPreconditioner>(Z, bz);

  // Two identical columns, so that A'A is singular.
  MatrixXd D(5, 2);
  D << -3, -3, -3, -3, -1, -1, -2, -2, 0, 0;
  VectorXd bd(5);
  bd << 3, -2, -4, 3, 1;
  check_lscg_rounding_floor<LeastSquareDiagonalPreconditioner<double> >(D, bd);
  check_lscg_rounding_floor<IdentityPreconditioner>(D, bd);
}

// info() is Success only if the solution meets the tolerance on the normal-equation residual, or its residual has
// reached the rounding floor c eps |A| |r|, for rank-deficient and inconsistent problems and for tolerances 1e-8, eps
// and 0, whatever the number of iterations allowed.
template <typename = void>
void test_lscg_truthful_info() {
  const double eps = NumTraits<double>::epsilon();
  const Index m = 12, n = 5;
  MatrixXd A = MatrixXd::Random(m, n);
  A.col(3) = A.col(0) + A.col(1);                              // rank n - 1
  const MatrixXd Ar = A.leftCols(2) * MatrixXd::Random(2, n);  // rank 2, for the last problem
  for (int problem = 0; problem < 2; ++problem) {
    const MatrixXd& M = problem ? Ar : A;
    const double normM = JacobiSVD<MatrixXd>(M).singularValues()(0);
    const VectorXd b = VectorXd::Random(m);  // inconsistent
    for (double tol : {1e-8, eps, 0.0}) {
      for (Index maxit : {Index(3), Index(50), Index(1000)}) {
        LeastSquaresConjugateGradient<MatrixXd> solver(M);
        solver.setTolerance(tol);
        solver.setMaxIterations(maxit);
        const VectorXd x = solver.solve(b);
        VERIFY(x.allFinite());
        VERIFY_LE(solver.iterations(), maxit);
        const VectorXd r = b - M * x;
        const double gradient = (M.adjoint() * r).norm();
        if (solver.info() == Success) {
          // The recurrence residual and the one computed here differ by rounding, hence the factor 20.
          VERIFY_LE(gradient, numext::maxi(tol * (M.adjoint() * b).norm(), 20 * eps * normM * r.norm()));
        }
        // The solution is a least-squares solution to working accuracy, or the iteration limit was reached.
        if (maxit == 1000) VERIFY_LE(gradient, 1e4 * eps * normM * (b.norm() + normM * x.norm()));
      }
    }
  }
}

// See check_sparse_small_rhs: the solvers returned NoConvergence or an inaccurate x for small |b|.
template <typename = void>
void test_lscg_small_rhs() {
  LeastSquaresConjugateGradient<SparseMatrix<double> > solver_double;
  LeastSquaresConjugateGradient<SparseMatrix<float> > solver_float;
  check_sparse_small_rhs(solver_double);
  check_sparse_small_rhs(solver_float);
}

// For a sparse matrix the gradient A'(b - A x) is computed from dot products of the few nonzeros of a column, so its
// rounding error is far below the sqrt(m) eps |A| |r| of a dense matrix with m rows, and the solver must not stop
// before it is as small as sqrt(L) eps |A| |r| for L nonzeros per column. It used to stop at the dense floor: for a
// 20000 x 200 matrix with 5 nonzeros per column the error of x was 1.8e-13, 400 times that of the solver without the
// floor.
template <typename Preconditioner>
void check_lscg_sparse_tall(const SparseMatrix<double>& A, const VectorXd& b) {
  const double eps = NumTraits<double>::epsilon();
  const MatrixXd D(A);
  const VectorXd ref = D.completeOrthogonalDecomposition().solve(b);
  const VectorXd sv = JacobiSVD<MatrixXd>(D).singularValues();
  const double normA = sv(0), sigmaMin = sv(sv.size() - 1), cond = normA / sigmaMin;
  const double length = double(A.nonZeros()) / double(A.cols());

  LeastSquaresConjugateGradient<SparseMatrix<double>, Preconditioner> solver(A);
  const VectorXd x = solver.solve(b);
  VERIFY(x.allFinite());
  VERIFY_IS_EQUAL(solver.info(), Success);
  // The gradient is at working accuracy, which bounds the error of x by |g| / sigma_min^2, besides the rounding error
  // of the reference.
  const VectorXd r = b - A * x;
  const double gradientBound = numext::sqrt(length) * eps * normA * r.norm();
  VERIFY_LE((A.adjoint() * r).norm(), gradientBound);
  VERIFY_LE((x - ref).norm(), gradientBound / (sigmaMin * sigmaMin) + 10 * eps * cond * ref.norm());
}

template <typename = void>
void test_lscg_sparse_tall() {
  const Index m = 5000, n = 50;
  std::vector<Triplet<double> > triplets;
  for (Index j = 0; j < n; ++j) {
    for (Index k = 0; k < 4; ++k)
      triplets.emplace_back((j * 997 + k * 1543 + 17) % m, j, 1.0 + double((7 * j + 3 * k) % 11) / 11.0);
    triplets.emplace_back(j * (m / n), j, 5.0);
  }
  SparseMatrix<double> A(m, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  VectorXd b(m);
  for (Index i = 0; i < m; ++i)
    b(i) = numext::sin(0.37 * double(i)) + 0.5 * numext::cos(1.3 * double(i));  // inconsistent
  check_lscg_sparse_tall<LeastSquareDiagonalPreconditioner<double> >(A, b);
  check_lscg_sparse_tall<IdentityPreconditioner>(A, b);
}

// Matrices whose entries are so large or small that |A p|^2 over- or underflows while the solution is representable.
// For a 1 x 1 matrix with a = 1e200 or 1e150, |A p|^2 overflowed to infinity, the step |g|^2 / |A p|^2 became 0 and the
// gradient floor infinite, so the solver returned x = 0 with error() = 1 and info() = Success; for diag(a, 2a) it
// returned NaN or NoConvergence. Both a sparse and a dense matrix must give info() = Success and a solution accurate to
// a few eps.
template <typename Preconditioner>
void check_lscg_extreme_matrix(double a, double rhs) {
  const double eps = NumTraits<double>::epsilon();
  SparseMatrix<double> A(2, 2);
  A.insert(0, 0) = a;
  A.insert(1, 1) = 2 * a;
  const Vector2d b = rhs * Vector2d(1, 1);
  const Vector2d ref = (rhs / a) * Vector2d(1, 0.5);

  LeastSquaresConjugateGradient<SparseMatrix<double>, Preconditioner> sparse_solver(A);
  Vector2d x = sparse_solver.solve(b);
  VERIFY_IS_EQUAL(sparse_solver.info(), Success);
  VERIFY(x.allFinite());
  VERIFY_LE((x - ref).stableNorm(), 100 * eps * ref.stableNorm());

  const Matrix2d D = A.toDense();
  LeastSquaresConjugateGradient<Matrix2d, Preconditioner> dense_solver(D);
  x = dense_solver.solve(b);
  VERIFY_IS_EQUAL(dense_solver.info(), Success);
  VERIFY(x.allFinite());
  VERIFY_LE((x - ref).stableNorm(), 100 * eps * ref.stableNorm());
}

template <typename = void>
void test_lscg_extreme_matrix() {
  for (double a : {1e200, 1e150, 1e-150, 1e-200}) {
    check_lscg_extreme_matrix<IdentityPreconditioner>(a, 1e-100);
    // The diagonal preconditioner forms the squared column norms, which overflow for a = 1e200.
    if (a != 1e200) check_lscg_extreme_matrix<LeastSquareDiagonalPreconditioner<double> >(a, 1e-100);
  }
  // A 1 x 1 matrix: the step is 1e-400 in terms of the unscaled residual for a = 1e200.
  for (double a : {1e200, 1e150}) {
    SparseMatrix<double> A(1, 1);
    A.insert(0, 0) = a;
    const VectorXd b = VectorXd::Constant(1, 1e-100);
    const double ref = 1e-100 / a;
    LeastSquaresConjugateGradient<SparseMatrix<double>, IdentityPreconditioner> solver(A);
    const VectorXd x = solver.solve(b);
    VERIFY_IS_EQUAL(solver.info(), Success);
    VERIFY_LE(numext::abs(x(0) - ref), 100 * NumTraits<double>::epsilon() * ref);
  }
}

EIGEN_DECLARE_TEST(lscg) {
  CALL_SUBTEST_1(test_lscg_T<double>());
  CALL_SUBTEST_2(test_lscg_T<std::complex<double> >());
  CALL_SUBTEST_3(test_least_square_diagonal_preconditioner_zero_columns<double>());
  CALL_SUBTEST_4(test_least_square_diagonal_preconditioner_zero_columns<std::complex<double> >());
  CALL_SUBTEST_5(test_lscg_extreme_rhs());
  CALL_SUBTEST_5(test_lscg_rounding_floor<>());
  CALL_SUBTEST_5(test_lscg_truthful_info<>());
  CALL_SUBTEST_5(test_lscg_sparse_tall<>());
  CALL_SUBTEST_5(test_lscg_small_rhs<>());
  CALL_SUBTEST_5(test_lscg_extreme_matrix<>());
  CALL_SUBTEST_5(
      (check_matrix_free_accessors<LeastSquaresConjugateGradient<MatrixFreeOperator, IdentityPreconditioner>>()));
}

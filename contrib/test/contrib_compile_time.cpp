// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of AutoDiff, NonLinearOptimization, NumericalDiff, Polynomials and StructuredMatrices. main.h
// adds instrumentation that breaks constant evaluation, so this test only compiles.

#include <contrib/Eigen/AutoDiff>
#include <contrib/Eigen/NonLinearOptimization>
#include <contrib/Eigen/NumericalDiff>
#include <contrib/Eigen/Polynomials>
#include <contrib/Eigen/StructuredMatrices>

using namespace Eigen;

// Matrix construction in a constant expression needs C++20.
#if EIGEN_COMP_CXXVER >= 20

constexpr bool near(double actual, double expected, double tol) { return numext::abs(actual - expected) <= tol; }

// Mixed derivative sizes go through both maybe_coherent_pad_helper specializations: the shorter operand is padded
// and the longer one is passed through.
constexpr bool autodiff() {
  using AD2 = AutoDiffScalar<Vector2d>;
  using AD3 = AutoDiffScalar<Vector3d>;
  const AD2 x(2.0, Vector2d(1, 0));
  const AD3 y(3.0, Vector3d(0, 1, 5));
  const AD3 sum = x + y;
  const AD3 difference = y - x;
  bool ok = sum.value() == 5 && sum.derivatives() == Vector3d(1, 1, 5);
  ok = ok && difference.value() == 1 && difference.derivatives() == Vector3d(-1, 1, 5);
  const AD2 product = x * x + x;
  return ok && product.value() == 6 && product.derivatives() == Vector2d(5, 0);
}

static_assert(autodiff());

// f(x) = (x0^2, x0 x1).
struct ProductFunctor {
  using Scalar = double;
  enum { InputsAtCompileTime = 2, ValuesAtCompileTime = 2 };
  using InputType = Vector2d;
  using ValueType = Vector2d;
  using JacobianType = Matrix2d;
  constexpr int inputs() const { return 2; }
  constexpr int values() const { return 2; }
  constexpr int operator()(const InputType& x, ValueType& f) const {
    f(0) = x(0) * x(0);
    f(1) = x(0) * x(1);
    return 0;
  }
};

template <NumericalDiffMode mode>
constexpr bool numerical_diff() {
  NumericalDiff<ProductFunctor, mode> diff;
  Matrix2d jac;
  const int nfev = diff.df(Vector2d(1.5, -2), jac);
  // Truncation error O(h) forward, O(h^2) central, with h = sqrt(eps) or cbrt(eps).
  constexpr double tol = 1e-6;
  return nfev == (mode == Forward ? 3 : 4) && near(jac(0, 0), 3, tol) && near(jac(0, 1), 0, tol) &&
         near(jac(1, 0), -2, tol) && near(jac(1, 1), 1.5, tol);
}

static_assert(numerical_diff<Forward>());
// Central mode's default step uses numext::cbrt, which only GCC folds.
#if EIGEN_COMP_GNUC_STRICT
static_assert(numerical_diff<Central>());
#endif

#if defined(__cpp_lib_constexpr_dynamic_alloc) && defined(__cpp_lib_constexpr_vector) && \
    defined(__cpp_lib_constexpr_algorithms)
// diag(1, 2, 2, 4) + 0.5 z z^T with a repeated pole, which deflates by a rotation.
constexpr bool dpr1() {
  VectorXd d(4), z(4);
  d << 4, 2, 1, 2;
  z << 1, -1, 0.5, 2;
  const double rho = 0.5;
  DPR1EigenSolver<double> es(d, rho, z);
  if (es.info() != Success) return false;
  const MatrixXd a = MatrixXd(d.asDiagonal()) + rho * z * z.transpose();
  const VectorXd& lambda = es.eigenvalues();
  const MatrixXd& v = es.eigenvectors();
  constexpr double tol = 1e-13;
  bool ok = near(lambda.sum(), a.trace(), tol);
  for (Index k = 0; k + 1 < lambda.size(); ++k) ok = ok && lambda[k] <= lambda[k + 1];
  ok = ok && (a * v - v * lambda.asDiagonal()).cwiseAbs().maxCoeff() <= tol;
  return ok && (v.transpose() * v - MatrixXd::Identity(4, 4)).cwiseAbs().maxCoeff() <= tol;
}

static_assert(dpr1());

// With diag = 1, the Gauss-Newton step is R^-1 qtb, the gradient direction g = R^T qtb, and the Cauchy step length
// sgnorm = |g|^3 / |R g|^2. A trust radius delta <= sgnorm takes the gradient step delta g / |g|; one with
// sgnorm < delta < |R^-1 qtb| takes the dogleg point of norm delta.
constexpr bool dogleg() {
  constexpr double tol = 1e-14;
  const VectorXd diag = VectorXd::Ones(2);
  MatrixXd r = MatrixXd::Identity(2, 2);
  VectorXd qtb(2), x(2);
  qtb << 3, 4;
  internal::dogleg<double>(r, diag, qtb, 1.0, x);
  bool ok = near(x[0], 0.6, tol) && near(x[1], 0.8, tol);

  // Gauss-Newton step (1, 0.5) and sgnorm = 5 sqrt(5) / 17 < 1.
  r(1, 1) = 2;
  qtb << 1, 1;
  internal::dogleg<double>(r, diag, qtb, 1.0, x);
  const Vector2d cauchy = 5.0 / 17 * Vector2d(1, 2);
  const Vector2d gauss_newton(1, 0.5);
  const double t = (x[0] - cauchy[0]) / (gauss_newton[0] - cauchy[0]);
  ok = ok && near(x.norm(), 1, tol) && t > 0 && t < 1;
  return ok && near(x[1], cauchy[1] + t * (gauss_newton[1] - cauchy[1]), tol);
}

static_assert(dogleg());

constexpr bool ridders() {
  Ridders<ProductFunctor> diff{ProductFunctor()};
  Matrix2d jac;
  Vector2d error;
  diff.df(Vector2d(1.5, -2), jac, error);
  constexpr double tol = 1e-10;
  return near(jac(0, 0), 3, tol) && near(jac(0, 1), 0, tol) && near(jac(1, 0), -2, tol) && near(jac(1, 1), 1.5, tol);
}

static_assert(ridders());

// Each operator is checked against the dense matrix built from its definition.
constexpr bool vandermonde_and_cauchy() {
  const Vector3d x(1, 2, 4), y(-1, 0.5, 3);
  const Vector3d b(1, 2, 3);
  Matrix3d vd, cd;
  for (Index i = 0; i < 3; ++i)
    for (Index j = 0; j < 3; ++j) {
      vd(i, j) = j == 0 ? 1 : vd(i, j - 1) * x(i);
      cd(i, j) = 1 / (x(i) - y(j));
    }
  const auto v = makeVandermonde(x);
  bool ok = Matrix3d(v).isApprox(vd) && near(v.determinant(), 6, 1e-12) && Vector3d(v * b).isApprox(vd * b);
  const BjorckPereyra<double> bp(v);
  ok = ok && bp.info() == Success && (vd * Vector3d(bp.solve(b))).isApprox(b);
  const auto c = makeCauchy(x, y);
  ok = ok && Matrix3d(c).isApprox(cd) && near(c.determinant(), cd.determinant(), 1e-12) &&
       Vector3d(c * b).isApprox(cd * b);
  const CauchyLU<double> lu(c);
  return ok && lu.info() == Success && (cd * Vector3d(lu.solve(b))).isApprox(b);
}

static_assert(vandermonde_and_cauchy());

constexpr bool low_rank_and_kronecker() {
  const Vector3d d(2, 3, 4), u(1, -1, 2), w(0.5, 1, -1);
  const Vector3d b(1, 2, 3);
  const Matrix3d dense = Matrix3d(d.asDiagonal()) + u * w.transpose();
  const auto dplr = makeDiagonalPlusLowRank(d, u, w);
  bool ok = Matrix3d(dplr).isApprox(dense) && near(dplr.determinant(), dense.determinant(), 1e-12) &&
            Vector3d(dplr * b).isApprox(dense * b) && (dense * dplr.solve(b)).isApprox(b);

  const Matrix2d p{{2, 1}, {0, 3}}, q{{1, -1}, {2, 1}};
  Matrix4d kd;
  for (Index i = 0; i < 4; ++i)
    for (Index j = 0; j < 4; ++j) kd(i, j) = p(i / 2, j / 2) * q(i % 2, j % 2);
  const KroneckerOperator<Matrix2d, Matrix2d> k(p, q);
  const Vector4d e(1, 2, 3, 4);
  return ok && Matrix4d(k).isApprox(kd) && near(k.determinant(), kd.determinant(), 1e-12) &&
         Vector4d(k * e).isApprox(kd * e) && (kd * k.solve(e)).isApprox(e);
}

static_assert(low_rank_and_kronecker());

// The leading 2x2 block of this Toeplitz matrix is singular, which the look-ahead steps over.
constexpr bool look_ahead_levinson() {
  VectorXd col(4), row(4);
  col << 1, 1, 2, 0.5;
  row << 1, 1, -1, 3;
  MatrixXd dense(4, 4);
  for (Index i = 0; i < 4; ++i)
    for (Index j = 0; j < 4; ++j) dense(i, j) = i >= j ? col(i - j) : row(j - i);
  const Toeplitz<double> t(col, row);
  const LookAheadLevinson<double> lev(t);
  const VectorXd b = VectorXd::LinSpaced(4, 1, 4);
  return lev.info() == Success && (dense * VectorXd(lev.solve(b))).isApprox(b);
}

static_assert(look_ahead_levinson());

// Before C++26, constant evaluation cannot form realView() of std::complex coefficients, which the complex
// eigensolver behind PolynomialSolver uses.
#if defined(__cpp_lib_tuple_like) && __cpp_lib_tuple_like >= 202311L
// (x - 1)(x - 2)(x + 3) = x^3 - 7x + 6.
constexpr bool polynomial_roots() {
  const Vector4d p(6, -7, 0, 1);
  const PolynomialSolver<double, 3> solver(p);
  bool real = false;
  const double greatest = solver.greatestRealRoot(real);
  return real && near(greatest, 2, 1e-12) && near(solver.smallestRealRoot(real), -3, 1e-12);
}

static_assert(polynomial_roots());
#endif
#endif

static_assert(!(isnan)(AutoDiffScalar<Vector2d>(1.0, Vector2d(1, 0))));

#endif

int main() { return 0; }

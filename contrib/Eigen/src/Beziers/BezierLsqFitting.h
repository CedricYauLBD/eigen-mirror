// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023-2024 Jacques Franc <jacquesfrancdev@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_BEZIER_LSQ_FITTING_H
#define EIGEN_BEZIER_LSQ_FITTING_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "BezierFitting.h"

namespace Eigen {

/**
 * \ingroup Beziers_Module
 * \brief Non-linear least-squares fitting of a Bezier curve which also optimizes the point parameters.
 *
 * For fixed parameters \f$ t \f$ the optimal control points are given by a linear least-squares problem (see
 * BezierFitting), so the fitting error only depends on \f$ t \f$ through the residual
 * \f$ r(t) = (I - A(t)A(t)^+)\,x = P^\perp(t)\,x \f$, stacked over the coordinates. This (variable projection)
 * residual is minimized with a damped Gauss-Newton method. Its Jacobian is
 * \f[ \frac{\partial r}{\partial t} = -\left( P^\perp \mathrm{diag}(\Pi x) + \Pi^T \mathrm{diag}(P^\perp x) \right),
 *     \qquad \Pi = A'(t)\,A(t)^+ \f]
 * where \f$ A' \f$ holds the derivatives of the Bernstein basis functions. The first and last parameters are kept
 * fixed, and the parameters are kept monotone between them, so that the points are traversed in order: each trial
 * step is projected onto that ordered set (projected Gauss-Newton).
 *
 * The Jacobian is dense, \f$ dn \times n \f$ for \f$ n \f$ points of dimension \f$ d \f$. Each Gauss-Newton iteration
 * forms and factorizes it in \f$ O(d\,n^3) \f$ time and \f$ O(d\,n^2) \f$ memory, so the method suits moderate
 * numbers of points.
 *
 * \sa C. F. Borges and T. A. Pastva, Total least squares fitting of Bezier and B-spline curves to ordered data,
 *     Computer Aided Geometric Design 19(4), 2002.
 *
 * \tparam BezierType The Bezier type to fit, e.g. Bezier2d.
 */
template <typename BezierType>
struct BezierLsqFitting {
  using Scalar = typename BezierType::Scalar;
  using ParameterVectorType = typename BezierType::ParameterVectorType;
  using VectorType = Matrix<Scalar, Dynamic, 1>;
  using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;

  /**
   * \brief Fits a Bezier curve to the points, starting from chord-length parameters.
   *
   * \param pts The data points, one per column. There must be at least degree+1 of them.
   * \param degree The degree of the fitted curve.
   * \param maxIterations The maximal number of Gauss-Newton iterations.
   * \param tolerance Iterations stop once the relative decrease of the squared residual, or the step, falls below it.
   */
  template <typename PointArrayType>
  static BezierType Fit(const PointArrayType& pts, Index degree, Index maxIterations = 100,
                        const Scalar& tolerance = NumTraits<Scalar>::dummy_precision()) {
    ParameterVectorType params;
    BezierFitting<BezierType>::ChordLengths(pts, params);
    return Fit(pts, degree, params, maxIterations, tolerance);
  }

  /**
   * \brief Fits a Bezier curve to the points, optimizing the given parameters in place.
   *
   * \param pts The data points, one per column. There must be at least degree+1 of them.
   * \param degree The degree of the fitted curve.
   * \param[in,out] params Initial parameters on input (one per point), optimized parameters on output.
   *                The first and last ones are left unchanged and must differ when degree >= 1. The interior ones
   *                are first projected onto the parameters monotone from params(0) to params(n-1), so the output
   *                is always monotone. If they then take fewer than degree+1 distinct values, they are
   *                reinitialized to chord lengths mapped onto [params(0), params(n-1)], or to uniform values if
   *                those repeat too.
   * \param maxIterations The maximal number of Gauss-Newton iterations.
   * \param tolerance Iterations stop once the relative decrease of the squared residual, or the step, falls below it.
   */
  template <typename PointArrayType>
  static BezierType Fit(const PointArrayType& pts, Index degree, ParameterVectorType& params, Index maxIterations = 100,
                        const Scalar& tolerance = NumTraits<Scalar>::dummy_precision());

  /**
   * \brief Returns the squared norm of the residual of the linear fit for the given parameters.
   */
  template <typename PointArrayType>
  static Scalar SquaredResidual(const PointArrayType& pts, Index degree, const ParameterVectorType& params);

  /**
   * \brief Computes the residual of the linear fit for the given parameters and its Jacobian with respect to them.
   *
   * \param[in] pts The n data points, one per column.
   * \param[in] degree The degree of the fitted curve.
   * \param[in] params One parameter per point.
   * \param[out] residual The stacked residual: entry j*n+i is the j-th coordinate of the error at the i-th point.
   * \param[out] jacobian The (Dimension*n) x n derivative of \a residual with respect to \a params.
   */
  template <typename PointArrayType>
  static void ResidualAndJacobian(const PointArrayType& pts, Index degree, const ParameterVectorType& params,
                                  VectorType& residual, MatrixType& jacobian);

 private:
  // Returns whether params takes at least count distinct values, in O(n * count).
  static bool HasDistinctParameters(const ParameterVectorType& params, Index count);
  // Euclidean projection of the interior parameters onto the set monotone from params(0) to params(n-1).
  static void ProjectOrdered(ParameterVectorType& params);
};

// With s = sign(t_{n-1} - t_0) and u = s t, isotonic regression of u_1..u_{n-2} by pool-adjacent-violators, then
// clamped to [u_0, u_{n-1}]; clamping a nondecreasing vector keeps it nondecreasing, so this is the exact projection.
// O(n), and the identity on parameters that are already ordered.
// \sa R. E. Barlow, D. J. Bartholomew, J. M. Bremner and H. D. Brunk, Statistical Inference under Order
//     Restrictions, Wiley, 1972.
template <typename BezierType>
void BezierLsqFitting<BezierType>::ProjectOrdered(ParameterVectorType& params) {
  const Index n = params.size();
  if (n <= 2) return;
  const Scalar sign = params(n - 1) < params(0) ? Scalar(-1) : Scalar(1);
  VectorType blockMean(n - 2);
  Matrix<Index, Dynamic, 1> blockSize(n - 2);
  Index numBlocks = 0;
  for (Index i = 1; i < n - 1; ++i) {
    Scalar mean = sign * params(i);
    Index size = 1;
    while (numBlocks > 0 && blockMean(numBlocks - 1) > mean) {
      --numBlocks;
      const Index merged = blockSize(numBlocks) + size;
      mean = (blockMean(numBlocks) * Scalar(blockSize(numBlocks)) + mean * Scalar(size)) / Scalar(merged);
      size = merged;
    }
    blockMean(numBlocks) = mean;
    blockSize(numBlocks) = size;
    ++numBlocks;
  }
  const Scalar lo = sign * params(0);
  const Scalar hi = sign * params(n - 1);
  for (Index b = 0, i = 1; b < numBlocks; i += blockSize(b), ++b)
    params.segment(i, blockSize(b)).setConstant(sign * numext::mini(numext::maxi(blockMean(b), lo), hi));
}

template <typename BezierType>
bool BezierLsqFitting<BezierType>::HasDistinctParameters(const ParameterVectorType& params, Index count) {
  if (count <= 1) return params.size() >= count;
  VectorType found(count);
  Index numFound = 0;
  for (Index i = 0; i < params.size(); ++i) {
    if ((found.head(numFound).array() == params(i)).any()) continue;
    found(numFound++) = params(i);
    if (numFound == count) return true;
  }
  return false;
}

template <typename BezierType>
template <typename PointArrayType>
typename BezierLsqFitting<BezierType>::Scalar BezierLsqFitting<BezierType>::SquaredResidual(
    const PointArrayType& pts, Index degree, const ParameterVectorType& params) {
  const MatrixType A = BezierFitting<BezierType>::BasisMatrix(params, degree);
  const MatrixType X = pts.matrix().transpose();
  const ColPivHouseholderQR<MatrixType> qr(A);
  return (X - A * qr.solve(X)).squaredNorm();
}

template <typename BezierType>
template <typename PointArrayType>
void BezierLsqFitting<BezierType>::ResidualAndJacobian(const PointArrayType& pts, Index degree,
                                                       const ParameterVectorType& params, VectorType& residual,
                                                       MatrixType& jacobian) {
  const Index n = pts.cols();
  const Index dim = pts.rows();
  eigen_assert(params.size() == n);

  const MatrixType A = BezierFitting<BezierType>::BasisMatrix(params, degree);
  MatrixType dA(n, degree + 1);
  for (Index i = 0; i < n; ++i) dA.row(i) = BezierType::BasisFunctionDerivatives(params(i), degree).matrix();

  const ColPivHouseholderQR<MatrixType> qr(A);
  const MatrixType pinvA = qr.solve(MatrixType::Identity(n, n));
  const MatrixType projOrth = MatrixType::Identity(n, n) - A * pinvA;  // P^perp = I - A A^+
  const MatrixType Pi = dA * pinvA;

  residual.resize(dim * n);
  jacobian.resize(dim * n, n);
  for (Index j = 0; j < dim; ++j) {
    const VectorType x = pts.matrix().row(j).transpose();
    const VectorType r = projOrth * x;
    const VectorType dx = Pi * x;  // derivative of the fitted curve at the parameters
    residual.segment(j * n, n) = r;
    jacobian.middleRows(j * n, n) = -(projOrth * dx.asDiagonal() + Pi.transpose() * r.asDiagonal());
  }
}

template <typename BezierType>
template <typename PointArrayType>
BezierType BezierLsqFitting<BezierType>::Fit(const PointArrayType& pts, Index degree, ParameterVectorType& params,
                                             Index maxIterations, const Scalar& tolerance) {
  const Index n = pts.cols();
  eigen_assert(params.size() == n);
  eigen_assert(n >= degree + 1 && "Fitting a Bezier curve of degree p requires at least p+1 points");

  ProjectOrdered(params);
  // rank A(t) = min(degree+1, #distinct t), since A = V(t) M with V Vandermonde and M invertible.
  if (degree >= 1 && !HasDistinctParameters(params, degree + 1)) {
    eigen_assert(params(0) != params(n - 1) && "The end parameters are kept fixed, so they must differ");
    ParameterVectorType init;
    BezierFitting<BezierType>::ChordLengths(pts, init);
    if (!HasDistinctParameters(init, degree + 1)) init = ParameterVectorType::LinSpaced(n, Scalar(0), Scalar(1));
    params.segment(1, n - 2) = params(0) + (params(n - 1) - params(0)) * init.segment(1, n - 2);
  }

  // With n == degree+1 distinct parameters the curve interpolates the points: nothing to optimize.
  const Index numFree = n - 2;
  if (numFree > 0 && n > degree + 1) {
    VectorType residual;
    MatrixType jacobian;
    ParameterVectorType trial;
    // The objective is always evaluated with SquaredResidual so that accepted steps are guaranteed to decrease it.
    Scalar f = SquaredResidual(pts, degree, params);
    for (Index iter = 0; iter < maxIterations; ++iter) {
      if (f <= (std::numeric_limits<Scalar>::min)()) break;
      ResidualAndJacobian(pts, degree, params, residual, jacobian);

      // Gauss-Newton step on the interior parameters (the end parameters are kept fixed).
      const CompleteOrthogonalDecomposition<MatrixType> cod(jacobian.middleCols(1, numFree));
      const VectorType step = -cod.solve(residual);
      if (!step.allFinite()) break;

      // Backtracking line search on the squared residual.
      Scalar alpha(1);
      Scalar fTrial = f;
      bool accepted = false;
      for (int ls = 0; ls < 40; ++ls) {
        trial = params;
        trial.segment(1, numFree) += alpha * step.transpose().array();
        ProjectOrdered(trial);
        // P(t + alpha s) == t means s is in the normal cone of the convex ordered set at t, hence so is beta s for
        // every beta > 0: no smaller alpha can move t, nor decrease f.
        if ((trial == params).all()) break;
        fTrial = SquaredResidual(pts, degree, trial);
        if (fTrial < f) {
          accepted = true;
          break;
        }
        alpha /= Scalar(2);
      }
      if (!accepted) break;

      const Scalar stepNorm = (trial - params).abs().maxCoeff();
      params.swap(trial);
      const Scalar decrease = f - fTrial;
      f = fTrial;
      // Stop on a negligible step, or on a negligible decrease achieved by a full (undamped) Gauss-Newton step.
      if (stepNorm <= tolerance || (alpha == Scalar(1) && decrease <= tolerance * (f + decrease))) break;
    }
  }

  return BezierFitting<BezierType>::Fit(pts, degree, params);
}

}  // namespace Eigen

#endif  // EIGEN_BEZIER_LSQ_FITTING_H

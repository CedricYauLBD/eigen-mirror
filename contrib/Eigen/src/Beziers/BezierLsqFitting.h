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
 * fixed, and the parameters are kept within the range they span.
 *
 * \sa C. F. Borges and T. A. Pastva, Total least squares fitting of Bezier and B-spline curves to ordered data,
 *     Computer Aided Geometric Design 19(4), 2002.
 *
 * \tparam BezierType The Bezier type to fit, e.g. Bezier2d.
 */
template <typename BezierType>
struct BezierLsqFitting {
  typedef typename BezierType::Scalar Scalar;
  typedef typename BezierType::ParameterVectorType ParameterVectorType;
  typedef Matrix<Scalar, Dynamic, 1> VectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

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
   *                The first and last ones are left unchanged.
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
   * \param[out] residual The stacked residual: entry j*n+i is the j-th coordinate of the error at the i-th point.
   * \param[out] jacobian The (Dimension*n) x n derivative of \a residual with respect to \a params.
   */
  template <typename PointArrayType>
  static void ResidualAndJacobian(const PointArrayType& pts, Index degree, const ParameterVectorType& params,
                                  VectorType& residual, MatrixType& jacobian);
};

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

  // With n == degree+1 the curve interpolates the points for any parameters: nothing to optimize.
  const Index numFree = n - 2;
  if (numFree > 0 && n > degree + 1) {
    const Scalar lo = numext::mini(params(0), params(n - 1));
    const Scalar hi = numext::maxi(params(0), params(n - 1));

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
        trial.segment(1, numFree) =
            (params.segment(1, numFree).matrix() + alpha * step.transpose()).cwiseMax(lo).cwiseMin(hi).array();
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

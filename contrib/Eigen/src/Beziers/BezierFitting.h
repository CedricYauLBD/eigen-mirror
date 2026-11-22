// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023-2024 Jacques Franc <jacquesfrancdev@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_BEZIER_FITTING_H
#define EIGEN_BEZIER_FITTING_H

#include <numeric>

#include "./InternalHeaderCheck.h"

#include "BezierFwd.h"

namespace Eigen {

/**
 * \ingroup Beziers_Module
 * \brief Linear least-squares fitting of a Bezier curve to an ordered set of points.
 *
 * Given points \f$ x_1, \ldots, x_n \f$ and parameters \f$ t_1, \ldots, t_n \f$, the control points are the
 * least-squares solution of \f$ A(t)\,P^T = X^T \f$ with \f$ A_{ij} = B_{j,p}(t_i) \f$. When the degree is
 * \f$ n-1 \f$ (and the parameters are distinct), the resulting curve interpolates the points.
 *
 * \tparam BezierType The Bezier type to fit, e.g. Bezier2d.
 */
template <typename BezierType>
struct BezierFitting {
  typedef typename BezierType::Scalar Scalar;
  typedef typename BezierType::ParameterVectorType ParameterVectorType;
  typedef typename BezierType::ControlPointVectorType ControlPointVectorType;
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixType;

  /**
   * \brief Computes normalized chord-length parameters of the points.
   *
   * \param[in] pts The data points, one per column.
   * \param[out] params The cumulated chord lengths normalized to [0,1]. Uniform parameters are returned when all
   *             points coincide.
   *
   * \sa Les Piegl and Wayne Tiller, The NURBS book (2nd ed.), 1997, 9.2.1 Global Curve Interpolation to Point Data
   */
  template <typename PointArrayType>
  static void ChordLengths(const PointArrayType& pts, ParameterVectorType& params);

  /**
   * \brief Assembles the n x (degree+1) matrix of Bernstein basis functions evaluated at \a params.
   */
  static MatrixType BasisMatrix(const ParameterVectorType& params, Index degree);

  /**
   * \brief Fits a Bezier curve of the given degree to the points using chord-length parameters.
   *
   * \param pts The data points, one per column. There must be at least degree+1 of them.
   * \param degree The degree of the fitted curve.
   */
  template <typename PointArrayType>
  static BezierType Fit(const PointArrayType& pts, Index degree);

  /**
   * \brief Fits a Bezier curve of the given degree to the points using the given parameters.
   *
   * \param pts The data points, one per column. There must be at least degree+1 of them.
   * \param degree The degree of the fitted curve.
   * \param params One parameter per point, usually in [0,1].
   */
  template <typename PointArrayType>
  static BezierType Fit(const PointArrayType& pts, Index degree, const ParameterVectorType& params);
};

template <typename BezierType>
template <typename PointArrayType>
void BezierFitting<BezierType>::ChordLengths(const PointArrayType& pts, ParameterVectorType& params) {
  const Index n = pts.cols();
  eigen_assert(n >= 2 && "ChordLengths requires at least two points");

  params.resize(n);
  params(0) = Scalar(0);
  params.tail(n - 1) = (pts.array().rightCols(n - 1) - pts.array().leftCols(n - 1)).matrix().colwise().norm();
  std::partial_sum(params.data(), params.data() + n, params.data());

  const Scalar total = params(n - 1);
  if (total > Scalar(0)) {
    params /= total;
  } else {
    params = ParameterVectorType::LinSpaced(n, Scalar(0), Scalar(1));
  }
  params(n - 1) = Scalar(1);
}

template <typename BezierType>
typename BezierFitting<BezierType>::MatrixType BezierFitting<BezierType>::BasisMatrix(const ParameterVectorType& params,
                                                                                      Index degree) {
  MatrixType A(params.size(), degree + 1);
  for (Index i = 0; i < params.size(); ++i) {
    auto row = A.row(i);
    internal::bernstein_basis(params(i), degree, row);
  }
  return A;
}

template <typename BezierType>
template <typename PointArrayType>
BezierType BezierFitting<BezierType>::Fit(const PointArrayType& pts, Index degree, const ParameterVectorType& params) {
  eigen_assert(degree >= 0);
  eigen_assert(params.size() == pts.cols());
  eigen_assert(pts.cols() >= degree + 1 && "Fitting a Bezier curve of degree p requires at least p+1 points");
  eigen_assert(pts.rows() == BezierType::Dimension);

  const MatrixType A = BasisMatrix(params, degree);
  const ColPivHouseholderQR<MatrixType> qr(A);
  const MatrixType ctrls = qr.solve(MatrixType(pts.matrix().transpose())).transpose();
  return BezierType(ControlPointVectorType(ctrls));
}

template <typename BezierType>
template <typename PointArrayType>
BezierType BezierFitting<BezierType>::Fit(const PointArrayType& pts, Index degree) {
  ParameterVectorType params;
  ChordLengths(pts, params);
  return Fit(pts, degree, params);
}

}  // namespace Eigen

#endif  // EIGEN_BEZIER_FITTING_H

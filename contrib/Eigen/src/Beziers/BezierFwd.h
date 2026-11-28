// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023-2024 Jacques Franc <jacquesfrancdev@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_BEZIER_FWD_H
#define EIGEN_BEZIER_FWD_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Scalar, int Dim, int Degree = Dynamic>
class Bezier;

template <typename BezierType>
struct BezierTraits;

/**
 * \ingroup Beziers_Module
 * \brief Compile-time attributes of the Bezier class.
 */
template <typename Scalar_, int Dim_, int Degree_>
struct BezierTraits<Bezier<Scalar_, Dim_, Degree_> > {
  using Scalar = Scalar_; /*!< The curve's scalar type. */
  enum { Dimension = Dim_ /*!< The curve's dimension. */ };
  enum { Degree = Degree_ /*!< The curve's degree, or Dynamic. */ };
  enum { OrderAtCompileTime = Degree_ == Dynamic ? Dynamic : Degree_ + 1 /*!< Number of control points. */ };

  /** \brief The point type the curve is representing. */
  using PointType = Array<Scalar, Dimension, 1>;

  /** \brief The data type representing the curve's control points (one per column). */
  using ControlPointVectorType = Array<Scalar, Dimension, OrderAtCompileTime>;

  /** \brief The data type used to store the values of the Bernstein basis functions. */
  using BasisVectorType = Array<Scalar, 1, OrderAtCompileTime>;

  /** \brief The data type used to store the curve's derivatives (column k holds the k-th derivative). */
  using DerivativeType = Array<Scalar, Dimension, Dynamic>;

  /** \brief The data type used to store curve parameters (one per data point). */
  using ParameterVectorType = Array<Scalar, 1, Dynamic>;

  /** \brief The data type used to store the Bernstein basis functions and their derivatives (row k holds the k-th
   * derivatives). */
  using BasisDerivativeType = Array<Scalar, Dynamic, OrderAtCompileTime>;
};

/** \ingroup Beziers_Module \brief 2D float Bezier curve with dynamic degree. */
using Bezier2f = Bezier<float, 2>;
/** \ingroup Beziers_Module \brief 2D double Bezier curve with dynamic degree. */
using Bezier2d = Bezier<double, 2>;
/** \ingroup Beziers_Module \brief 3D float Bezier curve with dynamic degree. */
using Bezier3f = Bezier<float, 3>;
/** \ingroup Beziers_Module \brief 3D double Bezier curve with dynamic degree. */
using Bezier3d = Bezier<double, 3>;

}  // namespace Eigen

#endif  // EIGEN_BEZIER_FWD_H

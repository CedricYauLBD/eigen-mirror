// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023-2024 Jacques Franc <jacquesfrancdev@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_BEZIER_H
#define EIGEN_BEZIER_H

#include "./InternalHeaderCheck.h"

#include "BezierFwd.h"

namespace Eigen {

namespace internal {

/** \internal
 * Fills \a N (of size \a degree + 1) with the Bernstein polynomials B_{j,degree}(u), j = 0..degree, using the
 * numerically stable triangular recurrence B_{j,k} = (1-u) B_{j,k-1} + u B_{j-1,k-1}
 * (Piegl & Tiller, The NURBS Book, A1.3).
 */
template <typename Scalar, typename Derived>
void bernstein_basis(const Scalar& u, Index degree, DenseBase<Derived>& N) {
  eigen_assert(degree >= 0 && N.size() == degree + 1);
  const Scalar u1 = Scalar(1) - u;
  N.coeffRef(0) = Scalar(1);
  for (Index k = 1; k <= degree; ++k) {
    Scalar saved(0);
    for (Index j = 0; j < k; ++j) {
      const Scalar tmp = N.coeff(j);
      N.coeffRef(j) = saved + u1 * tmp;
      saved = u * tmp;
    }
    N.coeffRef(k) = saved;
  }
}

}  // namespace internal

/**
 * \ingroup Beziers_Module
 * \class Bezier
 * \brief A Bezier curve of arbitrary dimension, expressed in the Bernstein basis over the parameter range [0,1].
 *
 * \f[ C(u) = \sum_{j=0}^{p} B_{j,p}(u)\, P_j, \qquad B_{j,p}(u) = \binom{p}{j} u^j (1-u)^{p-j} \f]
 *
 * \tparam Scalar_ The curve's scalar type.
 * \tparam Dim_ The curve's dimension (number of rows of a point).
 * \tparam Degree_ The curve's degree, or Dynamic (default) when it is only known at run-time.
 */
template <typename Scalar_, int Dim_, int Degree_>
class Bezier {
 public:
  typedef Scalar_ Scalar; /*!< The curve's scalar type. */
  enum { Dimension = Dim_ /*!< The curve's dimension. */ };
  enum { Degree = Degree_ /*!< The curve's degree, or Dynamic. */ };

  typedef BezierTraits<Bezier> Traits;
  typedef typename Traits::PointType PointType;
  typedef typename Traits::ControlPointVectorType ControlPointVectorType;
  typedef typename Traits::BasisVectorType BasisVectorType;
  typedef typename Traits::DerivativeType DerivativeType;
  typedef typename Traits::ParameterVectorType ParameterVectorType;

  EIGEN_STATIC_ASSERT(Dim_ > 0, YOU_MADE_A_PROGRAMMING_MISTAKE)
  EIGEN_STATIC_ASSERT(Degree_ == Dynamic || Degree_ >= 0, YOU_MADE_A_PROGRAMMING_MISTAKE)

  /**
   * \brief Creates a constant zero curve. For curves with dynamic degree the resulting degree is 0.
   */
  Bezier() : m_ctrls(ControlPointVectorType::Zero(Dimension, Degree == Dynamic ? 1 : Degree + 1)) {}

  /**
   * \brief Creates a curve from its control points.
   * \param ctrls A Dimension x (degree+1) array or matrix holding one control point per column.
   */
  template <typename OtherArrayType>
  explicit Bezier(const OtherArrayType& ctrls) : m_ctrls(ctrls) {
    eigen_assert(m_ctrls.cols() >= 1);
  }

  /** \brief Returns the control points, one per column. */
  const ControlPointVectorType& ctrls() const { return m_ctrls; }

  /** \brief Returns the degree of the curve, i.e. the number of control points minus one. */
  Index degree() const { return Degree == Dynamic ? m_ctrls.cols() - 1 : Index(Degree); }

  /** \brief Evaluates the curve at the parameter \a u (usually in [0,1]). */
  PointType operator()(Scalar u) const { return (m_ctrls.matrix() * basisFunctions(u).matrix().transpose()).array(); }

  /**
   * \brief Evaluates the curve and its derivatives up to \a order at the parameter \a u.
   * \returns A Dimension x (order+1) array whose k-th column is the k-th derivative (column 0 is the point itself).
   *          Derivatives of order greater than the degree are zero.
   */
  DerivativeType derivatives(Scalar u, Index order) const;

  /** \brief Returns the degree+1 Bernstein basis functions evaluated at \a u. */
  BasisVectorType basisFunctions(Scalar u) const { return BasisFunctions(u, degree()); }

  /** \brief Returns the first derivatives of the degree+1 Bernstein basis functions evaluated at \a u. */
  BasisVectorType basisFunctionDerivatives(Scalar u) const { return BasisFunctionDerivatives(u, degree()); }

  /** \brief Returns the Bernstein basis functions B_{j,degree}(u), j = 0..degree. */
  static BasisVectorType BasisFunctions(Scalar u, Index degree);

  /**
   * \brief Returns the first derivatives of the Bernstein basis functions of the given degree at \a u,
   * \f$ B'_{j,p}(u) = p\,(B_{j-1,p-1}(u) - B_{j,p-1}(u)) \f$.
   */
  static BasisVectorType BasisFunctionDerivatives(Scalar u, Index degree);

 private:
  ControlPointVectorType m_ctrls; /*!< Control points, one per column. */
};

template <typename Scalar_, int Dim_, int Degree_>
typename Bezier<Scalar_, Dim_, Degree_>::BasisVectorType Bezier<Scalar_, Dim_, Degree_>::BasisFunctions(Scalar u,
                                                                                                        Index degree) {
  eigen_assert(degree >= 0);
  BasisVectorType N(degree + 1);
  internal::bernstein_basis(u, degree, N);
  return N;
}

template <typename Scalar_, int Dim_, int Degree_>
typename Bezier<Scalar_, Dim_, Degree_>::BasisVectorType Bezier<Scalar_, Dim_, Degree_>::BasisFunctionDerivatives(
    Scalar u, Index degree) {
  eigen_assert(degree >= 0);
  BasisVectorType dN = BasisVectorType::Zero(degree + 1);
  if (degree == 0) return dN;

  Array<Scalar, 1, Dynamic> N(degree);
  internal::bernstein_basis(u, degree - 1, N);
  const Scalar p = Scalar(degree);
  dN.head(degree) -= p * N;
  dN.tail(degree) += p * N;
  return dN;
}

template <typename Scalar_, int Dim_, int Degree_>
typename Bezier<Scalar_, Dim_, Degree_>::DerivativeType Bezier<Scalar_, Dim_, Degree_>::derivatives(Scalar u,
                                                                                                    Index order) const {
  eigen_assert(order >= 0);
  const Index p = degree();
  DerivativeType der = DerivativeType::Zero(Dimension, order + 1);

  // The k-th derivative is a Bezier curve of degree p-k whose control points are the
  // k-th forward differences of the control points scaled by p!/(p-k)!.
  Array<Scalar, Dimension, Dynamic> diffs = m_ctrls;
  Array<Scalar, 1, Dynamic> N;
  const Index maxOrder = numext::mini(order, p);
  for (Index k = 0; k <= maxOrder; ++k) {
    const Index m = p - k;
    N.resize(m + 1);
    internal::bernstein_basis(u, m, N);
    der.col(k) = (diffs.matrix() * N.matrix().transpose()).array();
    if (k < maxOrder) {
      Array<Scalar, Dimension, Dynamic> next = Scalar(m) * (diffs.rightCols(m) - diffs.leftCols(m));
      diffs.swap(next);
    }
  }
  return der;
}

}  // namespace Eigen

#endif  // EIGEN_BEZIER_H

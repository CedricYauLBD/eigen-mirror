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

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

#include "BezierFwd.h"

namespace Eigen {

namespace internal {

/** \internal
 * Raises the Bernstein polynomials held in N(0..k-1) from degree k-1 to degree k in place:
 * B_{j,k} = (1-u) B_{j,k-1} + u B_{j-1,k-1} (Piegl & Tiller, The NURBS Book, A1.3).
 */
template <typename Scalar, typename Derived>
void bernstein_basis_step(const Scalar& u, Index k, DenseBase<Derived>& N) {
  eigen_assert(k >= 1 && N.size() > k);
  const Scalar u1 = Scalar(1) - u;
  Scalar saved(0);
  for (Index j = 0; j < k; ++j) {
    const Scalar tmp = N.coeff(j);
    N.coeffRef(j) = saved + u1 * tmp;
    saved = u * tmp;
  }
  N.coeffRef(k) = saved;
}

/** \internal
 * Fills \a N (of size \a degree + 1) with the Bernstein polynomials B_{j,degree}(u), j = 0..degree, using the
 * numerically stable triangular recurrence of bernstein_basis_step.
 */
template <typename Scalar, typename Derived>
void bernstein_basis(const Scalar& u, Index degree, DenseBase<Derived>& N) {
  eigen_assert(degree >= 0 && N.size() == degree + 1);
  N.coeffRef(0) = Scalar(1);
  for (Index k = 1; k <= degree; ++k) bernstein_basis_step(u, k, N);
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
  using Scalar = Scalar_; /*!< The curve's scalar type. */
  enum { Dimension = Dim_ /*!< The curve's dimension. */ };
  enum { Degree = Degree_ /*!< The curve's degree, or Dynamic. */ };

  using Traits = BezierTraits<Bezier>;
  using PointType = typename Traits::PointType;
  using ControlPointVectorType = typename Traits::ControlPointVectorType;
  using BasisVectorType = typename Traits::BasisVectorType;
  using BasisDerivativeType = typename Traits::BasisDerivativeType;
  using DerivativeType = typename Traits::DerivativeType;
  using ParameterVectorType = typename Traits::ParameterVectorType;

  EIGEN_STATIC_ASSERT(Dim_ > 0, YOU_MADE_A_PROGRAMMING_MISTAKE)
  EIGEN_STATIC_ASSERT(Degree_ == Dynamic || Degree_ >= 0, YOU_MADE_A_PROGRAMMING_MISTAKE)

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF(bool(ControlPointVectorType::NeedsToAlign))

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

  /**
   * \brief Returns the degree+1 Bernstein basis functions and their derivatives through \a order evaluated at \a u.
   * \sa BasisFunctionDerivatives(Scalar, Index, Index)
   */
  BasisDerivativeType basisFunctionDerivatives(Scalar u, Index order) const {
    return BasisFunctionDerivatives(u, degree(), order);
  }

  /** \brief Returns the Bernstein basis functions B_{j,degree}(u), j = 0..degree. */
  static BasisVectorType BasisFunctions(Scalar u, Index degree);

  /**
   * \brief Returns the first derivatives of the Bernstein basis functions of the given degree at \a u,
   * \f$ B'_{j,p}(u) = p\,(B_{j-1,p-1}(u) - B_{j,p-1}(u)) \f$.
   */
  static BasisVectorType BasisFunctionDerivatives(Scalar u, Index degree);

  /**
   * \brief Returns the Bernstein basis functions of the given degree and their derivatives through \a order at \a u.
   *
   * \returns An (order+1) x (degree+1) array whose row k holds \f$ B^{(k)}_{j,p}(u) \f$, j = 0..p: row 0 equals
   *          BasisFunctions(u, degree) and row 1 equals BasisFunctionDerivatives(u, degree). Rows of order greater
   *          than the degree are zero, as in derivatives(). The k-th derivative of a curve is
   *          ctrls().matrix() * row(k).matrix().transpose().
   */
  static BasisDerivativeType BasisFunctionDerivatives(Scalar u, Index degree, Index order);

  /**
   * \brief Splits the curve at \a u into two curves of the same degree, each reparameterized to [0,1]:
   * \f$ L(s) = C(u\,s) \f$ and \f$ R(s) = C(u + (1-u)\,s) \f$.
   *
   * \a u is usually in [0,1]; other values extrapolate the curve. \a left or \a right may be \c *this, which
   * reuses its storage. \a left and \a right must be distinct.
   *
   * \sa keepLeft(), keepRight()
   */
  void split(Scalar u, Bezier& left, Bezier& right) const;

  /** \brief Replaces the curve by its part on [0,u], reparameterized to [0,1]: \f$ C(s) \leftarrow C(u\,s) \f$. */
  void keepLeft(Scalar u);

  /**
   * \brief Replaces the curve by its part on [u,1], reparameterized to [0,1]: \f$ C(s) \leftarrow C(u + (1-u)\,s) \f$.
   */
  void keepRight(Scalar u);

 private:
  ControlPointVectorType m_ctrls; /*!< Control points, one per column. */
};

template <typename Scalar_, int Dim_, int Degree_>
typename Bezier<Scalar_, Dim_, Degree_>::BasisVectorType Bezier<Scalar_, Dim_, Degree_>::BasisFunctions(Scalar u,
                                                                                                        Index degree) {
  eigen_assert(degree >= 0);
  BasisVectorType N;
  N.resize(degree + 1);
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
typename Bezier<Scalar_, Dim_, Degree_>::BasisDerivativeType Bezier<Scalar_, Dim_, Degree_>::BasisFunctionDerivatives(
    Scalar u, Index degree, Index order) {
  eigen_assert(degree >= 0 && order >= 0);
  BasisDerivativeType dN = BasisDerivativeType::Zero(order + 1, degree + 1);

  // B^{(k)}_{.,p} = D_p D_{p-1} ... D_{p-k+1} B_{.,p-k},  (D_q w)_j = q (w_{j-1} - w_j),  w_{-1} = w_q = 0,
  // so a single run of the degree-raising recurrence provides every B_{.,p-k} for k = 0..min(order, p).
  const Index maxOrder = numext::mini(order, degree);
  BasisVectorType N;
  N.resize(degree + 1);
  N(0) = Scalar(1);
  for (Index q = 0; q <= degree; ++q) {
    if (q > 0) internal::bernstein_basis_step(u, q, N);
    const Index k = degree - q;
    if (k > maxOrder) continue;
    dN.row(k).head(q + 1) = N.head(q + 1);
    // D_r in place; descending j keeps w_{j-1} unmodified, and w_r is still zero.
    for (Index r = q + 1; r <= degree; ++r) {
      const Scalar p = Scalar(r);
      for (Index j = r; j > 0; --j) dN(k, j) = p * (dN(k, j - 1) - dN(k, j));
      dN(k, 0) = -p * dN(k, 0);
    }
  }
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

// de Casteljau: P_j^(r) = (1-u) P_j^(r-1) + u P_{j+1}^(r-1), r = 1..p, j = 0..p-r.
// The control points of C on [0,u] are L_r = P_0^(r); those of C on [u,1] are R_j = P_j^(p-j).

template <typename Scalar_, int Dim_, int Degree_>
void Bezier<Scalar_, Dim_, Degree_>::split(Scalar u, Bezier& left, Bezier& right) const {
  eigen_assert(&left != &right);
  const Index p = degree();
  const Scalar u1 = Scalar(1) - u;
  // Copying into right first makes the split alias-safe: afterwards only right is read, and right != left.
  right.m_ctrls = m_ctrls;
  ControlPointVectorType& w = right.m_ctrls;
  left.m_ctrls.resize(Dimension, p + 1);
  left.m_ctrls.col(0) = w.col(0);
  for (Index r = 1; r <= p; ++r) {
    for (Index j = 0; j <= p - r; ++j) w.col(j) = u1 * w.col(j) + u * w.col(j + 1);
    left.m_ctrls.col(r) = w.col(0);
  }
}

template <typename Scalar_, int Dim_, int Degree_>
void Bezier<Scalar_, Dim_, Degree_>::keepLeft(Scalar u) {
  const Index p = degree();
  const Scalar u1 = Scalar(1) - u;
  // Descending j keeps col(j-1) at level r-1; col(r) holds L_r after level r and is not written again.
  for (Index r = 1; r <= p; ++r)
    for (Index j = p; j >= r; --j) m_ctrls.col(j) = u1 * m_ctrls.col(j - 1) + u * m_ctrls.col(j);
}

template <typename Scalar_, int Dim_, int Degree_>
void Bezier<Scalar_, Dim_, Degree_>::keepRight(Scalar u) {
  const Index p = degree();
  const Scalar u1 = Scalar(1) - u;
  // col(p-r) holds R_{p-r} after level r and is not written again.
  for (Index r = 1; r <= p; ++r)
    for (Index j = 0; j <= p - r; ++j) m_ctrls.col(j) = u1 * m_ctrls.col(j) + u * m_ctrls.col(j + 1);
}

}  // namespace Eigen

#endif  // EIGEN_BEZIER_H

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023-2024 Jacques Franc <jacquesfrancdev@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <contrib/Eigen/Beziers>
// The Splines module defines similarly named fitting helpers: both modules must be usable together.
#include <contrib/Eigen/Splines>

namespace Eigen {
// Explicit instantiations to make sure every member compiles.
template class Bezier<float, 2>;
template class Bezier<double, 2>;
template class Bezier<double, 3>;
template class Bezier<double, 3, 4>;
}  // namespace Eigen

// Reference evaluation with de Casteljau's algorithm.
template <typename BezierType>
typename BezierType::PointType de_casteljau(const BezierType& bezier, typename BezierType::Scalar u) {
  typedef typename BezierType::Scalar Scalar;
  Array<Scalar, BezierType::Dimension, Dynamic> pts = bezier.ctrls();
  for (Index k = pts.cols() - 1; k > 0; --k)
    for (Index j = 0; j < k; ++j) pts.col(j) = (Scalar(1) - u) * pts.col(j) + u * pts.col(j + 1);
  return pts.col(0);
}

template <typename BezierType>
BezierType random_bezier(Index degree) {
  typedef typename BezierType::ControlPointVectorType ControlPointVectorType;
  return BezierType(ControlPointVectorType::Random(BezierType::Dimension, degree + 1));
}

Bezier2d closed_bezier2d() {
  MatrixXd ctrls(17, 2);
  ctrls << 0.019179215083188927, 0.5014426226461556, 0.07578298866809455, 0.6272287861681682, 0.23028531870112856,
      0.6585204561258422, 0.4355047785423239, 0.6051379977357872, 0.5606342893515399, 0.5657667117341052,
      0.698963448913503, 0.5137022048973284, 0.8363308600446957, 0.42067588768141195, 0.9111660386650744,
      0.33744825375290965, 0.949775359726436, 0.26688131776570834, 0.8806177491369818, 0.290794936959791,
      0.7598463060961842, 0.338646954695609, 0.6632997728043561, 0.35545974207796527, 0.49984900085024137,
      0.37782774946221076, 0.31779513364586287, 0.3888700462599447, 0.19934675444750571, 0.39717732256051197,
      0.0676894342410318, 0.4210026653285819, 0.019179215083188927, 0.5014426226461556;
  return Bezier2d(ctrls.transpose());
}

Bezier3d bezier3d() {
  MatrixXd ctrls(17, 3);
  ctrls << -0.3433, 0.0000, -0.0320, -0.2966, 0.0000, -0.1535, -0.1969, 0.0000, -0.1890, -0.0914, 0.0000, -0.1115, 0, 0,
      0, 0.1574, -0.1792, 0, 0.2845, -0.1677, 0, 0.3874, -0.1026, 0, 0.4057, 0.0034, 0, 0.3876, 0.1602, 0, 0.2986,
      0.2405, 0, 0.2033, 0.2079, 0, 0.1333, 0.1301, 0, 0.0415, 0.0410, 0, -0.1027, 0.0000, 0.1117, -0.2583, 0.0000,
      0.1289, -0.3200, 0.0000, 0.0485;
  return Bezier3d(ctrls.transpose());
}

MatrixXd sample_points2d() {
  MatrixXd points(56, 2);
  points << 0.0774842828204569, 0.517799352750809, 0.115600965224479, 0.588996763754045, 0.180471152216546,
      0.592233009708738, 0.260400777990299, 0.605177993527508, 0.30586963013179, 0.663430420711974, 0.321897521568248,
      0.770226537216828, 0.353017133541551, 0.889967637540453, 0.425626871332994, 0.919093851132686, 0.4875513481668,
      0.877022653721683, 0.521205077920086, 0.750809061488673, 0.552235915066702, 0.611650485436893, 0.57921539194086,
      0.566343042071197, 0.678869170116779, 0.55663430420712, 0.753423883656554, 0.530744336569579, 0.792710779685091,
      0.469255663430421, 0.744489907916293, 0.385113268608414, 0.639461217506396, 0.355987055016181, 0.549459684123026,
      0.333333333333333, 0.513804485549879, 0.258899676375404, 0.4802233897457, 0.142394822006472, 0.446835984472476,
      0.0453074433656957, 0.391876296314231, 0.035598705501618, 0.342485210920918, 0.0841423948220064,
      0.316474186701531, 0.226537216828479, 0.302254073553979, 0.300970873786408, 0.268212962738784, 0.388349514563107,
      0.201333236488068, 0.433656957928803, 0.101808585332785, 0.45631067961165, 0.0945529381098739, 0.509631963293767,
      0.134725372505773, 0.578757511861718, 0.199019549555863, 0.579718146129354, 0.278548930699064, 0.589764264558888,
      0.325548323493437, 0.645828064870019, 0.345096129303353, 0.75093804178338, 0.38002289259694, 0.868334319283686,
      0.452869068732226, 0.894643002395088, 0.512683374327569, 0.85087836756425, 0.541619527912392, 0.724840275056433,
      0.567513343948069, 0.58608507770063, 0.592641360336182, 0.540325878375657, 0.690903059640701, 0.527276995012184,
      0.763772752123369, 0.499085638045549, 0.800518621493206, 0.436891552334291, 0.749901877587446, 0.355307381474978,
      0.644980489622395, 0.330118789442282, 0.555150552359894, 0.310814865146435, 0.517301688560339, 0.238402339301227,
      0.480051354670408, 0.124293039078178, 0.443663541123837, 0.0293886374054204, 0.388951153240247,
      0.0216817268788538, 0.34176097436055, 0.0714171902987489, 0.320945559689243, 0.213200882375185, 0.309447943750697,
      0.287336625264556, 0.278786749479152, 0.374964920671787, 0.214181549300559, 0.422102693632091, 0.11649486355828,
      0.447954842695759;
  return points.transpose();
}

// Regression values for two reference curves.
void eval_reference_curves() {
  RowVectorXd u(5);
  u << 0., 0.0625, 0.5, 0.875, 1.;

  {
    const Bezier3d bezier = bezier3d();
    MatrixXd pts(5, 3);
    pts << -0.3433, 0, -0.032, -0.27728162213663976, -0.000411435256665096979, -0.112175483988611449,
        0.312828517150878949, 0.0254942016601562467, -0.00110005340576171885, -0.107859292148618932,
        0.0281185205109899944, 0.0728095629535183558, -0.32, 0, 0.0485;
    for (Index i = 0; i < u.size(); ++i) VERIFY_IS_APPROX(Vector3d(bezier(u(i))), Vector3d(pts.row(i).transpose()));
  }
  {
    const Bezier2d bezier = closed_bezier2d();
    MatrixXd pts(5, 2);
    pts << 0.019179215083188927, 0.5014426226461556, 0.113916819711366801, 0.585998302779115687, 0.821838153320643583,
        0.349692781272395892, 0.212353205417059404, 0.410623077085491894, 0.019179215083188927, 0.5014426226461556;
    for (Index i = 0; i < u.size(); ++i) VERIFY_IS_APPROX(Vector2d(bezier(u(i))), Vector2d(pts.row(i).transpose()));
  }
}

template <typename BezierType>
void check_evaluation() {
  typedef typename BezierType::Scalar Scalar;
  typedef typename BezierType::BasisVectorType BasisVectorType;
  typedef typename BezierType::PointType PointType;

  const Index degree = internal::random<Index>(0, 12);
  const BezierType bezier = random_bezier<BezierType>(degree);
  VERIFY_IS_EQUAL(bezier.degree(), degree);

  // Default curve: constant zero of degree 0.
  const BezierType zero;
  VERIFY_IS_EQUAL(zero.degree(), Index(0));
  VERIFY_IS_EQUAL(zero(Scalar(0.3)).matrix().norm(), Scalar(0));

  // End point interpolation.
  VERIFY_IS_APPROX(bezier(Scalar(0)), PointType(bezier.ctrls().col(0)));
  VERIFY_IS_APPROX(bezier(Scalar(1)), PointType(bezier.ctrls().col(degree)));

  for (int k = 0; k < 10; ++k) {
    const Scalar u = internal::random<Scalar>(Scalar(0), Scalar(1));

    // Bernstein basis: non-negative partition of unity matching the closed form.
    const BasisVectorType N = bezier.basisFunctions(u);
    VERIFY_IS_EQUAL(N.size(), degree + 1);
    VERIFY((N >= Scalar(0)).all());
    VERIFY_IS_APPROX(N.sum(), Scalar(1));
    for (Index j = 0; j <= degree; ++j) {
      Scalar binom(1);
      for (Index i = 0; i < j; ++i) binom = binom * Scalar(degree - i) / Scalar(i + 1);
      const Scalar expected = binom * std::pow(u, Scalar(j)) * std::pow(Scalar(1) - u, Scalar(degree - j));
      VERIFY(numext::abs(N(j) - expected) <= test_precision<Scalar>());
    }

    // Evaluation against de Casteljau.
    VERIFY_IS_APPROX(bezier(u), de_casteljau(bezier, u));
  }
}

template <typename BezierType>
void check_derivatives() {
  typedef typename BezierType::Scalar Scalar;
  typedef typename BezierType::BasisVectorType BasisVectorType;
  typedef typename BezierType::DerivativeType DerivativeType;
  typedef typename BezierType::ControlPointVectorType ControlPointVectorType;

  const Index degree = internal::random<Index>(0, 10);
  const BezierType bezier = random_bezier<BezierType>(degree);
  const Scalar u = internal::random<Scalar>(Scalar(0.1), Scalar(0.9));
  const Scalar h = Scalar(1e-6);

  // Basis derivatives sum to zero and match central differences.
  const BasisVectorType dN = bezier.basisFunctionDerivatives(u);
  VERIFY(numext::abs(dN.sum()) <= test_precision<Scalar>() * Scalar(degree + 1));
  const BasisVectorType fdN = (bezier.basisFunctions(u + h) - bezier.basisFunctions(u - h)) / (Scalar(2) * h);
  VERIFY((dN - fdN).abs().maxCoeff() <= Scalar(1e-6) * numext::maxi(Scalar(1), dN.abs().maxCoeff()));

  const Index order = degree + 2;
  const DerivativeType der = bezier.derivatives(u, order);
  VERIFY_IS_EQUAL(der.rows(), Index(BezierType::Dimension));
  VERIFY_IS_EQUAL(der.cols(), order + 1);
  VERIFY_IS_APPROX(der.col(0).matrix(), bezier(u).matrix());

  // First derivative from the basis derivatives.
  VERIFY_IS_APPROX(der.col(1).matrix(), bezier.ctrls().matrix() * dN.matrix().transpose());

  // Each derivative against central differences of the previous one.
  for (Index k = 1; k <= degree; ++k) {
    const DerivativeType dp = bezier.derivatives(u + h, k - 1), dm = bezier.derivatives(u - h, k - 1);
    const Array<Scalar, BezierType::Dimension, 1> fd = (dp.col(k - 1) - dm.col(k - 1)) / (Scalar(2) * h);
    VERIFY((der.col(k) - fd).abs().maxCoeff() <= Scalar(1e-5) * numext::maxi(Scalar(1), der.col(k).abs().maxCoeff()));
  }

  // Derivatives of order higher than the degree vanish.
  VERIFY_IS_EQUAL(der.col(degree + 1).matrix().norm(), Scalar(0));
  VERIFY_IS_EQUAL(der.col(degree + 2).matrix().norm(), Scalar(0));

  // Fixed degree curves agree with dynamic ones.
  const ControlPointVectorType c = ControlPointVectorType::Random(BezierType::Dimension, 4);
  const Bezier<Scalar, BezierType::Dimension, 3> fixed(c);
  const BezierType dynamic(c);
  VERIFY_IS_EQUAL(fixed.degree(), Index(3));
  VERIFY_IS_APPROX(fixed(u), dynamic(u));
  VERIFY_IS_APPROX(fixed.derivatives(u, 2).matrix(), dynamic.derivatives(u, 2).matrix());
}

template <typename BezierType>
void check_chord_lengths() {
  typedef typename BezierType::Scalar Scalar;
  typedef typename BezierType::ParameterVectorType ParameterVectorType;
  typedef Array<Scalar, BezierType::Dimension, Dynamic> PointArrayType;

  const Index n = internal::random<Index>(2, 30);
  const PointArrayType pts = PointArrayType::Random(BezierType::Dimension, n);
  ParameterVectorType params;
  BezierFitting<BezierType>::ChordLengths(pts, params);
  VERIFY_IS_EQUAL(params.size(), n);
  VERIFY_IS_EQUAL(params(0), Scalar(0));
  VERIFY_IS_EQUAL(params(n - 1), Scalar(1));
  VERIFY((params.tail(n - 1) >= params.head(n - 1)).all());

  // Coinciding points fall back to uniform parameters.
  const PointArrayType same = pts.col(0).replicate(1, n);
  BezierFitting<BezierType>::ChordLengths(same, params);
  VERIFY_IS_APPROX(params.matrix(), ParameterVectorType::LinSpaced(n, Scalar(0), Scalar(1)).matrix());
}

template <typename BezierType>
void check_fitting() {
  typedef typename BezierType::Scalar Scalar;
  typedef typename BezierType::ParameterVectorType ParameterVectorType;
  typedef Array<Scalar, BezierType::Dimension, Dynamic> PointArrayType;

  // Degree n-1: the curve interpolates the points.
  {
    const Index n = internal::random<Index>(2, 7);
    const PointArrayType pts = PointArrayType::Random(BezierType::Dimension, n);
    ParameterVectorType params;
    BezierFitting<BezierType>::ChordLengths(pts, params);
    const BezierType bezier = BezierFitting<BezierType>::Fit(pts, n - 1);
    VERIFY_IS_EQUAL(bezier.degree(), n - 1);
    for (Index i = 0; i < n; ++i) VERIFY_IS_APPROX(bezier(params(i)).matrix(), pts.col(i).matrix());
  }

  // Points sampled from a curve are reproduced exactly with their parameters.
  {
    const Index degree = internal::random<Index>(1, 5);
    const BezierType ref = random_bezier<BezierType>(degree);
    const Index n = 4 * (degree + 1);
    const ParameterVectorType params = ParameterVectorType::LinSpaced(n, Scalar(0), Scalar(1));
    PointArrayType pts(BezierType::Dimension, n);
    for (Index i = 0; i < n; ++i) pts.col(i) = ref(params(i));
    const BezierType bezier = BezierFitting<BezierType>::Fit(pts, degree, params);
    VERIFY_IS_APPROX(bezier.ctrls().matrix(), ref.ctrls().matrix());
  }
}

template <typename BezierType>
void check_lsq_jacobian() {
  typedef typename BezierType::Scalar Scalar;
  typedef typename BezierType::ParameterVectorType ParameterVectorType;
  typedef BezierLsqFitting<BezierType> LsqFitting;
  typedef typename LsqFitting::VectorType VectorType;
  typedef typename LsqFitting::MatrixType MatrixType;
  typedef Array<Scalar, BezierType::Dimension, Dynamic> PointArrayType;

  const Index degree = internal::random<Index>(1, 5);
  const Index n = degree + 1 + internal::random<Index>(1, 10);
  const PointArrayType pts = PointArrayType::Random(BezierType::Dimension, n);
  ParameterVectorType params;
  BezierFitting<BezierType>::ChordLengths(pts, params);

  VectorType r, rp, rm;
  MatrixType J, dummy;
  LsqFitting::ResidualAndJacobian(pts, degree, params, r, J);
  VERIFY_IS_EQUAL(J.rows(), BezierType::Dimension * n);
  VERIFY_IS_EQUAL(J.cols(), n);
  VERIFY_IS_APPROX(r.squaredNorm(), LsqFitting::SquaredResidual(pts, degree, params));

  // The residual is the linear fit error.
  const BezierType bezier = BezierFitting<BezierType>::Fit(pts, degree, params);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < BezierType::Dimension; ++j)
      VERIFY(numext::abs(r(j * n + i) - (pts(j, i) - bezier(params(i))(j))) <= test_precision<Scalar>());

  // Jacobian against central differences.
  const Scalar h = Scalar(1e-6);
  for (Index i = 0; i < n; ++i) {
    ParameterVectorType tp = params, tm = params;
    tp(i) += h;
    tm(i) -= h;
    LsqFitting::ResidualAndJacobian(pts, degree, tp, rp, dummy);
    LsqFitting::ResidualAndJacobian(pts, degree, tm, rm, dummy);
    const VectorType fd = (rp - rm) / (Scalar(2) * h);
    VERIFY((J.col(i) - fd).cwiseAbs().maxCoeff() <=
           Scalar(1e-5) * numext::maxi(Scalar(1), J.col(i).cwiseAbs().maxCoeff()));
  }
}

// Samples a curve at warped (hence non chord-length) parameters.
template <typename BezierType>
Array<typename BezierType::Scalar, BezierType::Dimension, Dynamic> sample_warped(
    const BezierType& ref, Index n, typename BezierType::ParameterVectorType& exact) {
  typedef typename BezierType::Scalar Scalar;
  const typename BezierType::ParameterVectorType t =
      BezierType::ParameterVectorType::LinSpaced(n, Scalar(0), Scalar(1));
  exact = t + Scalar(0.5) * t * (Scalar(1) - t);
  Array<Scalar, BezierType::Dimension, Dynamic> pts(BezierType::Dimension, n);
  for (Index i = 0; i < n; ++i) pts.col(i) = ref(exact(i));
  return pts;
}

// Exact recovery of a known curve and of its parameters.
void check_lsq_recovery() {
  Matrix<double, 2, 4> ctrls;
  ctrls << 0., 1. / 3., 2. / 3., 1., 0., 1., -0.5, 0.3;
  const Bezier2d ref(ctrls);
  Bezier2d::ParameterVectorType exact, params;
  const Array<double, 2, Dynamic> pts = sample_warped(ref, 30, exact);

  BezierFitting<Bezier2d>::ChordLengths(pts, params);
  VERIFY(BezierLsqFitting<Bezier2d>::SquaredResidual(pts, 3, params) > 1e-6);
  const Bezier2d bezier = BezierLsqFitting<Bezier2d>::Fit(pts, 3, params);
  VERIFY_IS_APPROX(params.matrix(), exact.matrix());
  VERIFY_IS_APPROX(bezier.ctrls().matrix(), ref.ctrls().matrix());

  // The overload starting from chord lengths gives the same curve.
  VERIFY_IS_APPROX(BezierLsqFitting<Bezier2d>::Fit(pts, 3).ctrls().matrix(), ref.ctrls().matrix());
}

template <typename BezierType>
void check_lsq_fitting() {
  typedef typename BezierType::Scalar Scalar;
  typedef typename BezierType::ParameterVectorType ParameterVectorType;
  typedef typename BezierType::ControlPointVectorType ControlPointVectorType;
  typedef BezierLsqFitting<BezierType> LsqFitting;
  typedef Array<Scalar, BezierType::Dimension, Dynamic> PointArrayType;
  const bool isDouble = NumTraits<Scalar>::epsilon() < Scalar(1e-10);

  // Points sampled from a random curve at non chord-length parameters. The problem is non-convex and
  // ill-conditioned in flat valleys, so only the fit quality is checked. The reference curve is a graph over
  // its first coordinate (no cusps nor near self-intersections) so that chord lengths start in the right basin.
  const Index degree = internal::random<Index>(2, 3);
  ControlPointVectorType ctrls = ControlPointVectorType::Random(BezierType::Dimension, degree + 1);
  ctrls.row(0) = ParameterVectorType::LinSpaced(degree + 1, Scalar(0), Scalar(1));
  const BezierType ref(ctrls);
  const Index n = 30;
  ParameterVectorType exact, params;
  const PointArrayType pts = sample_warped(ref, n, exact);

  BezierFitting<BezierType>::ChordLengths(pts, params);
  const Scalar initial = LsqFitting::SquaredResidual(pts, degree, params);
  const BezierType bezier = LsqFitting::Fit(pts, degree, params);
  const Scalar final_residual = LsqFitting::SquaredResidual(pts, degree, params);

  VERIFY_IS_EQUAL(bezier.degree(), degree);
  VERIFY(final_residual <= initial);
  VERIFY_IS_EQUAL(params(0), Scalar(0));
  VERIFY_IS_EQUAL(params(n - 1), Scalar(1));
  VERIFY((params >= Scalar(0)).all() && (params <= Scalar(1)).all());
  if (isDouble) {
    VERIFY(final_residual <= Scalar(1e-6) * initial);
    for (Index i = 0; i < n; ++i) VERIFY((bezier(params(i)) - pts.col(i)).matrix().norm() <= Scalar(1e-3));
  }

  // With degree n-1 the points are interpolated whatever the parameters: nothing to optimize.
  BezierFitting<BezierType>::ChordLengths(pts.leftCols(4), params);
  const ParameterVectorType before = params;
  LsqFitting::Fit(pts.leftCols(4), 3, params);
  VERIFY((params == before).all());
}

// Noisy real-life data (a closed curve): the residual must decrease and the parameters stay in [0,1].
void check_lsq_fitting_data2d() {
  typedef Bezier2d::ParameterVectorType ParameterVectorType;
  const MatrixXd points = sample_points2d();
  const Index n = points.cols();
  for (Index degree = 3; degree <= 9; degree += 3) {
    ParameterVectorType params;
    BezierFitting<Bezier2d>::ChordLengths(points, params);
    const double initial = BezierLsqFitting<Bezier2d>::SquaredResidual(points, degree, params);
    const Bezier2d bezier = BezierLsqFitting<Bezier2d>::Fit(points, degree, params);
    const double final_residual = BezierLsqFitting<Bezier2d>::SquaredResidual(points, degree, params);
    VERIFY_IS_EQUAL(bezier.degree(), degree);
    VERIFY(final_residual < initial);
    VERIFY((params >= 0.).all() && (params <= 1.).all());
    VERIFY_IS_EQUAL(params(0), 0.);
    VERIFY_IS_EQUAL(params(n - 1), 1.);
  }
}

template <typename BezierType>
void beziers_all() {
  CALL_SUBTEST(check_evaluation<BezierType>());
  CALL_SUBTEST(check_derivatives<BezierType>());
  CALL_SUBTEST(check_chord_lengths<BezierType>());
  CALL_SUBTEST(check_fitting<BezierType>());
  CALL_SUBTEST(check_lsq_jacobian<BezierType>());
  CALL_SUBTEST(check_lsq_fitting<BezierType>());
}

EIGEN_DECLARE_TEST(beziers) {
  CALL_SUBTEST(eval_reference_curves());
  CALL_SUBTEST(check_lsq_recovery());
  CALL_SUBTEST(check_lsq_fitting_data2d());
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST(beziers_all<Bezier2d>());
    CALL_SUBTEST(beziers_all<Bezier3d>());
    CALL_SUBTEST(check_evaluation<Bezier2f>());
    CALL_SUBTEST(check_chord_lengths<Bezier2f>());
    CALL_SUBTEST(check_fitting<Bezier3f>());
    CALL_SUBTEST(check_lsq_fitting<Bezier2f>());
  }
}

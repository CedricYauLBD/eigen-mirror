// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Hauke Heibel <hauke.heibel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <Eigen/LU>   // required for MatrixBase::determinant
#include <Eigen/QR>   // required for HouseholderQR
#include <Eigen/SVD>  // required for SVD

using namespace Eigen;

//  Constructs a random matrix from the unitary group U(size).
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> randMatrixUnitary(int size) {
  using Scalar = T;
  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  // The Q factor of the QR decomposition of a random matrix is a random unitary matrix.
  // HouseholderQR is numerically stable and always succeeds, unlike Gram-Schmidt.
  MatrixType Q = MatrixType::Random(size, size).householderQr().householderQ();
  return Q;
}

//  Constructs a random matrix from the special unitary group SU(size).
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> randMatrixSpecialUnitary(int size) {
  using Scalar = T;

  using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  // initialize unitary matrix
  MatrixType Q = randMatrixUnitary<Scalar>(size);

  // tweak the first column to make the determinant be 1
  Q.col(0) *= numext::conj(Q.determinant());

  return Q;
}

template <typename MatrixType>
void run_test(unsigned dim, int num_elements) {
  using std::abs;
  using Scalar = typename internal::traits<MatrixType>::Scalar;
  using MatrixX = Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorX = Matrix<Scalar, Eigen::Dynamic, 1>;

  // MUST be positive because in any other case det(cR_t) may become negative for
  // odd dimensions!
  const Scalar c = abs(internal::random<Scalar>());

  MatrixX R = randMatrixSpecialUnitary<Scalar>(dim);
  VectorX t = Scalar(50) * VectorX::Random(dim, 1);

  MatrixX cR_t = MatrixX::Identity(dim + 1, dim + 1);
  cR_t.block(0, 0, dim, dim) = c * R;
  cR_t.block(0, dim, dim, 1) = t;

  MatrixX src = MatrixX::Random(dim + 1, num_elements);
  src.row(dim) = Matrix<Scalar, 1, Dynamic>::Constant(num_elements, Scalar(1));

  MatrixX dst = cR_t * src;

  MatrixX cR_t_umeyama = umeyama(src.block(0, 0, dim, num_elements), dst.block(0, 0, dim, num_elements));

  const Scalar error = (cR_t_umeyama * src - dst).norm() / dst.norm();
  VERIFY(error < Scalar(40) * std::numeric_limits<Scalar>::epsilon());
}

template <typename Scalar, int Dimension>
void run_fixed_size_test(int num_elements) {
  using std::abs;
  using MatrixX = Matrix<Scalar, Dimension + 1, Dynamic>;
  using HomMatrix = Matrix<Scalar, Dimension + 1, Dimension + 1>;
  using FixedMatrix = Matrix<Scalar, Dimension, Dimension>;
  using FixedVector = Matrix<Scalar, Dimension, 1>;

  const int dim = Dimension;

  // MUST be positive because in any other case det(cR_t) may become negative for
  // odd dimensions!
  // Also if c is to small compared to t.norm(), problem is ill-posed (cf. Bug 744)
  const Scalar c = internal::random<Scalar>(0.5, 2.0);

  FixedMatrix R = randMatrixSpecialUnitary<Scalar>(dim);
  FixedVector t = Scalar(32) * FixedVector::Random(dim, 1);

  HomMatrix cR_t = HomMatrix::Identity(dim + 1, dim + 1);
  cR_t.block(0, 0, dim, dim) = c * R;
  cR_t.block(0, dim, dim, 1) = t;

  MatrixX src = MatrixX::Random(dim + 1, num_elements);
  src.row(dim) = Matrix<Scalar, 1, Dynamic>::Constant(num_elements, Scalar(1));

  MatrixX dst = cR_t * src;

  Block<MatrixX, Dimension, Dynamic> src_block(src, 0, 0, dim, num_elements);
  Block<MatrixX, Dimension, Dynamic> dst_block(dst, 0, 0, dim, num_elements);

  HomMatrix cR_t_umeyama = umeyama(src_block, dst_block);

  const Scalar error = (cR_t_umeyama * src - dst).squaredNorm();

  VERIFY(error < Scalar(16) * std::numeric_limits<Scalar>::epsilon());
}

// A covariance matrix that is not finite has no SVD. umeyama() must then return NaN and not a transformation built from
// factors that the SVD did not compute.
template <typename PointsType>
void run_non_finite_test() {
  using Scalar = typename PointsType::Scalar;
  using MatrixX = Matrix<Scalar, Dynamic, Dynamic>;

  const Index dim = 3;
  const Index num_elements = 5;

  // the origin, one point on each axis, and (1, 1, 1)
  PointsType points = PointsType::Zero(dim, num_elements);
  points.block(0, 1, dim, dim).setIdentity();
  points.col(dim + 1).setOnes();

  // finite, but the products of two demeaned coordinates overflow
  const PointsType huge_points = ((std::numeric_limits<Scalar>::max)() / Scalar(8)) * points;
  VERIFY(huge_points.allFinite());
  PointsType nan_points = points;
  nan_points(1, 2) = std::numeric_limits<Scalar>::quiet_NaN();
  PointsType inf_points = points;
  inf_points(1, 2) = std::numeric_limits<Scalar>::infinity();

  const MatrixX identity = MatrixX::Identity(dim + 1, dim + 1);

  for (bool with_scaling : {false, true}) {
    // A successful call first: it frees blocks of the sizes that the failing calls below allocate.
    const MatrixX cR_t = umeyama(points, points, with_scaling);
    VERIFY_IS_APPROX(cR_t, identity);

    const auto verify_not_estimated = [&](const PointsType& src, const PointsType& dst) {
      const MatrixX not_estimated = umeyama(src, dst, with_scaling);
      VERIFY(not_estimated.topRows(dim).array().isNaN().all());
      VERIFY_IS_EQUAL(not_estimated.row(dim), identity.row(dim));
    };
    verify_not_estimated(huge_points, huge_points);
    verify_not_estimated(nan_points, points);
    verify_not_estimated(points, nan_points);
    verify_not_estimated(inf_points, points);
    verify_not_estimated(points, inf_points);
  }
}

EIGEN_DECLARE_TEST(umeyama) {
  for (int i = 0; i < g_repeat; ++i) {
    const int num_elements = internal::random<int>(40, 500);

    // works also for dimensions bigger than 3...
    for (unsigned dim = 2; dim < 8; ++dim) {
      CALL_SUBTEST_1(run_test<MatrixXd>(dim, num_elements));
      CALL_SUBTEST_2(run_test<MatrixXf>(dim, num_elements));
    }

    CALL_SUBTEST_3((run_fixed_size_test<float, 2>(num_elements)));
    CALL_SUBTEST_4((run_fixed_size_test<float, 3>(num_elements)));
    CALL_SUBTEST_5((run_fixed_size_test<float, 4>(num_elements)));

    CALL_SUBTEST_6((run_fixed_size_test<double, 2>(num_elements)));
    CALL_SUBTEST_7((run_fixed_size_test<double, 3>(num_elements)));
    CALL_SUBTEST_8((run_fixed_size_test<double, 4>(num_elements)));
  }

  CALL_SUBTEST_1(run_non_finite_test<MatrixXd>());
  CALL_SUBTEST_2(run_non_finite_test<MatrixXf>());
  CALL_SUBTEST_7((run_non_finite_test<Matrix<double, 3, Dynamic>>()));

  // Those two calls don't compile and result in meaningful error messages!
  // umeyama(MatrixXcf(),MatrixXcf());
  // umeyama(MatrixXcd(),MatrixXcd());
}

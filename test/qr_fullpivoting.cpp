// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/QR>
#include "solverbase.h"

template <typename MatrixType>
void qr() {
  static const int Rows = MatrixType::RowsAtCompileTime, Cols = MatrixType::ColsAtCompileTime;
  Index max_size = EIGEN_TEST_MAX_SIZE;
  Index min_size = numext::maxi(1, EIGEN_TEST_MAX_SIZE / 10);
  Index rows = Rows == Dynamic ? internal::random<Index>(min_size, max_size) : Rows,
        cols = Cols == Dynamic ? internal::random<Index>(min_size, max_size) : Cols,
        cols2 = Cols == Dynamic ? internal::random<Index>(min_size, max_size) : Cols,
        rank = internal::random<Index>(1, (std::min)(rows, cols) - 1);

  using Scalar = typename MatrixType::Scalar;
  using MatrixQType = Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime>;
  MatrixType m1;
  createRandomPIMatrixOfRank(rank, rows, cols, m1);
  FullPivHouseholderQR<MatrixType> qr(m1);
  VERIFY_IS_EQUAL(rank, qr.rank());
  VERIFY_IS_EQUAL(cols - qr.rank(), qr.dimensionOfKernel());
  VERIFY(!qr.isInjective());
  VERIFY(!qr.isInvertible());
  VERIFY(!qr.isSurjective());

  MatrixType r = qr.matrixQR();

  MatrixQType q = qr.matrixQ();
  VERIFY_IS_UNITARY(q);

  // FIXME need better way to construct trapezoid
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      if (i > j) r(i, j) = Scalar(0);

  MatrixType c = qr.matrixQ() * r * qr.colsPermutation().inverse();

  VERIFY_IS_APPROX(m1, c);

  // stress the ReturnByValue mechanism
  MatrixType tmp;
  VERIFY_IS_APPROX(tmp.noalias() = qr.matrixQ() * r, (qr.matrixQ() * r).eval());

  check_solverbase<MatrixType, MatrixType>(m1, qr, rows, cols, cols2);

  {
    MatrixType m2, m3;
    Index size = rows;
    // Create a random diagonally dominant (thus invertible) matrix.
    m1 = MatrixType::Random(size, size);
    m1.diagonal().array() += Scalar(2 * size);
    qr.compute(m1);
    MatrixType m1_inv = qr.inverse();
    m3 = m1 * MatrixType::Random(size, cols2);
    m2 = qr.solve(m3);
    VERIFY_IS_APPROX(m2, m1_inv * m3);
  }
}

template <typename MatrixType>
void qr_invertible() {
  using std::abs;
  using std::log;
  using RealScalar = typename NumTraits<typename MatrixType::Scalar>::Real;
  using Scalar = typename MatrixType::Scalar;

  Index max_size = numext::mini(50, EIGEN_TEST_MAX_SIZE);
  Index min_size = numext::maxi(1, EIGEN_TEST_MAX_SIZE / 10);
  Index size = internal::random<Index>(min_size, max_size);

  MatrixType m1(size, size), m2(size, size), m3(size, size);
  m1 = MatrixType::Random(size, size);

  if (std::is_same<RealScalar, float>::value) {
    // let's build a matrix more stable to inverse
    MatrixType a = MatrixType::Random(size, size * 2);
    m1 += a * a.adjoint();
  }

  FullPivHouseholderQR<MatrixType> qr(m1);
  VERIFY(qr.isInjective());
  VERIFY(qr.isInvertible());
  VERIFY(qr.isSurjective());

  check_solverbase<MatrixType, MatrixType>(m1, qr, size, size, size);

  // now construct a matrix with prescribed determinant
  m1.setZero();
  setRandomWellConditionedDiagonal(m1);
  Scalar det = m1.diagonal().prod();
  RealScalar absdet = abs(det);
  m3 = qr.matrixQ();  // get a unitary
  m1 = m3 * m1 * m3.adjoint();
  qr.compute(m1);
  VERIFY_IS_APPROX(det, qr.determinant());
  VERIFY_IS_APPROX(absdet, qr.absDeterminant());
  VERIFY_IS_APPROX(log(absdet), qr.logAbsDeterminant());
  VERIFY_IS_APPROX(numext::sign(det), qr.signDeterminant());
}

template <typename MatrixType>
void qr_verify_assert() {
  MatrixType tmp;

  FullPivHouseholderQR<MatrixType> qr;
  VERIFY_RAISES_ASSERT(qr.matrixQR())
  VERIFY_RAISES_ASSERT(qr.solve(tmp))
  VERIFY_RAISES_ASSERT(qr.transpose().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.adjoint().solve(tmp))
  VERIFY_RAISES_ASSERT(qr.matrixQ())
  VERIFY_RAISES_ASSERT(qr.dimensionOfKernel())
  VERIFY_RAISES_ASSERT(qr.isInjective())
  VERIFY_RAISES_ASSERT(qr.isSurjective())
  VERIFY_RAISES_ASSERT(qr.isInvertible())
  VERIFY_RAISES_ASSERT(qr.inverse())
  VERIFY_RAISES_ASSERT(qr.determinant())
  VERIFY_RAISES_ASSERT(qr.absDeterminant())
  VERIFY_RAISES_ASSERT(qr.logAbsDeterminant())
  VERIFY_RAISES_ASSERT(qr.signDeterminant())
}

// A right-hand side without columns, and an empty matrix, give an empty solution of the right size; sanitizer builds
// reported a reference to the first element of an empty workspace.
template <typename Decomposition, typename MatrixType>
void qr_empty_rhs() {
  const MatrixType A = MatrixType::Random(4, 4);
  const Decomposition dec(A);
  const MatrixType empty_rhs(4, 0);
  VERIFY_IS_EQUAL(dec.solve(empty_rhs).rows(), 4);
  VERIFY_IS_EQUAL(dec.solve(empty_rhs).cols(), 0);
  VERIFY_IS_EQUAL(dec.transpose().solve(empty_rhs).rows(), 4);
  VERIFY_IS_EQUAL(dec.transpose().solve(empty_rhs).cols(), 0);
  VERIFY_IS_EQUAL(dec.adjoint().solve(empty_rhs).cols(), 0);
  const MatrixType B = MatrixType::Random(4, 3);
  const MatrixType X = dec.solve(B);
  VERIFY_IS_EQUAL(X.cols(), 3);
  VERIFY_IS_APPROX(A * X, B);

  const Decomposition empty_dec{MatrixType(0, 0)};
  VERIFY_IS_EQUAL(empty_dec.solve(MatrixType(0, 2)).rows(), 0);
  VERIFY_IS_EQUAL(empty_dec.solve(MatrixType(0, 2)).cols(), 2);
  VERIFY_IS_EQUAL(empty_dec.transpose().solve(MatrixType(0, 0)).cols(), 0);
}

// matrixQ() is rows x rows whatever the shape of MatrixType, also when an expression evaluates it into a temporary.
template <typename MatrixType>
void qr_matrixq_in_expression(Index rows, Index cols) {
  using Scalar = typename MatrixType::Scalar;
  using MatrixQXpr = typename FullPivHouseholderQR<MatrixType>::MatrixQReturnType;
  using DynamicMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  STATIC_CHECK(int(MatrixQXpr::ColsAtCompileTime) == int(MatrixType::RowsAtCompileTime));
  STATIC_CHECK(int(MatrixQXpr::MaxColsAtCompileTime) == int(MatrixType::MaxRowsAtCompileTime));

  MatrixType m1 = MatrixType::Random(rows, cols);
  FullPivHouseholderQR<MatrixType> qr(m1);

  // Assigning matrixQ() evaluates it into the destination, without a temporary.
  DynamicMatrix q = qr.matrixQ();
  VERIFY_IS_EQUAL(q.rows(), rows);
  VERIFY_IS_EQUAL(q.cols(), rows);
  VERIFY_IS_UNITARY(q);
  DynamicMatrix r = qr.matrixQR().template triangularView<Upper>();
  VERIFY_IS_APPROX(DynamicMatrix(m1), q * r * qr.colsPermutation().inverse());

  DynamicMatrix m2 = DynamicMatrix::Random(rows, 3);
  VERIFY_IS_APPROX(DynamicMatrix(qr.matrixQ().eval()), q);
  VERIFY_IS_APPROX(DynamicMatrix(qr.matrixQ() * m2), q * m2);
  VERIFY_IS_APPROX(DynamicMatrix(qr.matrixQ().adjoint() * m2), q.adjoint() * m2);
  VERIFY_IS_APPROX(DynamicMatrix(qr.matrixQ().transpose() * m2), q.transpose() * m2);
}

// PermutationIndex bounds the number of columns only: every pivot row here is beyond its range.
template <typename PermutationIndex>
void qr_tall_narrow_index() {
  using QRType = FullPivHouseholderQR<MatrixXd, PermutationIndex>;
  const Index rows = Index(NumTraits<PermutationIndex>::highest()) + 100, cols = 4;
  MatrixXd m = MatrixXd::Random(rows, cols);
  for (Index k = 0; k < cols; ++k) m(rows - 1 - k, k) = 100.0 / double(k + 1);

  QRType qr(m);
  VERIFY_IS_EQUAL(qr.rank(), cols);
  for (Index k = 0; k < cols; ++k) VERIFY_IS_EQUAL(Index(qr.rowsTranspositions()(k)), rows - 1 - k);

  MatrixXd q = qr.matrixQ();
  VERIFY_IS_UNITARY(q);
  MatrixXd r = qr.matrixQR().template triangularView<Upper>();
  VERIFY_IS_APPROX(m, q * r * qr.colsPermutation().inverse());

  MatrixXd b = MatrixXd::Random(rows, 3), c = MatrixXd::Random(cols, 3);
  MatrixXd x = qr.solve(b);
  VERIFY_IS_APPROX(m.adjoint() * (m * x), m.adjoint() * b);
  x = qr.transpose().solve(c);
  VERIFY_IS_APPROX(m.transpose() * x, c);
  x = qr.adjoint().solve(c);
  VERIFY_IS_APPROX(m.adjoint() * x, c);
}

EIGEN_DECLARE_TEST(qr_fullpivoting) {
  for (int i = 0; i < 1; i++) {
    CALL_SUBTEST_5(qr<Matrix3f>());
    CALL_SUBTEST_6(qr<Matrix3d>());
    CALL_SUBTEST_8(qr<Matrix2f>());
    CALL_SUBTEST_1(qr<MatrixXf>());
    CALL_SUBTEST_2(qr<MatrixXd>());
    CALL_SUBTEST_3(qr<MatrixXcd>());
  }

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(qr_invertible<MatrixXf>());
    CALL_SUBTEST_2(qr_invertible<MatrixXd>());
    CALL_SUBTEST_4(qr_invertible<MatrixXcf>());
    CALL_SUBTEST_3(qr_invertible<MatrixXcd>());
  }

  CALL_SUBTEST_2((qr_empty_rhs<FullPivHouseholderQR<MatrixXd>, MatrixXd>()));
  CALL_SUBTEST_4((qr_empty_rhs<FullPivHouseholderQR<MatrixXcf>, MatrixXcf>()));
  CALL_SUBTEST_1((qr_empty_rhs<FullPivHouseholderQR<MatrixXf>, MatrixXf>()));
  CALL_SUBTEST_5(qr_verify_assert<Matrix3f>());
  CALL_SUBTEST_6(qr_verify_assert<Matrix3d>());
  CALL_SUBTEST_1(qr_verify_assert<MatrixXf>());
  CALL_SUBTEST_2(qr_verify_assert<MatrixXd>());
  CALL_SUBTEST_4(qr_verify_assert<MatrixXcf>());
  CALL_SUBTEST_3(qr_verify_assert<MatrixXcd>());

  // Test problem size constructors
  CALL_SUBTEST_7(FullPivHouseholderQR<MatrixXf>(10, 20));
  CALL_SUBTEST_7((FullPivHouseholderQR<Matrix<float, 10, 20> >(10, 20)));
  CALL_SUBTEST_7((FullPivHouseholderQR<Matrix<float, 10, 20> >(Matrix<float, 10, 20>::Random())));
  CALL_SUBTEST_7((FullPivHouseholderQR<Matrix<float, 20, 10> >(20, 10)));
  CALL_SUBTEST_7((FullPivHouseholderQR<Matrix<float, 20, 10> >(Matrix<float, 20, 10>::Random())));

  CALL_SUBTEST_9((qr_matrixq_in_expression<Matrix<double, 5, 3> >(5, 3)));
  CALL_SUBTEST_9((qr_matrixq_in_expression<Matrix<float, 3, 5, RowMajor> >(3, 5)));
  CALL_SUBTEST_9((qr_matrixq_in_expression<Matrix<std::complex<double>, 6, 4> >(6, 4)));
  CALL_SUBTEST_9((qr_matrixq_in_expression<Matrix<double, 5, 1> >(5, 1)));
  CALL_SUBTEST_9((qr_matrixq_in_expression<Matrix<double, 1, 5> >(1, 5)));
  CALL_SUBTEST_9((qr_matrixq_in_expression<Matrix<double, Dynamic, Dynamic, ColMajor, 8, 3> >(8, 3)));
  CALL_SUBTEST_9((qr_matrixq_in_expression<Matrix<double, Dynamic, Dynamic, ColMajor, 8, 3> >(4, 3)));
  CALL_SUBTEST_9((qr_matrixq_in_expression<Matrix<float, Dynamic, 3, RowMajor> >(6, 3)));
  CALL_SUBTEST_9((qr_matrixq_in_expression<Matrix<std::complex<float>, 4, Dynamic> >(4, 7)));
  CALL_SUBTEST_9(qr_matrixq_in_expression<MatrixXcd>(7, 4));
  CALL_SUBTEST_10(qr_tall_narrow_index<signed char>());
  CALL_SUBTEST_10(qr_tall_narrow_index<unsigned char>());
}

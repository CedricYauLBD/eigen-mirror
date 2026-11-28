// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "matrix_functions.h"

template <typename MatrixType>
void testMatrixSqrt(const MatrixType& m) {
  MatrixType A;
  generateTestMatrix<MatrixType>::run(A, m.rows());
  MatrixType sqrtA = A.sqrt();
  VERIFY_IS_APPROX(sqrtA * sqrtA, A);
  MatrixType result(m.rows(), m.cols());
  MatrixBase<MatrixType>& destination = result;
  A.sqrt().evalTo(destination);
  VERIFY_IS_APPROX(result, sqrtA);
}

// Real matrices with complex conjugate eigenvalues: each 2x2 Schur block's root is V sqrt(D) V^-1, formed in complex
// arithmetic, which GCC 15.1 and 15.2 with FMA miscompiled for float (see EIGEN_GCC15_SLP_BUG).
template <typename Scalar>
void testMatrixSqrtComplexConjugatePairs() {
  using std::sqrt;
  using Matrix2 = Matrix<Scalar, 2, 2>;
  using Matrix4 = Matrix<Scalar, 4, 4>;
  const Scalar eps = NumTraits<Scalar>::epsilon();

  // [[a, -b], [b, a]] has eigenvalues a +- b i and principal square root [[c, -s], [s, c]] with
  // c = sqrt((a + r) / 2), s = b / (2 c), r = hypot(a, b).
  const Scalar a = internal::random<Scalar>(Scalar(0.5), Scalar(2));
  const Scalar b = internal::random<Scalar>(Scalar(0.5), Scalar(2));
  const Scalar c = sqrt((a + numext::hypot(a, b)) / Scalar(2));
  const Scalar s = b / (Scalar(2) * c);
  Matrix2 A, sqrtA_exact;
  A << a, -b, b, a;
  sqrtA_exact << c, -s, s, c;
  Matrix2 sqrtA = A.sqrt();
  VERIFY((sqrtA - sqrtA_exact).norm() <= Scalar(16) * eps * sqrtA_exact.norm());

  // A general 2x2 block [[p, q], [r, t]] has complex eigenvalues iff (p - t)^2 + 4 q r < 0; p + t > 0 keeps them off
  // the branch cut.
  const auto randomPairBlock = []() {
    Matrix2 m;
    const Scalar p = internal::random<Scalar>(Scalar(0.5), Scalar(2));
    const Scalar t = internal::random<Scalar>(Scalar(0.5), Scalar(2));
    const Scalar q = internal::random<Scalar>(Scalar(0.5), Scalar(2));
    const Scalar r = -((p - t) * (p - t) / (Scalar(4) * q) + internal::random<Scalar>(Scalar(0.25), Scalar(2)));
    m << p, q, r, t;
    return m;
  };
  A = randomPairBlock();
  sqrtA = A.sqrt();
  VERIFY((sqrtA * sqrtA - A).norm() <= Scalar(32) * eps * sqrtA.squaredNorm());

  Matrix4 B = Matrix4::Zero();
  B.template topLeftCorner<2, 2>() = randomPairBlock();
  B.template bottomRightCorner<2, 2>() = randomPairBlock();
  Matrix4 sqrtB = B.sqrt();
  VERIFY((sqrtB * sqrtB - B).norm() <= Scalar(32) * eps * sqrtB.squaredNorm());
  const Matrix2 upper = sqrtB.template topRightCorner<2, 2>(), lower = sqrtB.template bottomLeftCorner<2, 2>();
  VERIFY(upper.isZero());
  VERIFY(lower.isZero());
}

// RealSchur computes no U for an argument with an infinite or NaN entry, and sqrt() read U without checking info().
// Below the first subdiagonal such an entry does not reach the square root of T either, so the result was finite.
template <typename MatrixType>
void testNonFiniteEntry(Index size) {
  using Scalar = typename MatrixType::Scalar;
  const Index positions[3][2] = {{size - 1, 0}, {0, size - 1}, {size / 2, size / 2}};
  const Scalar inf = NumTraits<Scalar>::infinity();
  for (Scalar value : {NumTraits<Scalar>::quiet_NaN(), inf, -inf}) {
    for (const auto& position : positions) {
      MatrixType A = MatrixType::Identity(size, size);
      A(position[0], position[1]) = value;
      MatrixType sqrtA = A.sqrt();
      VERIFY(sqrtA.array().isNaN().all());
    }
  }
}

// EigenSolver forms the eigenvalues (a + d) / 2 +- i z of a 2x2 diagonal block [a b; c d] of T as d + (a - d) / 2.
// When a - d overflows it computes no eigenvectors, and the square root of the block read them anyway.
template <typename MatrixType>
void testDiagonalBlockOverflow() {
  using Scalar = typename MatrixType::Scalar;
  const Scalar big = NumTraits<Scalar>::highest();
  MatrixType A(2, 2);
  A << Scalar(0.6) * big, Scalar(0.9) * big, Scalar(-0.9) * big, Scalar(-0.6) * big;
  MatrixType sqrtA = A.sqrt();
  VERIFY(sqrtA.array().isNaN().all());
}

EIGEN_DECLARE_TEST(matrix_square_root) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(testMatrixSqrt(Matrix3cf()));
    CALL_SUBTEST_2(testMatrixSqrt(MatrixXcd(12, 12)));
    CALL_SUBTEST_3(testMatrixSqrt(Matrix4f()));
    CALL_SUBTEST_4(testMatrixSqrt(Matrix<double, Dynamic, Dynamic, RowMajor>(9, 9)));
    CALL_SUBTEST_5(testMatrixSqrt(Matrix<float, 1, 1>()));
    CALL_SUBTEST_5(testMatrixSqrt(Matrix<std::complex<float>, 1, 1>()));
    CALL_SUBTEST_6(testMatrixSqrtComplexConjugatePairs<float>());
    CALL_SUBTEST_6(testMatrixSqrtComplexConjugatePairs<double>());
  }
  CALL_SUBTEST_4((testNonFiniteEntry<Matrix<double, Dynamic, Dynamic, RowMajor>>(9)));
  CALL_SUBTEST_4((testDiagonalBlockOverflow<Matrix<double, Dynamic, Dynamic, RowMajor>>()));
  CALL_SUBTEST_5((testNonFiniteEntry<Matrix4f>(4)));
  CALL_SUBTEST_5((testNonFiniteEntry<Matrix<float, 1, 1>>(1)));
}

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

static long int nb_temporaries;

inline void on_temporary_creation() {
  // here's a great place to set a breakpoint when debugging failures in this test!
  nb_temporaries++;
}

#define EIGEN_SPARSE_CREATE_TEMPORARY_PLUGIN \
  { on_temporary_creation(); }

#include "sparse.h"
#include <map>
#include <set>

#define EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW
#include "AnnoyingScalar.h"
#include "NonCommutativeScalar.h"

#define VERIFY_EVALUATION_COUNT(XPR, N)                                                   \
  {                                                                                       \
    nb_temporaries = 0;                                                                   \
    CALL_SUBTEST(XPR);                                                                    \
    if (nb_temporaries != N) std::cerr << "nb_temporaries == " << nb_temporaries << "\n"; \
    VERIFY((#XPR) && nb_temporaries == N);                                                \
  }

template <typename Lhs, typename Rhs, typename = void>
struct has_product : std::false_type {};

template <typename Lhs, typename Rhs>
struct has_product<Lhs, Rhs, internal::void_t<decltype(std::declval<const Lhs&>() * std::declval<const Rhs&>())>>
    : std::true_type {};

template <typename SparseMatrixType>
void sparse_structured_view_product_sfinae() {
  using Scalar = typename SparseMatrixType::Scalar;
  using DenseMatrixType = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVectorType = Matrix<Scalar, Dynamic, 1>;
  using TriangularViewType = decltype(std::declval<SparseMatrixType&>().template triangularView<Lower>());
  using SelfAdjointViewType = decltype(std::declval<SparseMatrixType&>().template selfadjointView<Lower>());
  using DiagonalType = decltype(std::declval<const DenseVectorType&>().asDiagonal());

  STATIC_CHECK((has_product<TriangularViewType, SparseMatrixType>::value));
  STATIC_CHECK((has_product<SparseMatrixType, TriangularViewType>::value));
  STATIC_CHECK((has_product<TriangularViewType, DenseMatrixType>::value));
  STATIC_CHECK((has_product<DenseMatrixType, TriangularViewType>::value));
  STATIC_CHECK((has_product<TriangularViewType, DiagonalType>::value));
  STATIC_CHECK((has_product<DiagonalType, TriangularViewType>::value));

  STATIC_CHECK((has_product<SelfAdjointViewType, SparseMatrixType>::value));
  STATIC_CHECK((has_product<SparseMatrixType, SelfAdjointViewType>::value));
  STATIC_CHECK((has_product<SelfAdjointViewType, DenseMatrixType>::value));
  STATIC_CHECK((has_product<DenseMatrixType, SelfAdjointViewType>::value));
  STATIC_CHECK((has_product<SelfAdjointViewType, DiagonalType>::value));
  STATIC_CHECK((has_product<DiagonalType, SelfAdjointViewType>::value));
}

template <typename SparseMatrixType>
void sparse_product() {
  using StorageIndex = typename SparseMatrixType::StorageIndex;
  Index n = 100;
  const Index rows = internal::random<Index>(1, n);
  const Index cols = internal::random<Index>(1, n);
  const Index depth = internal::random<Index>(1, n);
  using Scalar = typename SparseMatrixType::Scalar;
  enum { Flags = SparseMatrixType::Flags };

  double density = (std::max)(8. / (rows * cols), 0.2);
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;
  using RowDenseVector = Matrix<Scalar, 1, Dynamic>;
  using ColSpVector = SparseVector<Scalar, 0, StorageIndex>;
  using RowSpVector = SparseVector<Scalar, RowMajor, StorageIndex>;

  Scalar s1 = internal::random<Scalar>();
  Scalar s2 = internal::random<Scalar>();

  // test matrix-matrix product
  {
    DenseMatrix refMat2 = DenseMatrix::Zero(rows, depth);
    DenseMatrix refMat2t = DenseMatrix::Zero(depth, rows);
    DenseMatrix refMat3 = DenseMatrix::Zero(depth, cols);
    DenseMatrix refMat3t = DenseMatrix::Zero(cols, depth);
    DenseMatrix refMat4 = DenseMatrix::Zero(rows, cols);
    DenseMatrix refMat4t = DenseMatrix::Zero(cols, rows);
    DenseMatrix refMat5 = DenseMatrix::Random(depth, cols);
    DenseMatrix refMat6 = DenseMatrix::Random(rows, rows);
    DenseMatrix dm4 = DenseMatrix::Zero(rows, rows);
    //     DenseVector dv1 = DenseVector::Random(rows);
    SparseMatrixType m2(rows, depth);
    SparseMatrixType m2t(depth, rows);
    SparseMatrixType m3(depth, cols);
    SparseMatrixType m3t(cols, depth);
    SparseMatrixType m4(rows, cols);
    SparseMatrixType m4t(cols, rows);
    SparseMatrixType m6(rows, rows);
    initSparse(density, refMat2, m2);
    initSparse(density, refMat2t, m2t);
    initSparse(density, refMat3, m3);
    initSparse(density, refMat3t, m3t);
    initSparse(density, refMat4, m4);
    initSparse(density, refMat4t, m4t);
    initSparse(density, refMat6, m6);

    //     int c = internal::random<int>(0,depth-1);

    // sparse * sparse
    VERIFY_IS_APPROX(m4 = m2 * m3, refMat4 = refMat2 * refMat3);
    VERIFY_IS_APPROX(m4 = m2t.transpose() * m3, refMat4 = refMat2t.transpose() * refMat3);
    VERIFY_IS_APPROX(m4 = m2t.transpose() * m3t.transpose(), refMat4 = refMat2t.transpose() * refMat3t.transpose());
    VERIFY_IS_APPROX(m4 = m2 * m3t.transpose(), refMat4 = refMat2 * refMat3t.transpose());

    VERIFY_IS_APPROX(m4 = m2 * m3 / s1, refMat4 = refMat2 * refMat3 / s1);
    VERIFY_IS_APPROX(m4 = m2 * m3 * s1, refMat4 = refMat2 * refMat3 * s1);
    VERIFY_IS_APPROX(m4 = s2 * m2 * m3 * s1, refMat4 = s2 * refMat2 * refMat3 * s1);
    VERIFY_IS_APPROX(m4 = (m2 + m2) * m3, refMat4 = (refMat2 + refMat2) * refMat3);
    VERIFY_IS_APPROX(m4 = m2 * m3.leftCols(cols / 2), refMat4 = refMat2 * refMat3.leftCols(cols / 2));
    VERIFY_IS_APPROX(m4 = m2 * (m3 + m3).leftCols(cols / 2),
                     refMat4 = refMat2 * (refMat3 + refMat3).leftCols(cols / 2));

    VERIFY_IS_APPROX(m4 = (m2 * m3).pruned(0), refMat4 = refMat2 * refMat3);
    VERIFY_IS_APPROX(m4 = (m2t.transpose() * m3).pruned(0), refMat4 = refMat2t.transpose() * refMat3);
    VERIFY_IS_APPROX(m4 = (m2t.transpose() * m3t.transpose()).pruned(0),
                     refMat4 = refMat2t.transpose() * refMat3t.transpose());
    VERIFY_IS_APPROX(m4 = (m2 * m3t.transpose()).pruned(0), refMat4 = refMat2 * refMat3t.transpose());

#ifndef EIGEN_SPARSE_PRODUCT_IGNORE_TEMPORARY_COUNT
    // make sure the right product implementation is called:
    if (!SparseMatrixType::IsRowMajor) {
      VERIFY_EVALUATION_COUNT(m4 = m2 * m3, 0);  // the product is sorted as computed, and moved into m4.
      VERIFY_EVALUATION_COUNT(m4 = (m2 * m3).pruned(0), 1);
      VERIFY_EVALUATION_COUNT(m4 = (m2 * m3).eval().pruned(0), 2);
    }
#endif

    // and that pruning is effective:
    {
      DenseMatrix Ad(2, 2);
      Ad << -1, 1, 1, 1;
      SparseMatrixType As(Ad.sparseView()), B(2, 2);
      VERIFY_IS_EQUAL((As * As.transpose()).eval().nonZeros(), 4);
      VERIFY_IS_EQUAL((Ad * Ad.transpose()).eval().sparseView().eval().nonZeros(), 2);
      VERIFY_IS_EQUAL((As * As.transpose()).pruned(1e-6).eval().nonZeros(), 2);
    }

    // dense ?= sparse * sparse
    VERIFY_IS_APPROX(dm4 = m2 * m3, refMat4 = refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 += m2 * m3, refMat4 += refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 -= m2 * m3, refMat4 -= refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 = m2t.transpose() * m3, refMat4 = refMat2t.transpose() * refMat3);
    VERIFY_IS_APPROX(dm4 += m2t.transpose() * m3, refMat4 += refMat2t.transpose() * refMat3);
    VERIFY_IS_APPROX(dm4 -= m2t.transpose() * m3, refMat4 -= refMat2t.transpose() * refMat3);
    VERIFY_IS_APPROX(dm4 = m2t.transpose() * m3t.transpose(), refMat4 = refMat2t.transpose() * refMat3t.transpose());
    VERIFY_IS_APPROX(dm4 += m2t.transpose() * m3t.transpose(), refMat4 += refMat2t.transpose() * refMat3t.transpose());
    VERIFY_IS_APPROX(dm4 -= m2t.transpose() * m3t.transpose(), refMat4 -= refMat2t.transpose() * refMat3t.transpose());
    VERIFY_IS_APPROX(dm4 = m2 * m3t.transpose(), refMat4 = refMat2 * refMat3t.transpose());
    VERIFY_IS_APPROX(dm4 += m2 * m3t.transpose(), refMat4 += refMat2 * refMat3t.transpose());
    VERIFY_IS_APPROX(dm4 -= m2 * m3t.transpose(), refMat4 -= refMat2 * refMat3t.transpose());
    VERIFY_IS_APPROX(dm4 = m2 * m3 * s1, refMat4 = refMat2 * refMat3 * s1);

    // test aliasing
    m4 = m2;
    refMat4 = refMat2;
    VERIFY_IS_APPROX(m4 = m4 * m3, refMat4 = refMat4 * refMat3);

    // sparse * dense matrix
    VERIFY_IS_APPROX(dm4 = m2 * refMat3, refMat4 = refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 = m2 * refMat3t.transpose(), refMat4 = refMat2 * refMat3t.transpose());
    VERIFY_IS_APPROX(dm4 = m2t.transpose() * refMat3, refMat4 = refMat2t.transpose() * refMat3);
    VERIFY_IS_APPROX(dm4 = m2t.transpose() * refMat3t.transpose(),
                     refMat4 = refMat2t.transpose() * refMat3t.transpose());

    VERIFY_IS_APPROX(dm4 = dm4 + m2 * refMat3, refMat4 = refMat4 + refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 += m2 * refMat3, refMat4 += refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 -= m2 * refMat3, refMat4 -= refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4.noalias() += m2 * refMat3, refMat4 += refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4.noalias() -= m2 * refMat3, refMat4 -= refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 = m2 * (refMat3 + refMat3), refMat4 = refMat2 * (refMat3 + refMat3));
    VERIFY_IS_APPROX(dm4 = m2t.transpose() * (refMat3 + refMat5) * 0.5,
                     refMat4 = refMat2t.transpose() * (refMat3 + refMat5) * 0.5);

    // sparse * dense expression without DirectAccessBit (e.g. CwiseNullaryOp)
    VERIFY_IS_APPROX(dm4 = m2 * DenseMatrix::Constant(depth, cols, s1),
                     refMat4 = refMat2 * DenseMatrix::Constant(depth, cols, s1));
    VERIFY_IS_APPROX(dm4 = m2 * DenseMatrix::Zero(depth, cols), refMat4 = refMat2 * DenseMatrix::Zero(depth, cols));

    // sparse * dense vector
    VERIFY_IS_APPROX(dm4.col(0) = m2 * refMat3.col(0), refMat4.col(0) = refMat2 * refMat3.col(0));
    VERIFY_IS_APPROX(dm4.col(0) = m2 * refMat3t.transpose().col(0),
                     refMat4.col(0) = refMat2 * refMat3t.transpose().col(0));
    VERIFY_IS_APPROX(dm4.col(0) = m2t.transpose() * refMat3.col(0),
                     refMat4.col(0) = refMat2t.transpose() * refMat3.col(0));
    VERIFY_IS_APPROX(dm4.col(0) = m2t.transpose() * refMat3t.transpose().col(0),
                     refMat4.col(0) = refMat2t.transpose() * refMat3t.transpose().col(0));

    // dense * sparse
    VERIFY_IS_APPROX(dm4 = refMat2 * m3, refMat4 = refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 = dm4 + refMat2 * m3, refMat4 = refMat4 + refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 += refMat2 * m3, refMat4 += refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 -= refMat2 * m3, refMat4 -= refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4.noalias() += refMat2 * m3, refMat4 += refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4.noalias() -= refMat2 * m3, refMat4 -= refMat2 * refMat3);
    VERIFY_IS_APPROX(dm4 = refMat2 * m3t.transpose(), refMat4 = refMat2 * refMat3t.transpose());
    VERIFY_IS_APPROX(dm4 = refMat2t.transpose() * m3, refMat4 = refMat2t.transpose() * refMat3);
    VERIFY_IS_APPROX(dm4 = refMat2t.transpose() * m3t.transpose(),
                     refMat4 = refMat2t.transpose() * refMat3t.transpose());

    // sparse * dense and dense * sparse outer product
    {
      Index c = internal::random<Index>(0, depth - 1);
      Index r = internal::random<Index>(0, rows - 1);
      Index c1 = internal::random<Index>(0, cols - 1);
      Index r1 = internal::random<Index>(0, depth - 1);
      DenseMatrix dm5 = DenseMatrix::Random(depth, cols);

      VERIFY_IS_APPROX(m4 = m2.col(c) * dm5.col(c1).transpose(), refMat4 = refMat2.col(c) * dm5.col(c1).transpose());
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(m4 = m2.middleCols(c, 1) * dm5.col(c1).transpose(),
                       refMat4 = refMat2.col(c) * dm5.col(c1).transpose());
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(dm4 = m2.col(c) * dm5.col(c1).transpose(), refMat4 = refMat2.col(c) * dm5.col(c1).transpose());

      VERIFY_IS_APPROX(m4 = dm5.col(c1) * m2.col(c).transpose(), refMat4 = dm5.col(c1) * refMat2.col(c).transpose());
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(m4 = dm5.col(c1) * m2.middleCols(c, 1).transpose(),
                       refMat4 = dm5.col(c1) * refMat2.col(c).transpose());
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(dm4 = dm5.col(c1) * m2.col(c).transpose(), refMat4 = dm5.col(c1) * refMat2.col(c).transpose());

      VERIFY_IS_APPROX(m4 = dm5.row(r1).transpose() * m2.col(c).transpose(),
                       refMat4 = dm5.row(r1).transpose() * refMat2.col(c).transpose());
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(dm4 = dm5.row(r1).transpose() * m2.col(c).transpose(),
                       refMat4 = dm5.row(r1).transpose() * refMat2.col(c).transpose());

      VERIFY_IS_APPROX(m4 = m2.row(r).transpose() * dm5.col(c1).transpose(),
                       refMat4 = refMat2.row(r).transpose() * dm5.col(c1).transpose());
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(m4 = m2.middleRows(r, 1).transpose() * dm5.col(c1).transpose(),
                       refMat4 = refMat2.row(r).transpose() * dm5.col(c1).transpose());
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(dm4 = m2.row(r).transpose() * dm5.col(c1).transpose(),
                       refMat4 = refMat2.row(r).transpose() * dm5.col(c1).transpose());

      VERIFY_IS_APPROX(m4 = dm5.col(c1) * m2.row(r), refMat4 = dm5.col(c1) * refMat2.row(r));
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(m4 = dm5.col(c1) * m2.middleRows(r, 1), refMat4 = dm5.col(c1) * refMat2.row(r));
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(dm4 = dm5.col(c1) * m2.row(r), refMat4 = dm5.col(c1) * refMat2.row(r));

      VERIFY_IS_APPROX(m4 = dm5.row(r1).transpose() * m2.row(r), refMat4 = dm5.row(r1).transpose() * refMat2.row(r));
      VERIFY_IS_EQUAL(m4.nonZeros(), (refMat4.array() != 0).count());
      VERIFY_IS_APPROX(dm4 = dm5.row(r1).transpose() * m2.row(r), refMat4 = dm5.row(r1).transpose() * refMat2.row(r));
    }

    VERIFY_IS_APPROX(m6 = m6 * m6, refMat6 = refMat6 * refMat6);

    // sparse matrix * sparse vector
    ColSpVector cv0(cols), cv1;
    DenseVector dcv0(cols), dcv1;
    initSparse(2 * density, dcv0, cv0);

    RowSpVector rv0(depth), rv1;
    RowDenseVector drv0(depth), drv1(rv1);
    initSparse(2 * density, drv0, rv0);

    VERIFY_IS_APPROX(cv1 = m3 * cv0, dcv1 = refMat3 * dcv0);
    VERIFY_IS_APPROX(rv1 = rv0 * m3, drv1 = drv0 * refMat3);
    VERIFY_IS_APPROX(cv1 = m3t.adjoint() * cv0, dcv1 = refMat3t.adjoint() * dcv0);
    VERIFY_IS_APPROX(cv1 = rv0 * m3, dcv1 = drv0 * refMat3);
    VERIFY_IS_APPROX(rv1 = m3 * cv0, drv1 = refMat3 * dcv0);
  }

  // test matrix - diagonal product
  {
    DenseMatrix refM2 = DenseMatrix::Zero(rows, cols);
    DenseMatrix refM3 = DenseMatrix::Zero(rows, cols);
    DenseMatrix d3 = DenseMatrix::Zero(rows, cols);
    DiagonalMatrix<Scalar, Dynamic> d1(DenseVector::Random(cols));
    DiagonalMatrix<Scalar, Dynamic> d2(DenseVector::Random(rows));
    SparseMatrixType m2(rows, cols);
    SparseMatrixType m3(rows, cols);
    initSparse<Scalar>(density, refM2, m2);
    initSparse<Scalar>(density, refM3, m3);
    VERIFY_IS_APPROX(m3 = m2 * d1, refM3 = refM2 * d1);
    VERIFY_IS_APPROX(m3 = m2.transpose() * d2, refM3 = refM2.transpose() * d2);
    VERIFY_IS_APPROX(m3 = d2 * m2, refM3 = d2 * refM2);
    VERIFY_IS_APPROX(m3 = d1 * m2.transpose(), refM3 = d1 * refM2.transpose());

    // also check with a SparseWrapper:
    DenseVector v1 = DenseVector::Random(cols);
    DenseVector v2 = DenseVector::Random(rows);
    DenseVector v3 = DenseVector::Random(rows);
    VERIFY_IS_APPROX(m3 = m2 * v1.asDiagonal(), refM3 = refM2 * v1.asDiagonal());
    VERIFY_IS_APPROX(m3 = m2.transpose() * v2.asDiagonal(), refM3 = refM2.transpose() * v2.asDiagonal());
    VERIFY_IS_APPROX(m3 = v2.asDiagonal() * m2, refM3 = v2.asDiagonal() * refM2);
    VERIFY_IS_APPROX(m3 = v1.asDiagonal() * m2.transpose(), refM3 = v1.asDiagonal() * refM2.transpose());

    VERIFY_IS_APPROX(m3 = v2.asDiagonal() * m2 * v1.asDiagonal(), refM3 = v2.asDiagonal() * refM2 * v1.asDiagonal());

    VERIFY_IS_APPROX(v2 = m2 * v1.asDiagonal() * v1, refM2 * v1.asDiagonal() * v1);
    VERIFY_IS_APPROX(v3 = v2.asDiagonal() * m2 * v1, v2.asDiagonal() * refM2 * v1);

    // evaluate to a dense matrix to check the .row() and .col() iterator functions
    VERIFY_IS_APPROX(d3 = m2 * d1, refM3 = refM2 * d1);
    VERIFY_IS_APPROX(d3 = m2.transpose() * d2, refM3 = refM2.transpose() * d2);
    VERIFY_IS_APPROX(d3 = d2 * m2, refM3 = d2 * refM2);
    VERIFY_IS_APPROX(d3 = d1 * m2.transpose(), refM3 = d1 * refM2.transpose());
  }

  // test self-adjoint and triangular-view products
  {
    DenseMatrix b = DenseMatrix::Random(rows, rows);
    DenseMatrix x = DenseMatrix::Random(rows, rows);
    DenseMatrix refX = DenseMatrix::Random(rows, rows);
    DenseMatrix refUp = DenseMatrix::Zero(rows, rows);
    DenseMatrix refLo = DenseMatrix::Zero(rows, rows);
    DenseMatrix refS = DenseMatrix::Zero(rows, rows);
    DenseMatrix refA = DenseMatrix::Zero(rows, rows);
    SparseMatrixType mUp(rows, rows);
    SparseMatrixType mLo(rows, rows);
    SparseMatrixType mS(rows, rows);
    SparseMatrixType mA(rows, rows);
    initSparse<Scalar>(density, refA, mA);
    initSparse<Scalar>(density, refUp, mUp, ForceRealDiag | /*ForceNonZeroDiag|*/ MakeUpperTriangular);
    if (refUp.isZero()) {
      refUp(0, 0) = Scalar(1);
      mUp.coeffRef(0, 0) = Scalar(1);
    }
    refLo = refUp.adjoint();
    mLo = mUp.adjoint();
    refS = refUp + refLo;
    refS.diagonal() *= 0.5;
    mS = mUp + mLo;
    // TODO be able to address the diagonal....
    for (int k = 0; k < mS.outerSize(); ++k)
      for (typename SparseMatrixType::InnerIterator it(mS, k); it; ++it)
        if (it.index() == k) it.valueRef() *= Scalar(0.5);

    VERIFY_IS_APPROX(refS.adjoint(), refS);
    VERIFY_IS_APPROX(mS.adjoint(), mS);
    VERIFY_IS_APPROX(mS, refS);
    VERIFY_IS_APPROX(x = mS * b, refX = refS * b);

    // sparse selfadjointView with dense matrices
    VERIFY_IS_APPROX(x = mUp.template selfadjointView<Upper>() * b, refX = refS * b);
    VERIFY_IS_APPROX(x = mLo.template selfadjointView<Lower>() * b, refX = refS * b);
    VERIFY_IS_APPROX(x = mS.template selfadjointView<Upper | Lower>() * b, refX = refS * b);

    VERIFY_IS_APPROX(x = b * mUp.template selfadjointView<Upper>(), refX = b * refS);
    VERIFY_IS_APPROX(x = b * mLo.template selfadjointView<Lower>(), refX = b * refS);
    VERIFY_IS_APPROX(x = b * mS.template selfadjointView<Upper | Lower>(), refX = b * refS);

    VERIFY_IS_APPROX(x.noalias() += mUp.template selfadjointView<Upper>() * b, refX += refS * b);
    VERIFY_IS_APPROX(x.noalias() -= mLo.template selfadjointView<Lower>() * b, refX -= refS * b);
    VERIFY_IS_APPROX(x.noalias() += mS.template selfadjointView<Upper | Lower>() * b, refX += refS * b);

    DenseVector scale = DenseVector::Random(rows);
    VERIFY_IS_APPROX(x = mLo.template selfadjointView<Lower>() * scale.asDiagonal(), refX = refS * scale.asDiagonal());
    VERIFY_IS_APPROX(x = scale.asDiagonal() * mLo.template selfadjointView<Lower>(), refX = scale.asDiagonal() * refS);
    VERIFY_IS_APPROX(x = mUp.template selfadjointView<Upper>() * scale.asDiagonal(), refX = refS * scale.asDiagonal());
    VERIFY_IS_APPROX(x = scale.asDiagonal() * mUp.template selfadjointView<Upper>(), refX = scale.asDiagonal() * refS);
    VERIFY_IS_APPROX(x = mS.template selfadjointView<Upper | Lower>() * scale.asDiagonal(),
                     refX = refS * scale.asDiagonal());
    VERIFY_IS_APPROX(x = scale.asDiagonal() * mS.template selfadjointView<Upper | Lower>(),
                     refX = scale.asDiagonal() * refS);

    // sparse selfadjointView with sparse matrices
    SparseMatrixType mSres(rows, rows);
    VERIFY_IS_APPROX(mSres = mLo.template selfadjointView<Lower>() * mS,
                     refX = refLo.template selfadjointView<Lower>() * refS);
    VERIFY_IS_APPROX(mSres = mS * mLo.template selfadjointView<Lower>(),
                     refX = refS * refLo.template selfadjointView<Lower>());
    VERIFY_IS_APPROX(mSres = mLo.template selfadjointView<Lower>() * scale.asDiagonal(),
                     refX = refS * scale.asDiagonal());
    VERIFY_IS_APPROX(mSres = scale.asDiagonal() * mLo.template selfadjointView<Lower>(),
                     refX = scale.asDiagonal() * refS);
    VERIFY_IS_APPROX(mSres = mUp.template selfadjointView<Upper>() * scale.asDiagonal(),
                     refX = refS * scale.asDiagonal());
    VERIFY_IS_APPROX(mSres = scale.asDiagonal() * mUp.template selfadjointView<Upper>(),
                     refX = scale.asDiagonal() * refS);
    VERIFY_IS_APPROX(mSres = mS.template selfadjointView<Upper | Lower>() * scale.asDiagonal(),
                     refX = refS * scale.asDiagonal());
    VERIFY_IS_APPROX(mSres = scale.asDiagonal() * mS.template selfadjointView<Upper | Lower>(),
                     refX = scale.asDiagonal() * refS);

    // sparse triangularView with dense matrices
    VERIFY_IS_APPROX(x = mA.template triangularView<Upper>() * b, refX = refA.template triangularView<Upper>() * b);
    VERIFY_IS_APPROX(x = mA.template triangularView<Lower>() * b, refX = refA.template triangularView<Lower>() * b);
    VERIFY_IS_APPROX(x = b * mA.template triangularView<Upper>(), refX = b * refA.template triangularView<Upper>());
    VERIFY_IS_APPROX(x = b * mA.template triangularView<Lower>(), refX = b * refA.template triangularView<Lower>());

    // sparse triangularView with sparse matrices
    VERIFY_IS_APPROX(mSres = mA.template triangularView<Lower>() * mS,
                     refX = refA.template triangularView<Lower>() * refS);
    VERIFY_IS_APPROX(mSres = mS * mA.template triangularView<Lower>(),
                     refX = refS * refA.template triangularView<Lower>());
    VERIFY_IS_APPROX(mSres = mA.template triangularView<Upper>() * mS,
                     refX = refA.template triangularView<Upper>() * refS);
    VERIFY_IS_APPROX(mSres = mS * mA.template triangularView<Upper>(),
                     refX = refS * refA.template triangularView<Upper>());

    VERIFY_IS_APPROX(mSres = mA.template triangularView<UnitLower>() * scale.asDiagonal(),
                     refX = DenseMatrix(refA.template triangularView<UnitLower>()) * scale.asDiagonal());
    VERIFY_IS_APPROX(mSres = scale.asDiagonal() * mA.template triangularView<UnitLower>(),
                     refX = scale.asDiagonal() * DenseMatrix(refA.template triangularView<UnitLower>()));
    VERIFY_IS_APPROX(mSres = mA.template triangularView<UnitUpper>() * scale.asDiagonal(),
                     refX = DenseMatrix(refA.template triangularView<UnitUpper>()) * scale.asDiagonal());
    VERIFY_IS_APPROX(mSres = scale.asDiagonal() * mA.template triangularView<UnitUpper>(),
                     refX = scale.asDiagonal() * DenseMatrix(refA.template triangularView<UnitUpper>()));
  }
}

// New test for Bug in SparseTimeDenseProduct
template <typename SparseMatrixType, typename DenseMatrixType>
void sparse_product_regression_test() {
  // This code does not compile with afflicted versions of the bug
  SparseMatrixType sm1(3, 2);
  DenseMatrixType m2(2, 2);
  sm1.setZero();
  m2.setZero();

  DenseMatrixType m3 = sm1 * m2;

  // This code produces a segfault with afflicted versions of another SparseTimeDenseProduct
  // bug

  SparseMatrixType sm2(20000, 2);
  sm2.setZero();
  DenseMatrixType m4(sm2 * m2);

  VERIFY_IS_APPROX(m4(0, 0), 0.0);
}

template <typename Scalar>
void bug_942() {
  using Vector = Matrix<Scalar, Dynamic, 1>;
  using ColSpMat = SparseMatrix<Scalar, ColMajor>;
  using RowSpMat = SparseMatrix<Scalar, RowMajor>;
  ColSpMat cmA(1, 1);
  cmA.insert(0, 0) = 1;

  RowSpMat rmA(1, 1);
  rmA.insert(0, 0) = 1;

  Vector d(1);
  d[0] = 2;

  double res = 2;

  VERIFY_IS_APPROX((cmA * d.asDiagonal()).eval().coeff(0, 0), res);
  VERIFY_IS_APPROX((d.asDiagonal() * rmA).eval().coeff(0, 0), res);
  VERIFY_IS_APPROX((rmA * d.asDiagonal()).eval().coeff(0, 0), res);
  VERIFY_IS_APPROX((d.asDiagonal() * cmA).eval().coeff(0, 0), res);
}

template <typename Real>
void test_mixing_types() {
  using Cplx = std::complex<Real>;
  using SpMatReal = SparseMatrix<Real>;
  using SpMatCplx = SparseMatrix<Cplx>;
  using SpRowMatCplx = SparseMatrix<Cplx, RowMajor>;
  using DenseMatReal = Matrix<Real, Dynamic, Dynamic>;
  using DenseMatCplx = Matrix<Cplx, Dynamic, Dynamic>;

  Index n = internal::random<Index>(1, 100);
  double density = (std::max)(8. / static_cast<double>(n * n), 0.2);

  SpMatReal sR1(n, n);
  SpMatCplx sC1(n, n), sC2(n, n), sC3(n, n);
  SpRowMatCplx sCR(n, n);
  DenseMatReal dR1(n, n);
  DenseMatCplx dC1(n, n), dC2(n, n), dC3(n, n);

  initSparse<Real>(density, dR1, sR1);
  initSparse<Cplx>(density, dC1, sC1);
  initSparse<Cplx>(density, dC2, sC2);

  VERIFY_IS_APPROX(sC2 = (sR1 * sC1), dC3 = dR1.template cast<Cplx>() * dC1);
  VERIFY_IS_APPROX(sC2 = (sC1 * sR1), dC3 = dC1 * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(sC2 = (sR1.transpose() * sC1), dC3 = dR1.template cast<Cplx>().transpose() * dC1);
  VERIFY_IS_APPROX(sC2 = (sC1.transpose() * sR1), dC3 = dC1.transpose() * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(sC2 = (sR1 * sC1.transpose()), dC3 = dR1.template cast<Cplx>() * dC1.transpose());
  VERIFY_IS_APPROX(sC2 = (sC1 * sR1.transpose()), dC3 = dC1 * dR1.template cast<Cplx>().transpose());
  VERIFY_IS_APPROX(sC2 = (sR1.transpose() * sC1.transpose()),
                   dC3 = dR1.template cast<Cplx>().transpose() * dC1.transpose());
  VERIFY_IS_APPROX(sC2 = (sC1.transpose() * sR1.transpose()),
                   dC3 = dC1.transpose() * dR1.template cast<Cplx>().transpose());

  VERIFY_IS_APPROX(sCR = (sR1 * sC1), dC3 = dR1.template cast<Cplx>() * dC1);
  VERIFY_IS_APPROX(sCR = (sC1 * sR1), dC3 = dC1 * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(sCR = (sR1.transpose() * sC1), dC3 = dR1.template cast<Cplx>().transpose() * dC1);
  VERIFY_IS_APPROX(sCR = (sC1.transpose() * sR1), dC3 = dC1.transpose() * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(sCR = (sR1 * sC1.transpose()), dC3 = dR1.template cast<Cplx>() * dC1.transpose());
  VERIFY_IS_APPROX(sCR = (sC1 * sR1.transpose()), dC3 = dC1 * dR1.template cast<Cplx>().transpose());
  VERIFY_IS_APPROX(sCR = (sR1.transpose() * sC1.transpose()),
                   dC3 = dR1.template cast<Cplx>().transpose() * dC1.transpose());
  VERIFY_IS_APPROX(sCR = (sC1.transpose() * sR1.transpose()),
                   dC3 = dC1.transpose() * dR1.template cast<Cplx>().transpose());

  VERIFY_IS_APPROX(sC2 = (sR1 * sC1).pruned(), dC3 = dR1.template cast<Cplx>() * dC1);
  VERIFY_IS_APPROX(sC2 = (sC1 * sR1).pruned(), dC3 = dC1 * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(sC2 = (sR1.transpose() * sC1).pruned(), dC3 = dR1.template cast<Cplx>().transpose() * dC1);
  VERIFY_IS_APPROX(sC2 = (sC1.transpose() * sR1).pruned(), dC3 = dC1.transpose() * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(sC2 = (sR1 * sC1.transpose()).pruned(), dC3 = dR1.template cast<Cplx>() * dC1.transpose());
  VERIFY_IS_APPROX(sC2 = (sC1 * sR1.transpose()).pruned(), dC3 = dC1 * dR1.template cast<Cplx>().transpose());
  VERIFY_IS_APPROX(sC2 = (sR1.transpose() * sC1.transpose()).pruned(),
                   dC3 = dR1.template cast<Cplx>().transpose() * dC1.transpose());
  VERIFY_IS_APPROX(sC2 = (sC1.transpose() * sR1.transpose()).pruned(),
                   dC3 = dC1.transpose() * dR1.template cast<Cplx>().transpose());

  VERIFY_IS_APPROX(sCR = (sR1 * sC1).pruned(), dC3 = dR1.template cast<Cplx>() * dC1);
  VERIFY_IS_APPROX(sCR = (sC1 * sR1).pruned(), dC3 = dC1 * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(sCR = (sR1.transpose() * sC1).pruned(), dC3 = dR1.template cast<Cplx>().transpose() * dC1);
  VERIFY_IS_APPROX(sCR = (sC1.transpose() * sR1).pruned(), dC3 = dC1.transpose() * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(sCR = (sR1 * sC1.transpose()).pruned(), dC3 = dR1.template cast<Cplx>() * dC1.transpose());
  VERIFY_IS_APPROX(sCR = (sC1 * sR1.transpose()).pruned(), dC3 = dC1 * dR1.template cast<Cplx>().transpose());
  VERIFY_IS_APPROX(sCR = (sR1.transpose() * sC1.transpose()).pruned(),
                   dC3 = dR1.template cast<Cplx>().transpose() * dC1.transpose());
  VERIFY_IS_APPROX(sCR = (sC1.transpose() * sR1.transpose()).pruned(),
                   dC3 = dC1.transpose() * dR1.template cast<Cplx>().transpose());

  VERIFY_IS_APPROX(dC2 = (sR1 * sC1), dC3 = dR1.template cast<Cplx>() * dC1);
  VERIFY_IS_APPROX(dC2 = (sC1 * sR1), dC3 = dC1 * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(dC2 = (sR1.transpose() * sC1), dC3 = dR1.template cast<Cplx>().transpose() * dC1);
  VERIFY_IS_APPROX(dC2 = (sC1.transpose() * sR1), dC3 = dC1.transpose() * dR1.template cast<Cplx>());
  VERIFY_IS_APPROX(dC2 = (sR1 * sC1.transpose()), dC3 = dR1.template cast<Cplx>() * dC1.transpose());
  VERIFY_IS_APPROX(dC2 = (sC1 * sR1.transpose()), dC3 = dC1 * dR1.template cast<Cplx>().transpose());
  VERIFY_IS_APPROX(dC2 = (sR1.transpose() * sC1.transpose()),
                   dC3 = dR1.template cast<Cplx>().transpose() * dC1.transpose());
  VERIFY_IS_APPROX(dC2 = (sC1.transpose() * sR1.transpose()),
                   dC3 = dC1.transpose() * dR1.template cast<Cplx>().transpose());

  VERIFY_IS_APPROX(dC2 = dR1 * sC1, dC3 = dR1.template cast<Cplx>() * sC1);
  VERIFY_IS_APPROX(dC2 = sR1 * dC1, dC3 = sR1.template cast<Cplx>() * dC1);
  VERIFY_IS_APPROX(dC2 = dC1 * sR1, dC3 = dC1 * sR1.template cast<Cplx>());
  VERIFY_IS_APPROX(dC2 = sC1 * dR1, dC3 = sC1 * dR1.template cast<Cplx>());

  VERIFY_IS_APPROX(dC2 = dR1.row(0) * sC1, dC3 = dR1.template cast<Cplx>().row(0) * sC1);
  VERIFY_IS_APPROX(dC2 = sR1 * dC1.col(0), dC3 = sR1.template cast<Cplx>() * dC1.col(0));
  VERIFY_IS_APPROX(dC2 = dC1.row(0) * sR1, dC3 = dC1.row(0) * sR1.template cast<Cplx>());
  VERIFY_IS_APPROX(dC2 = sC1 * dR1.col(0), dC3 = sC1 * dR1.template cast<Cplx>().col(0));
}

// Test mixed storage types
template <int OrderA, int OrderB, int OrderC>
void test_mixed_storage_imp() {
  using Real = float;
  using DenseMat = Matrix<Real, Dynamic, Dynamic>;

  // Case: Large inputs but small result
  {
    SparseMatrix<Real, OrderA> A(8, 512);
    SparseMatrix<Real, OrderB> B(512, 8);
    DenseMat refA(8, 512);
    DenseMat refB(512, 8);

    initSparse<Real>(0.1, refA, A);
    initSparse<Real>(0.1, refB, B);

    SparseMatrix<Real, OrderC, std::int8_t> result;
    SparseMatrix<Real, OrderC> result_large;
    DenseMat refResult;

    VERIFY_IS_APPROX(result = (A * B), refResult = refA * refB);
  }

  // Case: Small input but large result
  {
    SparseMatrix<Real, OrderA, std::int8_t> A(127, 8);
    SparseMatrix<Real, OrderB, std::int8_t> B(8, 127);
    DenseMat refA(127, 8);
    DenseMat refB(8, 127);

    initSparse<Real>(0.01, refA, A);
    initSparse<Real>(0.01, refB, B);

    SparseMatrix<Real, OrderC> result;
    SparseMatrix<Real, OrderC> result_large;
    DenseMat refResult;

    VERIFY_IS_APPROX(result = (A * B), refResult = refA * refB);
  }
}

template <typename = void>
void test_mixed_storage() {
  test_mixed_storage_imp<RowMajor, RowMajor, RowMajor>();
  test_mixed_storage_imp<RowMajor, RowMajor, ColMajor>();
  test_mixed_storage_imp<RowMajor, ColMajor, RowMajor>();
  test_mixed_storage_imp<RowMajor, ColMajor, ColMajor>();
  test_mixed_storage_imp<ColMajor, RowMajor, RowMajor>();
  test_mixed_storage_imp<ColMajor, RowMajor, ColMajor>();
  test_mixed_storage_imp<ColMajor, ColMajor, RowMajor>();
  test_mixed_storage_imp<ColMajor, ColMajor, ColMajor>();
}

// The pruned sparse*sparse product must construct, move, and destroy the coefficients properly for
// non-trivially-copyable scalars (issue #2873).
template <typename = void>
void test_pruned_product_custom_scalar() {
  using SpMat = SparseMatrix<AnnoyingScalar>;
  using DenseMat = Matrix<AnnoyingScalar, Dynamic, Dynamic>;
  int instances_before = AnnoyingScalar::instances;
  {
    for (double density : {0.02, 0.3}) {
      const Index n = 50;
      SpMat A(n, n), B(n, n);
      for (Index k = 0; k < Index(density * n * n); ++k) {
        A.coeffRef(internal::random<Index>(0, n - 1), internal::random<Index>(0, n - 1)) =
            AnnoyingScalar(internal::random<float>());
        B.coeffRef(internal::random<Index>(0, n - 1), internal::random<Index>(0, n - 1)) =
            AnnoyingScalar(internal::random<float>());
      }
      SpMat C = (A * B).pruned();
      DenseMat refC = DenseMat(A).lazyProduct(DenseMat(B));
      VERIFY_IS_APPROX(DenseMat(C), refC);
    }
    // A single long column in a large matrix.
    {
      const Index n = 1200;
      SpMat A(n, n), B(n, n);
      for (Index i = 0; i < 900; ++i) A.coeffRef(i, 0) = AnnoyingScalar(float(i % 7) + 1.0f);
      B.coeffRef(0, 0) = AnnoyingScalar(2.0f);
      SpMat C = (A * B).pruned();
      VERIFY_IS_EQUAL(C.nonZeros(), 900);
      for (Index i = 0; i < 900; i += 123) {
        VERIFY_IS_APPROX(C.coeff(i, 0), AnnoyingScalar(2.0f * (float(i % 7) + 1.0f)));
      }
    }
  }
  // Every constructed AnnoyingScalar must have been destroyed again.
  VERIFY_IS_EQUAL(AnnoyingScalar::instances, instances_before);
}

// A long vector whose lowest index arrives last.
template <typename = void>
void test_pruned_product_front_insertion() {
  const int n = 1000;
  SparseMatrix<double> A(n, 2), B(2, 100);
  for (int i = 1; i <= n / 2; ++i) A.insert(i, 0) = 1.0;
  A.insert(0, 1) = 2.0;
  B.insert(0, 0) = 1.0;
  B.insert(1, 0) = 1.0;
  SparseMatrix<double> C = (A * B).pruned();
  VERIFY_IS_EQUAL(C.nonZeros(), n / 2 + 1);
  VERIFY_IS_APPROX(MatrixXd(C.toDense()), MatrixXd(A.toDense() * B.toDense()));
}

// Result vectors whose products arrive out of order and all cancel, between vectors that keep entries: an emptied
// vector must leave nothing behind for the next one.
template <typename = void>
void test_pruned_product_cancelled_vector() {
  SparseMatrix<double> A(12, 4), B(4, 5);
  A.insert(5, 0) = 1.0;
  A.insert(9, 0) = 1.0;
  A.insert(2, 1) = 1.0;
  A.insert(2, 2) = -1.0;
  A.insert(5, 2) = -1.0;
  A.insert(9, 2) = -1.0;
  A.insert(9, 3) = 3.0;
  A.insert(5, 3) = 2.0;
  for (Index j : {0, 2, 3})
    for (Index k = 0; k < 3; ++k) B.insert(k, j) = 1.0;
  B.insert(0, 1) = 1.0;
  B.insert(3, 1) = 1.0;
  B.insert(3, 4) = 1.0;
  B.insert(1, 4) = 1.0;
  SparseMatrix<double> C = (A * B).pruned();
  VERIFY_IS_EQUAL(C.innerIndicesAreSorted(), C.outerSize());
  VERIFY_IS_EQUAL(C.nonZeros(), Index(5));
  VERIFY_IS_EQUAL((MatrixXd(C) - MatrixXd(A) * MatrixXd(B)).norm(), 0.0);
}

// The product summed entry by entry, so that a repeated inner index of an operand vector counts every time.
template <typename Dense, typename Lhs, typename Rhs>
Dense sparse_product_reference(const Lhs& lhs, const Rhs& rhs) {
  Dense l = Dense::Zero(lhs.rows(), lhs.cols()), r = Dense::Zero(rhs.rows(), rhs.cols());
  for (Index k = 0; k < lhs.outerSize(); ++k)
    for (typename Lhs::InnerIterator it(lhs, k); it; ++it) l(it.row(), it.col()) += it.value();
  for (Index k = 0; k < rhs.outerSize(); ++k)
    for (typename Rhs::InnerIterator it(rhs, k); it; ++it) r(it.row(), it.col()) += it.value();
  return l * r;
}

// Result vectors of two to four terms with at most 8 entries in all are sorted and summed by the network of
// bucket_sorter, which takes terms in any order and with repeated indices; longer ones take the other paths. Terms
// interleave and share indices; one vector is unsorted and one repeats an index. Integer values make sums exact.
template <typename Scalar, int Order>
void test_sparse_product_small_vectors() {
  using SpMat = SparseMatrix<Scalar, Order>;
  using ColMat = SparseMatrix<Scalar, ColMajor>;
  using Dense = Matrix<Scalar, Dynamic, Dynamic>;
  const Index rows = 60, mid = 12, cols = 50;
  // Vector k has k % 4 + 1 entries, every (k % 3 + 2)-th row from k; vector mid - 1 is empty.
  ColMat a(rows, mid), b(mid, cols);
  a.reserve(VectorXi::Constant(mid, 4));
  for (Index k = 0; k + 1 < mid; ++k)
    for (Index q = 0; q <= k % 4; ++q) a.insert(k + q * (k % 3 + 2), k) = Scalar(double(1 + (q + k) % 4));
  a.makeCompressed();
  // Vector 3 is reversed and vector 7 repeats its first index.
  std::reverse(a.innerIndexPtr() + a.outerIndexPtr()[3], a.innerIndexPtr() + a.outerIndexPtr()[4]);
  a.innerIndexPtr()[a.outerIndexPtr()[7] + 1] = a.innerIndexPtr()[a.outerIndexPtr()[7]];
  // Column j of b sums j % 4 + 1 vectors of a from (5 j) % mid, so sums have 1 to 16 entries.
  for (Index j = 0; j < cols; ++j)
    for (Index t = 0; t <= j % 4; ++t) b.insert((5 * j + 3 * t) % mid, j) = Scalar(double(t % 2 ? -1 : 2));
  b.makeCompressed();
  // Row-major, the terms of a result row are rows of the right operand: multiply the transposes, b^T a^T.
  const bool rowMajor = Order == RowMajor;
  Dense ref = sparse_product_reference<Dense>(a, b);
  if (rowMajor) ref = Dense(ref.transpose());
  for (bool compressed : {true, false}) {
    // a, or a^T row-major, from the arrays of a: a conversion would reject the reversed and the repeated indices.
    SpMat terms(rowMajor ? mid : rows, rowMajor ? rows : mid);
    terms.resizeNonZeros(a.nonZeros());
    std::copy(a.outerIndexPtr(), a.outerIndexPtr() + mid + 1, terms.outerIndexPtr());
    std::copy(a.innerIndexPtr(), a.innerIndexPtr() + a.nonZeros(), terms.innerIndexPtr());
    std::copy(a.valuePtr(), a.valuePtr() + a.nonZeros(), terms.valuePtr());
    SpMat weights = rowMajor ? SpMat(b.transpose()) : SpMat(b);
    if (!compressed) terms.uncompress();
    SpMat c = rowMajor ? SpMat(weights * terms) : SpMat(terms * weights);
    VERIFY_IS_EQUAL(c.innerIndicesAreSorted(), c.outerSize());
    VERIFY_IS_EQUAL(Dense(c), ref);
    SpMat p = rowMajor ? SpMat((weights * terms).pruned()) : SpMat((terms * weights).pruned());
    VERIFY_IS_EQUAL(p.innerIndicesAreSorted(), p.outerSize());
    VERIFY_IS_EQUAL(Dense(p), ref);
    VERIFY(p.nonZeros() <= c.nonZeros());
  }
}

// Indices of 2^61 or more do not fit the keys of bucket_sorter: a small sum of them must take another path.
template <typename = void>
void test_sparse_product_small_vectors_huge_indices() {
  using StorageIndex = long long;
  using SpMat = SparseMatrix<double, ColMajor, StorageIndex>;
  if (sizeof(Index) < 8) return;
  const Index rows = Index(1) << 62;
  const StorageIndex top = StorageIndex(rows - 1), half = StorageIndex(1) << 61;
  SpMat a(rows, 2), b(2, 1);
  a.insert(half - 1, 0) = 1.0;
  a.insert(top - 2, 0) = 2.0;
  a.insert(half, 1) = 3.0;
  a.insert(top - 2, 1) = 4.0;
  a.insert(top, 1) = 5.0;
  b.insert(0, 0) = 2.0;
  b.insert(1, 0) = -1.0;
  SpMat c = a * b;
  VERIFY_IS_EQUAL(c.innerIndicesAreSorted(), c.outerSize());
  VERIFY_IS_EQUAL(c.nonZeros(), Index(4));
  const StorageIndex expectedIndices[] = {half - 1, half, top - 2, top};
  const double expectedValues[] = {2.0, -3.0, 0.0, -5.0};
  for (Index p = 0; p < 4; ++p) {
    VERIFY_IS_EQUAL(c.innerIndexPtr()[p], expectedIndices[p]);
    VERIFY_IS_EQUAL(c.valuePtr()[p], expectedValues[p]);
  }
}

// Result vectors of one or two operand vectors are merged, which needs sorted operand vectors without a repeated
// index. An operand that breaks this must take the general path and give the same product.
template <typename = void>
void test_sparse_product_few_terms_unsorted() {
  const Index rows = 200, mid = 12, cols = 30;
  SparseMatrix<double> A(rows, mid), B(mid, cols);
  for (Index k = 0; k < mid; ++k)
    for (Index q = 0; q < 20; ++q) A.coeffRef((7 * q + 13 * k) % rows, k) = double(1 + (q + k) % 5);
  A.makeCompressed();
  for (Index j = 0; j < cols; ++j) {
    B.coeffRef(j % mid, j) = 2.0;
    if (j % 3) B.coeffRef((5 * j + 1) % mid, j) = -1.0;
  }
  B.makeCompressed();
  MatrixXd ref = MatrixXd(A) * MatrixXd(B);
  for (int variant = 0; variant < 3; ++variant) {
    SparseMatrix<double> lhs = A;
    // 1: every vector reversed; 2: one index out of place in every other vector.
    for (Index k = 0; k < mid; ++k) {
      Index begin = lhs.outerIndexPtr()[k], end = lhs.outerIndexPtr()[k + 1];
      if (variant == 1) {
        std::reverse(lhs.innerIndexPtr() + begin, lhs.innerIndexPtr() + end);
        std::reverse(lhs.valuePtr() + begin, lhs.valuePtr() + end);
      } else if (variant == 2 && k % 2 == 0) {
        std::swap(lhs.innerIndexPtr()[begin + 3], lhs.innerIndexPtr()[end - 1]);
        std::swap(lhs.valuePtr()[begin + 3], lhs.valuePtr()[end - 1]);
      }
    }
    SparseMatrix<double> C = lhs * B;
    VERIFY_IS_EQUAL(C.innerIndicesAreSorted(), C.outerSize());
    VERIFY_IS_EQUAL((MatrixXd(C) - ref).norm(), 0.0);
    SparseMatrix<double> P = (lhs * B).pruned();
    VERIFY_IS_EQUAL(P.innerIndicesAreSorted(), P.outerSize());
    VERIFY_IS_EQUAL((MatrixXd(P) - ref).norm(), 0.0);
    VERIFY(P.nonZeros() <= C.nonZeros());
  }
}

// An inner dimension of 2^62: indices there do not fit the packed keys of the bucket sort, which must then sort
// by insertion. Three terms per vector with distinct indices keep the vectors on the appended path, and nothing of
// the inner dimension is allocated.
template <typename = void>
void test_sparse_product_huge_inner_dimension() {
  using StorageIndex = long long;
  using SpMat = SparseMatrix<double, ColMajor, StorageIndex>;
  if (sizeof(Index) < 8) return;
  const Index rows = Index(1) << 62, perVector = 80, mid = 6;
  SpMat a(rows, mid), b(mid, 2);
  a.reserve(VectorXi::Constant(mid, int(perVector)));
  std::map<StorageIndex, double> expected[2];
  for (Index k = 0; k < mid; ++k)
    for (Index q = 0; q < perVector; ++q) {
      // Spread over the whole range, shifted per vector except for every seventh index, which the vectors share.
      StorageIndex i = StorageIndex(rows - 1 - ((perVector - 1 - q) * (rows / 100) + (q % 7 == 0 ? 0 : k * 1000 + q)));
      a.insertBackUncompressed(i, k) = double(1 + (q + k) % 5);
    }
  for (Index j = 0; j < 2; ++j)
    for (Index t = 0; t < 3; ++t) b.insert(3 * j + t, j) = double(t + 2);
  for (Index j = 0; j < 2; ++j)
    for (SpMat::InnerIterator bi(b, j); bi; ++bi)
      for (SpMat::InnerIterator ai(a, bi.index()); ai; ++ai) expected[j][ai.index()] += ai.value() * bi.value();
  SpMat c = a * b;
  VERIFY_IS_EQUAL(c.innerIndicesAreSorted(), c.outerSize());
  VERIFY_IS_EQUAL(c.nonZeros(), Index(expected[0].size() + expected[1].size()));
  VERIFY(Index(expected[0].size()) < 3 * perVector);
  for (Index j = 0; j < 2; ++j)
    for (SpMat::InnerIterator it(c, j); it; ++it) {
      auto found = expected[j].find(it.index());
      VERIFY(found != expected[j].end() && found->second == it.value());
    }
  VERIFY(c.innerIndexPtr()[c.nonZeros() - 1] > rows / 2);
}

template <typename = void>
void test_pruned_product_row_vector() {
  SparseMatrix<double> A(4, 3);
  A.insert(0, 1) = 1;
  A.insert(3, 0) = 3;
  SparseVector<double, RowMajor> r(4);
  r.insert(0) = 2;
  r.insert(3) = 5;
  SparseVector<double, RowMajor> c = (r * A).pruned();
  VERIFY_IS_APPROX(MatrixXd(c.toDense()), MatrixXd(r.toDense() * A.toDense()));

  // A row-major product whose nested lhs evaluates column-major.
  SparseMatrix<double> P(4, 4), Q(4, 4);
  SparseMatrix<double, RowMajor> S(4, 4);
  for (int i = 0; i < 4; ++i) {
    P.insert(i, i) = i + 1;
    Q.insert(i, (i + 1) % 4) = 2;
    S.insert(i, (i + 2) % 4) = 3;
  }
  SparseMatrix<double, RowMajor> R = (P * Q * S).pruned();
  VERIFY_IS_APPROX(MatrixXd(R.toDense()), MatrixXd(P.toDense() * Q.toDense() * S.toDense()));
}

template <typename = void>
void test_sparse_product_mul_assign() {
  SparseMatrix<double> A(3, 3), B(3, 3);
  A.insert(0, 1) = 2;
  A.insert(2, 2) = 3;
  B.insert(1, 0) = 5;
  B.insert(2, 1) = 7;
  const MatrixXd ref = A.toDense() * B.toDense();
  A *= B;
  VERIFY_IS_APPROX(MatrixXd(A.toDense()), ref);
}

// A Block with no rows or no columns has data() == nullptr but a non-zero outerStride(); the product kernels must not
// offset that pointer. The failure is undefined behavior that only a sanitizer reports (-fsanitize=undefined).
template <int SparseOrder, int DenseOrder>
void test_sparse_dense_product_empty_blocks() {
  using DenseMatrix = Matrix<double, Dynamic, Dynamic, DenseOrder>;
  DenseMatrix big = DenseMatrix::Random(6, 6);
  DenseMatrix dstStorage = DenseMatrix::Constant(6, 6, 7.0);
  for (Index m : {0, 3})
    for (Index k : {0, 3})
      for (Index c : {0, 2}) {
        SparseMatrix<double, SparseOrder> A(m, k);
        for (Index i = 0; k > 0 && i < m; ++i) A.insert(i, i % k) = double(i + 1);
        const DenseMatrix refA = DenseMatrix(A);
        const DenseMatrix B = DenseMatrix::Random(k, c);
        big.block(1, 1, k, c) = B;
        const auto rhs = big.block(1, 1, k, c);
        const DenseMatrix ref = refA * B;

        const DenseMatrix Y = A * rhs;
        VERIFY_IS_EQUAL(Y.rows(), m);
        VERIFY_IS_EQUAL(Y.cols(), c);
        VERIFY_IS_APPROX(Y, ref);
        const DenseMatrix Z = rhs.transpose() * A.transpose();
        VERIFY_IS_EQUAL(Z.rows(), c);
        VERIFY_IS_EQUAL(Z.cols(), m);
        VERIFY_IS_APPROX(Z, ref.transpose());

        auto dst = dstStorage.block(1, 1, m, c);
        dst.noalias() = A * rhs;
        VERIFY_IS_APPROX(DenseMatrix(dst), ref);
        dst.noalias() += A * rhs;
        VERIFY_IS_APPROX(DenseMatrix(dst), DenseMatrix(2 * ref));
      }
}

template <int Order>
void test_sparse_cwise_product_operand_types() {
  using Left = noncommutative_scalar::TaggedScalar<0>;
  using Right = noncommutative_scalar::TaggedScalar<1>;
  using Result = noncommutative_scalar::TaggedScalar<2>;
  SparseMatrix<Left, Order> lhs(3, 3);
  lhs.reserve(VectorXi::Constant(3, 3));
  lhs.insert(0, 1) = Left(2);
  lhs.insert(2, 0) = Left(-3);
  Matrix<Right, Dynamic, Dynamic, Order> rhs(3, 3);
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i) rhs(i, j) = Right(int(i + 2 * j + 1));

  for (int compressed = 0; compressed < 2; ++compressed) {
    SparseMatrix<Result, Order> result = lhs.cwiseProduct(rhs);
    VERIFY_IS_EQUAL(result.nonZeros(), 2);
    for (Index j = 0; j < 3; ++j)
      for (Index i = 0; i < 3; ++i) VERIFY_IS_EQUAL(result.coeff(i, j).value, lhs.coeff(i, j).value * rhs(i, j).value);
    lhs.makeCompressed();
  }

  Matrix<Left, Dynamic, Dynamic, Order> denseLeft(3, 3);
  SparseMatrix<Right, Order> sparseRight(3, 3);
  sparseRight.reserve(VectorXi::Constant(3, 3));
  sparseRight.insert(0, 1) = Right(4);
  sparseRight.insert(2, 0) = Right(5);
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i) denseLeft(i, j) = Left(int(i + 2 * j + 1));
  for (int compressed = 0; compressed < 2; ++compressed) {
    const SparseMatrix<Result, Order> result = denseLeft.cwiseProduct(sparseRight);
    VERIFY_IS_EQUAL(result.nonZeros(), 2);
    for (Index j = 0; j < 3; ++j)
      for (Index i = 0; i < 3; ++i)
        VERIFY_IS_EQUAL(result.coeff(i, j).value, denseLeft(i, j).value * sparseRight.coeff(i, j).value);
    sparseRight.makeCompressed();
  }
}

template <int Order>
void test_sparse_cwise_sum_operand_types() {
  using Left = noncommutative_scalar::TaggedScalar<0>;
  using Right = noncommutative_scalar::TaggedScalar<1>;
  using Result = noncommutative_scalar::TaggedScalar<2>;
  SparseMatrix<Left, Order> lhs(3, 3);
  SparseMatrix<Right, Order> rhs(3, 3);
  lhs.reserve(VectorXi::Constant(3, 3));
  rhs.reserve(VectorXi::Constant(3, 3));
  lhs.insert(0, 1) = Left(2);
  lhs.insert(2, 0) = Left(-3);
  rhs.insert(0, 1) = Right(4);
  rhs.insert(1, 2) = Right(5);
  Matrix<Left, Dynamic, Dynamic, Order> denseLhs(3, 3);
  Matrix<Right, Dynamic, Dynamic, Order> denseRhs(3, 3);
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i) {
      denseLhs(i, j) = Left(int(i + j + 1));
      denseRhs(i, j) = Right(int(i - j - 1));
    }
  for (int compressed = 0; compressed < 2; ++compressed) {
    const SparseMatrix<Result, Order> sparse = lhs + rhs;
    const Matrix<Result, Dynamic, Dynamic, Order> denseSparse = denseLhs + rhs;
    const Matrix<Result, Dynamic, Dynamic, Order> sparseDense = lhs + denseRhs;
    for (Index j = 0; j < 3; ++j)
      for (Index i = 0; i < 3; ++i) {
        VERIFY_IS_EQUAL(sparse.coeff(i, j).value, lhs.coeff(i, j).value + rhs.coeff(i, j).value);
        VERIFY_IS_EQUAL(denseSparse(i, j).value, denseLhs(i, j).value + rhs.coeff(i, j).value);
        VERIFY_IS_EQUAL(sparseDense(i, j).value, lhs.coeff(i, j).value + denseRhs(i, j).value);
      }
    lhs.makeCompressed();
    rhs.makeCompressed();
  }

  Matrix<double, 2, 2, Order> real = Matrix<double, 2, 2, Order>::Ones();
  SparseMatrix<std::complex<double>, Order> complex(2, 2);
  complex.insert(0, 1) = std::complex<double>(2, 3);
  const Matrix<std::complex<double>, 2, 2, Order> result = real + complex;
  for (Index j = 0; j < 2; ++j)
    for (Index i = 0; i < 2; ++i) VERIFY_IS_EQUAL(result(i, j), std::complex<double>(real(i, j)) + complex.coeff(i, j));
}

template <int Order>
void test_sparse_diagonal_products() {
  using DiagonalScalar = noncommutative_scalar::TaggedScalar<0>;
  using MatrixScalar = noncommutative_scalar::TaggedScalar<1>;
  using Scalar = noncommutative_scalar::NoncommutingBlock;
  DiagonalMatrix<DiagonalScalar, 3> diagonal;
  DiagonalMatrix<Scalar, 3> blockDiagonal;
  for (Index i = 0; i < 3; ++i) {
    diagonal.diagonal()(i) = DiagonalScalar(int(i) + 2);
    blockDiagonal.diagonal()(i).values = {{1, 0, int(i) + 2, 1}};
  }
  SparseMatrix<MatrixScalar, Order> matrix(3, 3);
  SparseMatrix<Scalar, Order> blocks(3, 3);
  matrix.reserve(VectorXi::Constant(3, 3));
  blocks.reserve(VectorXi::Constant(3, 3));
  matrix.insert(0, 1) = MatrixScalar(5);
  matrix.insert(2, 0) = MatrixScalar(-2);
  blocks.insert(0, 1).values = {{1, 2, 0, 1}};
  blocks.insert(2, 0).values = {{2, 1, 0, 1}};
  for (int compressed = 0; compressed < 2; ++compressed) {
    const SparseMatrix<noncommutative_scalar::TaggedScalar<2>, Order> left = diagonal * matrix;
    const SparseMatrix<noncommutative_scalar::TaggedScalar<3>, Order> right = matrix * diagonal;
    const SparseMatrix<Scalar, Order> blockLeft = blockDiagonal * blocks;
    const SparseMatrix<Scalar, Order> blockRight = blocks * blockDiagonal;
    VERIFY_IS_EQUAL(left.nonZeros(), matrix.nonZeros());
    VERIFY_IS_EQUAL(right.nonZeros(), matrix.nonZeros());
    for (Index j = 0; j < 3; ++j)
      for (Index i = 0; i < 3; ++i) {
        VERIFY_IS_EQUAL(left.coeff(i, j).value, diagonal.diagonal()(i).value * matrix.coeff(i, j).value);
        VERIFY_IS_EQUAL(right.coeff(i, j).value, matrix.coeff(i, j).value * diagonal.diagonal()(j).value);
        const Scalar expectedLeft = blockDiagonal.diagonal()(i) * blocks.coeff(i, j);
        const Scalar expectedRight = blocks.coeff(i, j) * blockDiagonal.diagonal()(j);
        VERIFY(blockLeft.coeff(i, j).values == expectedLeft.values);
        VERIFY(blockRight.coeff(i, j).values == expectedRight.values);
      }
    matrix.makeCompressed();
    blocks.makeCompressed();
  }
}

// Pruned selects the pruned product and drops the entries whose product row or column is empty.
template <bool Pruned, int LOrder, int ROrder, int DstOrder>
void test_sparse_product_operand_order() {
  using Scalar = noncommutative_scalar::NoncommutingBlock;
  SparseMatrix<Scalar, LOrder> lhs(2, 3);
  SparseMatrix<Scalar, ROrder> rhs(3, 2);
  lhs.reserve(VectorXi::Constant(lhs.outerSize(), 3));
  rhs.reserve(VectorXi::Constant(rhs.outerSize(), 3));
  lhs.insert(0, 0).values = {{1, 2, 0, 1}};
  lhs.insert(0, 2).values = {{2, 1, 0, 1}};
  rhs.insert(0, 1).values = {{1, 0, 3, 1}};
  rhs.insert(2, 1).values = {{2, 0, 1, 1}};
  if (!Pruned) {
    lhs.insert(1, 1).values = {{1, -1, 1, 1}};
    rhs.insert(1, 0).values = {{1, 1, 2, 1}};
  }
  for (int compressed = 0; compressed < 2; ++compressed) {
    SparseMatrix<Scalar, DstOrder> sparse(2, 2), publicSparse;
    Matrix<Scalar, Dynamic, Dynamic, DstOrder> dense;
    if (Pruned) {
      internal::conservative_sparse_sparse_product_selector<decltype(lhs), decltype(rhs), decltype(sparse)>::run(
          lhs, rhs, sparse, internal::sparse_product_keep_above<double>{0.0});
      publicSparse = (lhs * rhs).pruned(Scalar(0), 0.0);
    } else {
      sparse = lhs * rhs;
      dense = lhs * rhs;
    }
    VERIFY(sparse.innerIndicesAreSorted());
    VERIFY_IS_EQUAL(sparse.nonZeros(), Pruned ? 1 : 2);
    for (Index j = 0; j < 2; ++j)
      for (Index i = 0; i < 2; ++i) {
        Scalar expected;
        for (Index k = 0; k < 3; ++k) expected += lhs.coeff(i, k) * rhs.coeff(k, j);
        const Scalar other = Pruned ? publicSparse.coeff(i, j) : dense(i, j);
        VERIFY(sparse.coeff(i, j).values == expected.values);
        VERIFY(other.values == expected.values);
      }
    lhs.makeCompressed();
    rhs.makeCompressed();
  }
}

// Operands that both lack the storage order of the result. The selector converts the operands, or, when the kernel in
// their storage order has far fewer terms and few products, runs it there and converts the result; the shapes take
// each way, and the last two have far fewer terms but too many products. The scalar
// does not commute, so a swapped operand order shows, and some sums cancel to zero, which the pruned product drops.
template <int OperandOrder>
void test_sparse_product_result_conversion() {
  using Scalar = noncommutative_scalar::NoncommutingBlock;
  constexpr int ResOrder = OperandOrder == ColMajor ? RowMajor : ColMajor;
  using Operand = SparseMatrix<Scalar, OperandOrder>;
  using Result = SparseMatrix<Scalar, ResOrder>;
  auto random_operand = [](Index rows, Index cols, Index perColumn) {
    Operand m(rows, cols);
    m.reserve(VectorXi::Constant(m.outerSize(), int(perColumn * cols / m.outerSize() + perColumn)));
    for (Index j = 0; j < cols; ++j) {
      std::set<Index> chosen;
      while (Index(chosen.size()) < perColumn) chosen.insert(internal::random<Index>(0, rows - 1));
      // A fifth of the entries are stored zeros, so that some sums are zero.
      for (Index i : chosen) {
        Scalar& entry = m.insert(i, j);
        if (internal::random<int>(0, 4) != 0)
          for (int& value : entry.values) value = internal::random<int>(-2, 2);
      }
    }
    return m;
  };
  struct Shape {
    Index rows, mid, cols, lhsPerColumn, rhsPerColumn;
  };
  const Shape shapes[] = {
      {60, 40, 30, 6, 1}, {30, 40, 60, 1, 6}, {40, 40, 40, 3, 3}, {60, 10, 30, 30, 4}, {30, 10, 60, 12, 5}};
  for (int s = 0; s < 5; ++s) {
    const Shape& shape = shapes[s];
    Operand lhs = random_operand(shape.rows, shape.mid, shape.lhsPerColumn);
    Operand rhs = random_operand(shape.mid, shape.cols, shape.rhsPerColumn);
    const bool convertsResult = OperandOrder == ColMajor ? internal::sparse_product_converts_result(lhs, rhs)
                                                         : internal::sparse_product_converts_result(rhs, lhs);
    const Index lhsNonZeros = lhs.nonZeros(), rhsNonZeros = rhs.nonZeros();
    const bool fewerTerms = OperandOrder == ColMajor ? lhsNonZeros > 2 * rhsNonZeros : rhsNonZeros > 2 * lhsNonZeros;
    VERIFY_IS_EQUAL(fewerTerms, s == (OperandOrder == ColMajor ? 0 : 1) || s == (OperandOrder == ColMajor ? 3 : 4));
    VERIFY_IS_EQUAL(convertsResult, s == (OperandOrder == ColMajor ? 0 : 1));

    // The reference sums the products in a dense matrix and marks the entries that the patterns generate.
    using Dense = Matrix<Scalar, Dynamic, Dynamic>;
    Dense expected = Dense::Constant(shape.rows, shape.cols, Scalar(0));
    Matrix<bool, Dynamic, Dynamic> generated = Matrix<bool, Dynamic, Dynamic>::Constant(shape.rows, shape.cols, false);
    for (Index lhsOuter = 0; lhsOuter < lhs.outerSize(); ++lhsOuter)
      for (typename Operand::InnerIterator lhsIt(lhs, lhsOuter); lhsIt; ++lhsIt)
        for (Index rhsOuter = 0; rhsOuter < rhs.outerSize(); ++rhsOuter)
          for (typename Operand::InnerIterator rhsIt(rhs, rhsOuter); rhsIt; ++rhsIt)
            if (rhsIt.row() == lhsIt.col()) {
              expected(lhsIt.row(), rhsIt.col()) += lhsIt.value() * rhsIt.value();
              generated(lhsIt.row(), rhsIt.col()) = true;
            }
    Index generatedCount = 0, nonZeroCount = 0;
    for (Index j = 0; j < shape.cols; ++j)
      for (Index i = 0; i < shape.rows; ++i) {
        generatedCount += generated(i, j);
        nonZeroCount += expected(i, j).values != Scalar(0).values;
      }
    VERIFY(nonZeroCount < generatedCount);

    for (int compressed = 0; compressed < 2; ++compressed) {
      Result conservative = lhs * rhs;
      // The public pruned product runs the selector for its own plain type; this selector is reached directly.
      Result pruned(shape.rows, shape.cols);
      internal::conservative_sparse_sparse_product_selector<Operand, Operand, Result>::run(
          lhs, rhs, pruned, internal::sparse_product_keep_above<double>{0.0});
      Result publicPruned = (lhs * rhs).pruned(Scalar(0), 0.0);
      for (const Result* result : {&conservative, &pruned, &publicPruned}) {
        VERIFY(result->isCompressed());
        VERIFY_IS_EQUAL(result->innerIndicesAreSorted(), result->outerSize());
        VERIFY_IS_EQUAL(result->nonZeros(), result == &conservative ? generatedCount : nonZeroCount);
        for (Index j = 0; j < shape.cols; ++j)
          for (Index i = 0; i < shape.rows; ++i) VERIFY(result->coeff(i, j).values == expected(i, j).values);
      }
      lhs.makeCompressed();
      rhs.makeCompressed();
    }
  }
}

template <typename = void>
void test_sparse_outer_operand_types() {
  SparseVector<noncommutative_scalar::TaggedScalar<0>> column(2);
  column.insert(1) = noncommutative_scalar::TaggedScalar<0>(3);
  Matrix<noncommutative_scalar::TaggedScalar<1>, 1, Dynamic> row(3);
  for (Index i = 0; i < row.size(); ++i) row(i) = noncommutative_scalar::TaggedScalar<1>(int(i + 2));
  using LeftEval = internal::sparse_dense_outer_product_evaluator<decltype(column), decltype(row), false>;
  const LeftEval left(column, row);
  for (Index j = 0; j < row.size(); ++j) {
    typename LeftEval::InnerIterator it(left, j);
    VERIFY(it);
    VERIFY_IS_EQUAL(it.row(), 1);
    VERIFY_IS_EQUAL(it.value().value, 3 * row(j).value);
    ++it;
    VERIFY(!it);
  }

  SparseVector<noncommutative_scalar::TaggedScalar<1>, RowMajor> sparseRow(3);
  sparseRow.insert(2) = noncommutative_scalar::TaggedScalar<1>(5);
  Matrix<noncommutative_scalar::TaggedScalar<0>, Dynamic, 1> denseColumn(2);
  denseColumn(0) = noncommutative_scalar::TaggedScalar<0>(2);
  denseColumn(1) = noncommutative_scalar::TaggedScalar<0>(4);
  using RightEval = internal::sparse_dense_outer_product_evaluator<decltype(denseColumn), decltype(sparseRow), true>;
  const RightEval right(denseColumn, sparseRow);
  for (Index i = 0; i < denseColumn.size(); ++i) {
    typename RightEval::InnerIterator it(right, i);
    VERIFY(it);
    VERIFY_IS_EQUAL(it.col(), 2);
    VERIFY_IS_EQUAL(it.value().value, denseColumn(i).value * 5);
  }
}

template <typename = void>
void test_sparse_outer_quaternion_order() {
  using Scalar = noncommutative_scalar::Quaternion;
  const Scalar i(0, 1, 0, 0), j(0, 0, 1, 0), k(0, 0, 0, 1);
  Matrix<Scalar, 1, 1> dense;
  SparseVector<Scalar, RowMajor> row(1);
  row.insert(0) = j;
  dense(0) = i;
  using Outer = internal::sparse_dense_outer_product_evaluator<decltype(dense), decltype(row), true>;
  const Outer outer(dense, row);
  VERIFY(typename Outer::InnerIterator(outer, 0).value() == k);
}

template <int Order>
void test_dense_sparse_quaternion_order() {
  using Scalar = noncommutative_scalar::Quaternion;
  const Scalar i(0, 1, 0, 0), j(0, 0, 1, 0);
  Matrix<Scalar, Dynamic, Dynamic, Order> dense = Matrix<Scalar, Dynamic, Dynamic, Order>::Constant(3, 3, i);
  Matrix<Scalar, 1, Dynamic> row = Matrix<Scalar, 1, Dynamic>::Constant(3, i);
  SparseMatrix<Scalar, Order> sparse(3, 3);
  sparse.reserve(VectorXi::Constant(sparse.outerSize(), 3));
  sparse.insert(0, 0) = j;
  sparse.insert(1, 2) = j;
  for (int compressed = 0; compressed < 2; ++compressed) {
    const Matrix<Scalar, Dynamic, Dynamic> product = dense * sparse;
    Matrix<Scalar, Dynamic, Dynamic> scaled = Matrix<Scalar, Dynamic, Dynamic>::Zero(3, 3);
    scaled.noalias() += j * (dense * sparse);
    Matrix<Scalar, Dynamic, Dynamic> sparseLeft = Matrix<Scalar, Dynamic, Dynamic>::Zero(3, 3);
    sparseLeft.noalias() += i * (sparse * dense);
    const Matrix<Scalar, 1, Dynamic> rowSparse = row * sparse;
    const SparseMatrix<Scalar, Order> cwise = dense.cwiseProduct(sparse);
    for (Index col = 0; col < sparse.cols(); ++col) {
      Scalar expectedRow;
      for (Index k = 0; k < row.size(); ++k) expectedRow += row(k) * sparse.coeff(k, col);
      VERIFY(rowSparse(col) == expectedRow);
      for (Index r = 0; r < dense.rows(); ++r) {
        Scalar expected;
        for (Index k = 0; k < dense.cols(); ++k) expected += dense(r, k) * sparse.coeff(k, col);
        VERIFY(product(r, col) == expected);
        VERIFY(scaled(r, col) == j * expected);
        Scalar expectedLeft;
        for (Index k = 0; k < dense.rows(); ++k) expectedLeft += sparse.coeff(r, k) * dense(k, col);
        VERIFY(sparseLeft(r, col) == i * expectedLeft);
        VERIFY(cwise.coeff(r, col) == dense(r, col) * sparse.coeff(r, col));
      }
    }
    sparse.makeCompressed();
  }
}

// Constructing an Array passes it to the dense * sparse kernel, which must not add a Matrix expression to it (#3227).
template <int Order>
void test_dense_sparse_array_destination() {
  const Index n = 6;
  const MatrixXd dense = MatrixXd::Random(n, n);
  const SparseMatrix<double, Order> sparse = MatrixXd::Random(n, n).sparseView(0.5, 1);
  const MatrixXd sparseDense(sparse);
  const ArrayXXd product = dense * sparse;
  VERIFY_IS_APPROX(product.matrix(), dense * sparseDense);
  // An expression without compressed storage takes the iterator kernel.
  const ArrayXXd absProduct = dense * sparse.cwiseAbs();
  VERIFY_IS_APPROX(absProduct.matrix(), dense * sparseDense.cwiseAbs());
  const ArrayXXd sparseFirst = sparse * dense;
  VERIFY_IS_APPROX(sparseFirst.matrix(), sparseDense * dense);
}

template <typename = void>
void test_sparse_vector_dense_product() {
  SparseVector<double> sv(3);
  sv.insert(0) = 1.0;
  sv.insert(2) = 2.0;

  MatrixXd dm = MatrixXd::Random(3, 2);
  MatrixXd res = sv.transpose() * dm;
  MatrixXd ref = MatrixXd(sv).transpose() * dm;
  VERIFY_IS_APPROX(res, ref);
}

// The inner iterator of a triangular view yields the entries of the view once and is inert after its end, also for the
// unit diagonal that the view adds. The dense-times-sparse kernels increment an iterator past its end.
template <int Order, int Mode>
void check_sparse_triangular_view_products(Index n) {
  using SM = SparseMatrix<double, Order>;
  using DenseMat = Matrix<double, Dynamic, Dynamic>;
  using View = TriangularView<const SM, Mode>;
  DenseMat D = (DenseMat::Random(n, n).array().abs() < 0.5).select(DenseMat::Random(n, n), 0.0);
  SM A = D.sparseView();
  A.makeCompressed();
  const DenseMat T = D.template triangularView<Mode>();
  const DenseMat v = DenseMat::Random(n, 1), B = DenseMat::Random(n, 3), rv = DenseMat::Random(1, n),
                 C = DenseMat::Random(3, n);

  // Direct iteration, with two increments past the end.
  const View view = A.template triangularView<Mode>();
  internal::evaluator<View> eval(view);
  DenseMat fromIterators = DenseMat::Zero(n, n);
  for (Index k = 0; k < A.outerSize(); ++k) {
    Index count = 0;
    typename internal::evaluator<View>::InnerIterator it(eval, k);
    for (; it && count <= n; ++it, ++count) fromIterators(it.row(), it.col()) += it.value();
    VERIFY(count <= n);
    VERIFY(!it);
    ++it;
    VERIFY(!it);
    ++it;
    VERIFY(!it);
  }
  VERIFY_IS_APPROX(fromIterators, T);

  VERIFY_IS_APPROX(DenseMat(view * v), DenseMat(T * v));
  VERIFY_IS_APPROX(DenseMat(view * B), DenseMat(T * B));
  VERIFY_IS_APPROX(DenseMat(rv * view), DenseMat(rv * T));
  VERIFY_IS_APPROX(DenseMat(C * view), DenseMat(C * T));
  VERIFY_IS_APPROX(DenseMat(v.transpose() * view), DenseMat(v.transpose() * T));

  // The view of a transposed matrix.
  const DenseMat Tt = D.transpose().template triangularView<Mode>();
  VERIFY_IS_APPROX(DenseMat(A.transpose().template triangularView<Mode>() * v), DenseMat(Tt * v));
  VERIFY_IS_APPROX(DenseMat(A.transpose().template triangularView<Mode>() * B), DenseMat(Tt * B));
  VERIFY_IS_APPROX(DenseMat(rv * A.transpose().template triangularView<Mode>()), DenseMat(rv * Tt));
  VERIFY_IS_APPROX(DenseMat(C * A.transpose().template triangularView<Mode>()), DenseMat(C * Tt));
}

template <int Order>
void check_sparse_triangular_view_products_all_modes() {
  for (Index n : {1, 2, 5}) {
    check_sparse_triangular_view_products<Order, Lower>(n);
    check_sparse_triangular_view_products<Order, Upper>(n);
    check_sparse_triangular_view_products<Order, UnitLower>(n);
    check_sparse_triangular_view_products<Order, UnitUpper>(n);
    check_sparse_triangular_view_products<Order, StrictlyLower>(n);
    check_sparse_triangular_view_products<Order, StrictlyUpper>(n);
  }
}

// Products large enough that the kernel takes each of its ways of building a vector. Long vectors whose indices
// rarely repeat go through a direct table if they are dense in their range, and are otherwise appended to buckets.
// A bucket is sorted by insertion; one that a cluster overfills goes through a direct table or is split again. The
// other vectors are marked and then sorted: already in order, by insertion, through a direct table, or in buckets.
// The reference sums the same products through setFromTriplets. The entries are signed, so some sums cancel, and
// the pruned product must drop exactly those at or below its tolerance on each of the paths.
template <int LhsOrder, int RhsOrder, int ResOrder>
void sparse_product_large() {
  using Triplets = std::vector<Triplet<double>>;
  enum Pattern { Scattered, Run, Window, Pooled };
  // perColumn entries per column: at random rows, in one contiguous run from a random row, at random rows among the
  // 3000 that follow a random row, or 100 of the 300 scattered rows of one of three pools, so that columns of a pool
  // share most of their rows.
  auto random_matrix = [](Index rows, Index cols, Index perColumn, Pattern pattern) {
    Triplets triplets;
    for (Index j = 0; j < cols; ++j) {
      Index first = internal::random<Index>(0, rows - 1);
      for (Index k = 0; k < perColumn; ++k) {
        Index i = internal::random<Index>(0, rows - 1);
        if (pattern == Run) i = (first + k) % rows;
        if (pattern == Window) i = (first + internal::random<Index>(0, 2999)) % rows;
        if (pattern == Pooled) i = ((j % 3) * 7919 + ((j * 37 + k * 3) % 300) * 127) % rows;
        Index v = (i + 3 * j + k) % 7 - 3;
        triplets.emplace_back(i, j, double(v == 0 ? 4 : v));
      }
    }
    SparseMatrix<double> m(rows, cols);
    m.setFromTriplets(triplets.begin(), triplets.end());
    return m;
  };
  struct Shape {
    Index rows, mid, cols, lhsPerColumn, rhsPerColumn;
    Pattern pattern;
  };
  const Shape shapes[] = {
      {3000, 3000, 300, 40, 40, Scattered},  // about 1200 distinct indices of 1600 products on 3000 rows: direct
      {60000, 200, 200, 300, 3, Scattered},  // 900 products on 60000 rows: appended, buckets sorted by insertion
      {200000, 300, 200, 100, 4, Run},       // four runs of 100: appended, each cluster through a direct table
      {400000, 300, 200, 60, 4, Window},     // four windows of 60: appended, each cluster split again
      {400, 400, 200, 60, 60, Scattered},    // 3600 products on 400 rows: marked, then direct as dense in its range
      {40000, 300, 200, 100, 30, Pooled},    // 3000 products on 900 scattered rows: marked, sorted in buckets
      {5000, 500, 500, 20, 3, Run},          // three runs of 20: marked, then in order or sorted by insertion
      {5000, 500, 500, 20, 2, Run},          // two runs of 20: merged
      {60000, 200, 200, 300, 2, Scattered},  // two scattered vectors of 300: merged
      {20000, 300, 100, 40, 1, Run},         // one run of 40: copied; few rhs entries
  };
  for (const Shape& shape : shapes) {
    SparseMatrix<double> a = random_matrix(shape.rows, shape.mid, shape.lhsPerColumn, shape.pattern);
    SparseMatrix<double> b = random_matrix(shape.mid, shape.cols, shape.rhsPerColumn, Scattered);
    Triplets triplets;
    for (Index j = 0; j < b.outerSize(); ++j)
      for (SparseMatrix<double>::InnerIterator bIt(b, j); bIt; ++bIt)
        for (SparseMatrix<double>::InnerIterator aIt(a, bIt.index()); aIt; ++aIt)
          triplets.emplace_back(aIt.index(), j, aIt.value() * bIt.value());
    SparseMatrix<double, ResOrder> ref(shape.rows, shape.cols);
    ref.setFromTriplets(triplets.begin(), triplets.end());

    SparseMatrix<double, LhsOrder> lhs = std::move(a);
    SparseMatrix<double, RhsOrder> rhs = std::move(b);
    SparseMatrix<double, ResOrder> res = lhs * rhs;
    VERIFY(res.isCompressed());
    VERIFY_IS_EQUAL(res.innerIndicesAreSorted(), res.outerSize());
    VERIFY_IS_EQUAL(res.nonZeros(), ref.nonZeros());
    // integer-valued entries: the sums are exact in any order
    VERIFY_IS_EQUAL((res - ref).norm(), 0.0);

    // The pruned product drops the sums v with |v| <= tolerance: exact zeros, then also 1 and 2.
    for (double tolerance : {0.0, 2.5}) {
      SparseMatrix<double, ResOrder> prunedRef = ref;
      prunedRef.prune([tolerance](Index, Index, double v) { return numext::abs(v) > tolerance; });
      SparseMatrix<double, ResOrder> pruned = (lhs * rhs).pruned(1.0, tolerance);
      VERIFY(pruned.isCompressed());
      VERIFY_IS_EQUAL(pruned.innerIndicesAreSorted(), pruned.outerSize());
      VERIFY_IS_EQUAL(pruned.nonZeros(), prunedRef.nonZeros());
      VERIFY_IS_EQUAL((pruned - prunedRef).norm(), 0.0);
      if (tolerance > 0.0) VERIFY(pruned.nonZeros() < res.nonZeros());
    }
  }
}

// Long inner vectors of the sorted product, for a scalar type, an index type and each storage order of the operands
// and of the result: these take the bucket paths, which the small products above do not reach. The scalars are
// integer-valued, so every sum is exact and the result is compared with == against products summed in a map.
//
// x has long inner vectors with a dense run, scattered indices, and indices at geometrically growing gaps, which
// are clusters within clusters and split more than once. The vectors of y choose one vector of x, or three with the
// same indices; a vector with three products per index makes the next one marked. For a row-major result the
// product is y^T x^T, so that the long vectors are the rows of the right operand. With unsortedOperands, the inner
// vectors of both operands are stored in reverse order.
template <typename Scalar>
Scalar sparse_product_test_value(Index v, Scalar*) {
  return Scalar(double(v));
}
template <typename Real>
std::complex<Real> sparse_product_test_value(Index v, std::complex<Real>*) {
  return std::complex<Real>(Real(v), Real(v % 3 - 1));
}
inline noncommutative_scalar::Quaternion sparse_product_test_value(Index v, noncommutative_scalar::Quaternion*) {
  return noncommutative_scalar::Quaternion(double(v), double(v % 3 - 1), double(v % 2), double(v % 5 - 2));
}

template <typename Scalar, typename StorageIndex, int LhsOrder, int RhsOrder, int ResOrder>
void sparse_product_long_vectors(bool unsortedOperands) {
  using Entry = std::pair<Index, Index>;
  using Entries = std::map<Entry, Scalar>;
  const Index inner = numext::mini<Index>(2000000, Index(NumTraits<StorageIndex>::highest()));
  const Index mid = 4, outer = 4;

  std::vector<Index> indices;
  for (Index k = 0; k < 30; ++k) indices.push_back(inner / 3 + k);
  for (Index k = 0; k < 60; ++k) indices.push_back((inner / 61) * k + 7);
  double gap = 1.0;
  for (Index k = 0; Index(gap) + k < inner / 4; ++k, gap *= 1.25) indices.push_back(inner / 2 + Index(gap) + k);
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  VERIFY(Index(indices.size()) > 64);

  // x(i, k) for i in indices; y(k, j): vectors 0 and 3 take x(:, 3), vectors 1 and 2 take x(:, 0..2).
  Entries x, y;
  for (Index k = 0; k < mid; ++k)
    for (Index q = 0; q < Index(indices.size()); ++q)
      x[Entry(indices[q], k)] = sparse_product_test_value(1 + (q + 2 * k) % 5, static_cast<Scalar*>(nullptr));
  y[Entry(3, 0)] = sparse_product_test_value(2, static_cast<Scalar*>(nullptr));
  y[Entry(3, 3)] = sparse_product_test_value(3, static_cast<Scalar*>(nullptr));
  for (Index j = 1; j < 3; ++j)
    for (Index k = 0; k < 3; ++k) y[Entry(k, j)] = sparse_product_test_value(1 + j + k, static_cast<Scalar*>(nullptr));

  const bool transposed = ResOrder == RowMajor;
  auto transpose = [](const Entries& m) {
    Entries t;
    for (const auto& e : m) t[Entry(e.first.second, e.first.first)] = e.second;
    return t;
  };
  const Entries l = transposed ? transpose(y) : x;
  const Entries r = transposed ? transpose(x) : y;
  const Index rows = transposed ? outer : inner, cols = transposed ? inner : outer;
  const Index middle = mid;

  auto build = [&](const Entries& m, Index nrows, Index ncols, auto& matrix) {
    std::vector<Triplet<Scalar, StorageIndex>> triplets;
    for (const auto& e : m) triplets.emplace_back(StorageIndex(e.first.first), StorageIndex(e.first.second), e.second);
    matrix.resize(nrows, ncols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    if (unsortedOperands)
      for (Index j = 0; j < matrix.outerSize(); ++j) {
        Index begin = matrix.outerIndexPtr()[j], end = matrix.outerIndexPtr()[j + 1];
        std::reverse(matrix.innerIndexPtr() + begin, matrix.innerIndexPtr() + end);
        std::reverse(matrix.valuePtr() + begin, matrix.valuePtr() + end);
      }
  };
  SparseMatrix<Scalar, LhsOrder, StorageIndex> lhs;
  SparseMatrix<Scalar, RhsOrder, StorageIndex> rhs;
  build(l, rows, middle, lhs);
  build(r, middle, cols, rhs);

  Entries ref;
  for (const auto& b : r)
    for (const auto& a : l)
      if (a.first.second == b.first.first) {
        Entry at(a.first.first, b.first.second);
        auto found = ref.find(at);
        if (found == ref.end())
          ref[at] = a.second * b.second;
        else
          found->second += a.second * b.second;
      }

  SparseMatrix<Scalar, ResOrder, StorageIndex> res = lhs * rhs;
  VERIFY(res.isCompressed());
  VERIFY_IS_EQUAL(res.innerIndicesAreSorted(), res.outerSize());
  VERIFY_IS_EQUAL(res.nonZeros(), Index(ref.size()));
  Index mismatches = 0;
  for (Index j = 0; j < res.outerSize(); ++j)
    for (typename SparseMatrix<Scalar, ResOrder, StorageIndex>::InnerIterator it(res, j); it; ++it) {
      auto found = ref.find(Entry(it.row(), it.col()));
      mismatches += found == ref.end() || !(found->second == it.value());
    }
  VERIFY_IS_EQUAL(mismatches, Index(0));

  // The pruned product keeps exactly the entries above its tolerance.
  using RealScalar = typename NumTraits<Scalar>::Real;
  const RealScalar tolerance = RealScalar(12);
  Index expected = 0;
  for (const auto& e : ref) expected += !(numext::abs(e.second) <= tolerance);
  VERIFY(expected > 0 && expected < Index(ref.size()));
  SparseMatrix<Scalar, ResOrder, StorageIndex> pruned = (lhs * rhs).pruned(Scalar(1), tolerance);
  VERIFY(pruned.isCompressed());
  VERIFY_IS_EQUAL(pruned.innerIndicesAreSorted(), pruned.outerSize());
  VERIFY_IS_EQUAL(pruned.nonZeros(), expected);
  mismatches = 0;
  for (Index j = 0; j < pruned.outerSize(); ++j)
    for (typename SparseMatrix<Scalar, ResOrder, StorageIndex>::InnerIterator it(pruned, j); it; ++it) {
      auto found = ref.find(Entry(it.row(), it.col()));
      mismatches += found == ref.end() || !(found->second == it.value()) || numext::abs(it.value()) <= tolerance;
    }
  VERIFY_IS_EQUAL(mismatches, Index(0));
}

template <typename Scalar, typename StorageIndex>
void sparse_product_long_vectors_all_orders() {
  for (bool unsortedOperands : {false, true}) {
    sparse_product_long_vectors<Scalar, StorageIndex, ColMajor, ColMajor, ColMajor>(unsortedOperands);
    sparse_product_long_vectors<Scalar, StorageIndex, RowMajor, ColMajor, ColMajor>(unsortedOperands);
    sparse_product_long_vectors<Scalar, StorageIndex, ColMajor, RowMajor, ColMajor>(unsortedOperands);
    sparse_product_long_vectors<Scalar, StorageIndex, RowMajor, RowMajor, ColMajor>(unsortedOperands);
    sparse_product_long_vectors<Scalar, StorageIndex, ColMajor, ColMajor, RowMajor>(unsortedOperands);
    sparse_product_long_vectors<Scalar, StorageIndex, RowMajor, ColMajor, RowMajor>(unsortedOperands);
    sparse_product_long_vectors<Scalar, StorageIndex, ColMajor, RowMajor, RowMajor>(unsortedOperands);
    sparse_product_long_vectors<Scalar, StorageIndex, RowMajor, RowMajor, RowMajor>(unsortedOperands);
  }
}

// A matrix may have highest() + 1 rows, so that highest() is an inner index. The direct table of the sorted product
// must keep that index: in the first vector of a product, in a vector that is marked and then sorted, and in a
// cluster at the end of a vector that is appended to buckets.
template <typename StorageIndex>
void sparse_product_highest_inner_index() {
  using SpMat = SparseMatrix<double, ColMajor, StorageIndex>;
  const Index rows = Index(NumTraits<StorageIndex>::highest()) + 1;
  {
    SpMat a(rows, 2), b(2, 2);
    a.insert(rows - 4, 0) = 1.0;
    a.insert(rows - 1, 0) = 2.0;
    a.insert(rows - 3, 1) = 3.0;
    a.insert(rows - 2, 1) = 4.0;
    a.insert(rows - 1, 1) = 5.0;
    for (Index j = 0; j < 2; ++j)
      for (Index i = 0; i < 2; ++i) b.insert(i, j) = 1.0;
    SpMat c = a * b;
    VERIFY_IS_EQUAL(c.innerIndicesAreSorted(), c.outerSize());
    VERIFY_IS_EQUAL(c.nonZeros(), Index(8));
    for (Index j = 0; j < 2; ++j) {
      VERIFY_IS_EQUAL(c.coeff(rows - 4, j), 1.0);
      VERIFY_IS_EQUAL(c.coeff(rows - 3, j), 3.0);
      VERIFY_IS_EQUAL(c.coeff(rows - 2, j), 4.0);
      VERIFY_IS_EQUAL(c.coeff(rows - 1, j), 7.0);
    }
  }
  {
    // row 0 and the last 12 rows, twice: 26 products
    const Index cluster = 12;
    SpMat a(rows, 2), b(2, 1);
    for (Index j = 0; j < 2; ++j) {
      a.insert(0, j) = 1.0;
      for (Index k = 0; k < cluster; ++k) a.insert(rows - cluster + k, j) = double(k + 1);
      b.insert(j, 0) = 1.0;
    }
    SpMat c = a * b;
    VERIFY_IS_EQUAL(c.innerIndicesAreSorted(), c.outerSize());
    VERIFY_IS_EQUAL(c.nonZeros(), cluster + 1);
    VERIFY_IS_EQUAL(c.coeff(0, 0), 2.0);
    for (Index k = 0; k < cluster; ++k) VERIFY_IS_EQUAL(c.coeff(rows - cluster + k, 0), double(2 * (k + 1)));
  }
}

EIGEN_DECLARE_TEST(sparse_product) {
  sparse_structured_view_product_sfinae<SparseMatrix<double, ColMajor>>();
  sparse_structured_view_product_sfinae<SparseMatrix<double, RowMajor>>();

  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1((test_sparse_vector_dense_product<>()));
    CALL_SUBTEST_1((check_sparse_triangular_view_products_all_modes<ColMajor>()));
    CALL_SUBTEST_1((check_sparse_triangular_view_products_all_modes<RowMajor>()));
    CALL_SUBTEST_1((test_pruned_product_row_vector<>()));
    CALL_SUBTEST_1((test_sparse_product_mul_assign<>()));
    CALL_SUBTEST_1((test_sparse_dense_product_empty_blocks<ColMajor, ColMajor>()));
    CALL_SUBTEST_1((test_sparse_dense_product_empty_blocks<ColMajor, RowMajor>()));
    CALL_SUBTEST_1((test_sparse_dense_product_empty_blocks<RowMajor, ColMajor>()));
    CALL_SUBTEST_1((test_sparse_dense_product_empty_blocks<RowMajor, RowMajor>()));
    CALL_SUBTEST_1((sparse_product<SparseMatrix<double, ColMajor>>()));
    CALL_SUBTEST_1((sparse_product<SparseMatrix<double, RowMajor>>()));
    CALL_SUBTEST_1((bug_942<double>()));
    CALL_SUBTEST_2((sparse_product<SparseMatrix<std::complex<double>, ColMajor>>()));
    CALL_SUBTEST_2((sparse_product<SparseMatrix<std::complex<double>, RowMajor>>()));
    CALL_SUBTEST_3((sparse_product<SparseMatrix<float, ColMajor, long int>>()));
    CALL_SUBTEST_4(
        (sparse_product_regression_test<SparseMatrix<double, RowMajor>, Matrix<double, Dynamic, Dynamic, RowMajor>>()));

    CALL_SUBTEST_5((test_mixing_types<float>()));
    CALL_SUBTEST_5((test_mixed_storage<>()));
    CALL_SUBTEST_5((test_sparse_cwise_product_operand_types<ColMajor>()));
    CALL_SUBTEST_7((test_sparse_cwise_sum_operand_types<RowMajor>()));
    CALL_SUBTEST_8((test_sparse_diagonal_products<ColMajor>()));
    CALL_SUBTEST_8((test_sparse_diagonal_products<RowMajor>()));
    CALL_SUBTEST_9((test_sparse_product_operand_order<false, ColMajor, ColMajor, ColMajor>()));
    CALL_SUBTEST_9((test_sparse_product_operand_order<false, ColMajor, ColMajor, RowMajor>()));
    CALL_SUBTEST_9((test_sparse_product_operand_order<false, ColMajor, RowMajor, ColMajor>()));
    CALL_SUBTEST_9((test_sparse_product_operand_order<false, ColMajor, RowMajor, RowMajor>()));
    CALL_SUBTEST_9((test_sparse_product_operand_order<false, RowMajor, ColMajor, ColMajor>()));
    CALL_SUBTEST_9((test_sparse_product_operand_order<false, RowMajor, ColMajor, RowMajor>()));
    CALL_SUBTEST_9((test_sparse_product_operand_order<false, RowMajor, RowMajor, ColMajor>()));
    CALL_SUBTEST_9((test_sparse_product_operand_order<false, RowMajor, RowMajor, RowMajor>()));
    CALL_SUBTEST_10((test_sparse_outer_operand_types<>()));
    CALL_SUBTEST_10((test_dense_sparse_array_destination<ColMajor>()));
    CALL_SUBTEST_10((test_dense_sparse_array_destination<RowMajor>()));
    CALL_SUBTEST_12((test_sparse_outer_quaternion_order<>()));
    CALL_SUBTEST_14((test_dense_sparse_quaternion_order<ColMajor>()));
    CALL_SUBTEST_14((test_dense_sparse_quaternion_order<RowMajor>()));
    CALL_SUBTEST_11((test_sparse_product_operand_order<true, ColMajor, ColMajor, ColMajor>()));
    CALL_SUBTEST_11((test_sparse_product_operand_order<true, ColMajor, ColMajor, RowMajor>()));
    CALL_SUBTEST_11((test_sparse_product_operand_order<true, ColMajor, RowMajor, ColMajor>()));
    CALL_SUBTEST_11((test_sparse_product_operand_order<true, ColMajor, RowMajor, RowMajor>()));
    CALL_SUBTEST_11((test_sparse_product_operand_order<true, RowMajor, ColMajor, ColMajor>()));
    CALL_SUBTEST_11((test_sparse_product_operand_order<true, RowMajor, ColMajor, RowMajor>()));
    CALL_SUBTEST_11((test_sparse_product_operand_order<true, RowMajor, RowMajor, ColMajor>()));
    CALL_SUBTEST_11((test_sparse_product_operand_order<true, RowMajor, RowMajor, RowMajor>()));
    CALL_SUBTEST_11((test_sparse_product_result_conversion<ColMajor>()));
    CALL_SUBTEST_11((test_sparse_product_result_conversion<RowMajor>()));

    CALL_SUBTEST_6((test_pruned_product_custom_scalar<>()));
    CALL_SUBTEST_6((test_pruned_product_front_insertion<>()));
    CALL_SUBTEST_6((test_pruned_product_cancelled_vector<>()));
    CALL_SUBTEST_6((test_sparse_product_few_terms_unsorted<>()));
    CALL_SUBTEST_6((test_sparse_product_small_vectors<double, ColMajor>()));
    CALL_SUBTEST_6((test_sparse_product_small_vectors<double, RowMajor>()));
    CALL_SUBTEST_6((test_sparse_product_small_vectors<std::complex<double>, ColMajor>()));
    CALL_SUBTEST_6((test_sparse_product_small_vectors_huge_indices<>()));
    CALL_SUBTEST_6((test_sparse_product_huge_inner_dimension<>()));
  }
  CALL_SUBTEST_3((sparse_product_long_vectors_all_orders<double, int>()));
  CALL_SUBTEST_4((sparse_product_long_vectors_all_orders<double, short>()));
  CALL_SUBTEST_4((sparse_product_long_vectors_all_orders<double, long long>()));
  CALL_SUBTEST_5((sparse_product_long_vectors_all_orders<std::complex<double>, int>()));
  CALL_SUBTEST_7((sparse_product_long_vectors_all_orders<noncommutative_scalar::Quaternion, int>()));
  CALL_SUBTEST_3((sparse_product_highest_inner_index<signed char>()));
  CALL_SUBTEST_3((sparse_product_highest_inner_index<short>()));
  CALL_SUBTEST_3((sparse_product_large<ColMajor, ColMajor, ColMajor>()));
  CALL_SUBTEST_3((sparse_product_large<RowMajor, RowMajor, RowMajor>()));
  CALL_SUBTEST_3((sparse_product_large<RowMajor, ColMajor, ColMajor>()));
  CALL_SUBTEST_3((sparse_product_large<ColMajor, ColMajor, RowMajor>()));
}

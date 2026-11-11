// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_CONSERVATIVESPARSESPARSEPRODUCT_H
#define EIGEN_CONSERVATIVESPARSESPARSEPRODUCT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename Lhs, typename Rhs, typename ResultType, bool ReverseProduct = false>
static void conservative_sparse_sparse_product_impl(const Lhs& lhs, const Rhs& rhs, ResultType& res,
                                                    bool sortedInsertion = false) {
  using LhsScalar = typename remove_all_t<Lhs>::Scalar;
  using RhsScalar = typename remove_all_t<Rhs>::Scalar;
  using ResScalar = typename remove_all_t<ResultType>::Scalar;
  const sparse_scalar_product_op<LhsScalar, RhsScalar, ReverseProduct> product;

  // make sure to call innerSize/outerSize since we fake the storage order.
  Index rows = lhs.innerSize();
  Index cols = rhs.outerSize();
  eigen_assert(lhs.outerSize() == rhs.innerSize());

  ei_declare_aligned_stack_constructed_variable(bool, mask, rows, 0);
  ei_declare_aligned_stack_constructed_variable(ResScalar, values, rows, 0);
  ei_declare_aligned_stack_constructed_variable(Index, indices, rows, 0);

  std::fill_n(mask, rows, false);

  evaluator<Lhs> lhsEval(lhs);
  evaluator<Rhs> rhsEval(rhs);

  // estimate the number of non zero entries
  // given a rhs column containing Y non zeros, we assume that the respective Y columns
  // of the lhs differs in average of one non zeros, thus the number of non zeros for
  // the product of a rhs column with the lhs is X+Y where X is the average number of non zero
  // per column of the lhs.
  // Therefore, we have nnz(lhs*rhs) = nnz(lhs) + nnz(rhs)
  Index estimated_nnz_prod = lhsEval.nonZerosEstimate() + rhsEval.nonZerosEstimate();

  res.setZero();
  res.reserve(Index(estimated_nnz_prod));
  // we compute each column of the result, one after the other
  for (Index j = 0; j < cols; ++j) {
    res.startVec(j);
    Index nnz = 0;
    for (typename evaluator<Rhs>::InnerIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt) {
      RhsScalar y = rhsIt.value();
      Index k = rhsIt.index();
      for (typename evaluator<Lhs>::InnerIterator lhsIt(lhsEval, k); lhsIt; ++lhsIt) {
        Index i = lhsIt.index();
        LhsScalar x = lhsIt.value();
        if (!mask[i]) {
          mask[i] = true;
          values[i] = product(x, y);
          indices[nnz] = i;
          ++nnz;
        } else
          product.accumulate(values[i], x, y);
      }
    }
    if (!sortedInsertion) {
      // unordered insertion
      for (Index k = 0; k < nnz; ++k) {
        Index i = indices[k];
        res.insertBackByOuterInnerUnordered(j, i) = values[i];
        mask[i] = false;
      }
    } else {
      // alternative ordered insertion code:
      const Index t200 = rows / 11;  // 11 == (log2(200)*1.39)
      const Index t = (rows * 100) / 139;

      // if the result is sparse enough => use a quick sort
      // otherwise => loop through the entire vector
      // In order to avoid to perform an expensive log2 when the
      // result is clearly very sparse we use a linear bound up to 200.
      if ((nnz < 200 && nnz < t200) || nnz * numext::log2(int(nnz)) < t) {
        if (nnz > 1) std::sort(indices, indices + nnz);
        for (Index k = 0; k < nnz; ++k) {
          Index i = indices[k];
          res.insertBackByOuterInner(j, i) = values[i];
          mask[i] = false;
        }
      } else {
        // dense path
        for (Index i = 0; i < rows; ++i) {
          if (mask[i]) {
            mask[i] = false;
            res.insertBackByOuterInner(j, i) = values[i];
          }
        }
      }
    }
  }
  res.finalize();
}

// Computes res = lhs * rhs with sorted inner vectors: each vector of res is accumulated by an
// ordered_hash_accumulator that lives in res's own free storage, so there is no workspace, no sort, and no transpose.
template <typename Lhs, typename Rhs, typename ResultType, bool ReverseProduct = false>
static void conservative_sparse_sparse_product_hash_impl(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
  using LhsScalar = typename remove_all_t<Lhs>::Scalar;
  using RhsScalar = typename remove_all_t<Rhs>::Scalar;
  using ResScalar = typename remove_all_t<ResultType>::Scalar;
  using StorageIndex = typename remove_all_t<ResultType>::StorageIndex;
  const sparse_scalar_product_op<LhsScalar, RhsScalar, ReverseProduct> product;

  // make sure to call innerSize/outerSize since we fake the storage order.
  Index cols = rhs.outerSize();
  Index mid = lhs.outerSize();
  eigen_assert(mid == rhs.innerSize());

  evaluator<Lhs> lhsEval(lhs);
  evaluator<Rhs> rhsEval(rhs);

  // The size and the index range of each lhs vector bound those of each result vector.
  ei_declare_aligned_stack_constructed_variable(Index, lhsCount, mid, 0);
  ei_declare_aligned_stack_constructed_variable(StorageIndex, lhsLo, mid, 0);
  ei_declare_aligned_stack_constructed_variable(StorageIndex, lhsHi, mid, 0);
  for (Index k = 0; k < mid; ++k) {
    Index count = 0;
    StorageIndex lo = NumTraits<StorageIndex>::highest(), hi = 0;
    for (typename evaluator<Lhs>::InnerIterator lhsIt(lhsEval, k); lhsIt; ++lhsIt, ++count) {
      lo = numext::mini(lo, StorageIndex(lhsIt.index()));
      hi = numext::maxi(hi, StorageIndex(lhsIt.index()));
    }
    lhsCount[k] = count;
    lhsLo[k] = lo;
    lhsHi[k] = hi;
  }

  res.setZero();
  res.data().reserve(lhsEval.nonZerosEstimate() + rhsEval.nonZerosEstimate());
  StorageIndex* outer = res.outerIndexPtr();
  ordered_hash_accumulator<ResScalar, StorageIndex> acc;
  Index nnz = 0;
  Index prevKeys = 0;
  for (Index j = 0; j < cols; ++j) {
    outer[j] = convert_index<StorageIndex>(nnz);
    Index maxKeys = 0;
    StorageIndex lo = NumTraits<StorageIndex>::highest(), hi = 0;
    for (typename evaluator<Rhs>::InnerIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt) {
      Index k = rhsIt.index();
      if (lhsCount[k] == 0) continue;
      maxKeys += lhsCount[k];
      lo = numext::mini(lo, lhsLo[k]);
      hi = numext::maxi(hi, lhsHi[k]);
    }
    if (maxKeys == 0) continue;
    Index size = acc.size(lo, hi, maxKeys);
    if (res.data().allocatedSize() < nnz + size) {
      res.data().resize(nnz);
      res.data().reserve(numext::maxi(size, nnz));
    }
    // The previous vector predicts the size of this one; maxKeys is often far above it.
    Index expectedKeys = numext::maxi<Index>(16, prevKeys + prevKeys / 4);
    acc.init(res.data().indexPtr() + nnz, res.data().valuePtr() + nnz, lo, hi, maxKeys, expectedKeys);
    for (typename evaluator<Rhs>::InnerIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt) {
      RhsScalar y = rhsIt.value();
      for (typename evaluator<Lhs>::InnerIterator lhsIt(lhsEval, rhsIt.index()); lhsIt; ++lhsIt) {
        bool inserted;
        ResScalar& value = acc.findOrInsert(StorageIndex(lhsIt.index()), inserted);
        if (inserted)
          value = product(lhsIt.value(), y);
        else
          product.accumulate(value, lhsIt.value(), y);
      }
    }
    prevKeys = acc.compact();
    nnz += prevKeys;
  }
  outer[cols] = convert_index<StorageIndex>(nnz);
  res.data().resize(nnz);
}

// Helper template to generate new sparse matrix types
template <class Source, int Order>
using WithStorageOrder = SparseMatrix<typename Source::Scalar, Order, typename Source::StorageIndex>;

template <typename Lhs, typename Rhs, typename ResultType,
          int LhsStorageOrder = (traits<Lhs>::Flags & RowMajorBit) ? RowMajor : ColMajor,
          int RhsStorageOrder = (traits<Rhs>::Flags & RowMajorBit) ? RowMajor : ColMajor,
          int ResStorageOrder = (traits<ResultType>::Flags & RowMajorBit) ? RowMajor : ColMajor>
struct conservative_sparse_sparse_product_selector;

// The hash kernel yields sorted vectors in the storage order of its operands, so each selector below converts at
// most one operand. When both operands have the storage order the result lacks, the unsorted kernel runs and the
// storage-order change of the result sorts it.
template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, ColMajor, ColMajor, ColMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
    ColMajorRes resCol(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_hash_impl<Lhs, Rhs, ColMajorRes>(lhs, rhs, resCol);
    res = resCol.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, RowMajor, ColMajor, ColMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using ColMajorLhs = WithStorageOrder<Lhs, ColMajor>;
    ColMajorLhs lhsCol = lhs;
    using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
    ColMajorRes resCol(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_hash_impl<ColMajorLhs, Rhs, ColMajorRes>(lhsCol, rhs, resCol);
    res = resCol.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, ColMajor, RowMajor, ColMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using ColMajorRhs = WithStorageOrder<Rhs, ColMajor>;
    ColMajorRhs rhsCol = rhs;
    using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
    ColMajorRes resCol(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_hash_impl<Lhs, ColMajorRhs, ColMajorRes>(lhs, rhsCol, resCol);
    res = resCol.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, RowMajor, RowMajor, ColMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
    RowMajorRes resRow(lhs.rows(), rhs.cols());
    // The storage-order change sorts the result, so the product can stay unsorted.
    internal::conservative_sparse_sparse_product_impl<Rhs, Lhs, RowMajorRes, true>(rhs, lhs, resRow);
    res = resRow;
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, ColMajor, ColMajor, RowMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
    ColMajorRes resCol(lhs.rows(), rhs.cols());
    // The storage-order change sorts the result, so the product can stay unsorted.
    internal::conservative_sparse_sparse_product_impl<Lhs, Rhs, ColMajorRes>(lhs, rhs, resCol);
    res = resCol;
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, RowMajor, ColMajor, RowMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using RowMajorRhs = WithStorageOrder<Rhs, RowMajor>;
    RowMajorRhs rhsRow = rhs;
    using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
    RowMajorRes resRow(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_hash_impl<RowMajorRhs, Lhs, RowMajorRes, true>(rhsRow, lhs, resRow);
    res = resRow.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, ColMajor, RowMajor, RowMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using RowMajorLhs = WithStorageOrder<Lhs, RowMajor>;
    RowMajorLhs lhsRow = lhs;
    using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
    RowMajorRes resRow(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_hash_impl<Rhs, RowMajorLhs, RowMajorRes, true>(rhs, lhsRow, resRow);
    res = resRow.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, RowMajor, RowMajor, RowMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
    RowMajorRes resRow(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_hash_impl<Rhs, Lhs, RowMajorRes, true>(rhs, lhs, resRow);
    res = resRow.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType, bool ReverseProduct = false>
static void sparse_sparse_to_dense_product_impl(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
  using LhsScalar = typename remove_all_t<Lhs>::Scalar;
  using RhsScalar = typename remove_all_t<Rhs>::Scalar;
  const sparse_scalar_product_op<LhsScalar, RhsScalar, ReverseProduct> product;
  Index cols = rhs.outerSize();
  eigen_assert(lhs.outerSize() == rhs.innerSize());

  evaluator<Lhs> lhsEval(lhs);
  evaluator<Rhs> rhsEval(rhs);

  for (Index j = 0; j < cols; ++j) {
    for (typename evaluator<Rhs>::InnerIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt) {
      RhsScalar y = rhsIt.value();
      Index k = rhsIt.index();
      for (typename evaluator<Lhs>::InnerIterator lhsIt(lhsEval, k); lhsIt; ++lhsIt) {
        Index i = lhsIt.index();
        LhsScalar x = lhsIt.value();
        product.accumulate(res.coeffRef(i, j), x, y);
      }
    }
  }
}

}  // end namespace internal

namespace internal {

template <typename Lhs, typename Rhs, typename ResultType,
          int LhsStorageOrder = (traits<Lhs>::Flags & RowMajorBit) ? RowMajor : ColMajor,
          int RhsStorageOrder = (traits<Rhs>::Flags & RowMajorBit) ? RowMajor : ColMajor>
struct sparse_sparse_to_dense_product_selector;

template <typename Lhs, typename Rhs, typename ResultType>
struct sparse_sparse_to_dense_product_selector<Lhs, Rhs, ResultType, ColMajor, ColMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    internal::sparse_sparse_to_dense_product_impl<Lhs, Rhs, ResultType>(lhs, rhs, res);
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct sparse_sparse_to_dense_product_selector<Lhs, Rhs, ResultType, RowMajor, ColMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using ColMajorLhs = WithStorageOrder<Lhs, ColMajor>;
    ColMajorLhs lhsCol(lhs);
    internal::sparse_sparse_to_dense_product_impl<ColMajorLhs, Rhs, ResultType>(lhsCol, rhs, res);
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct sparse_sparse_to_dense_product_selector<Lhs, Rhs, ResultType, ColMajor, RowMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    using ColMajorRhs = WithStorageOrder<Rhs, ColMajor>;
    ColMajorRhs rhsCol(rhs);
    internal::sparse_sparse_to_dense_product_impl<Lhs, ColMajorRhs, ResultType>(lhs, rhsCol, res);
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct sparse_sparse_to_dense_product_selector<Lhs, Rhs, ResultType, RowMajor, RowMajor> {
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res) {
    Transpose<ResultType> trRes(res);
    internal::sparse_sparse_to_dense_product_impl<Rhs, Lhs, Transpose<ResultType>, true>(rhs, lhs, trRes);
  }
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_CONSERVATIVESPARSESPARSEPRODUCT_H

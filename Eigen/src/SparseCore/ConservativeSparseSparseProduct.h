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

// Computes res = lhs * rhs without the entries that keep rejects. The inner vectors of res are not sorted.
template <typename Lhs, typename Rhs, typename ResultType, bool ReverseProduct = false,
          typename Keep = sparse_product_keep_all>
static void conservative_sparse_sparse_product_impl(const Lhs& lhs, const Rhs& rhs, ResultType& res,
                                                    const Keep& keep = Keep()) {
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
    for (Index k = 0; k < nnz; ++k) {
      Index i = indices[k];
      if (keep(values[i])) res.insertBackByOuterInnerUnordered(j, i) = values[i];
      mask[i] = false;
    }
  }
  res.finalize();
}

// Computes res = lhs * rhs with sorted inner vectors and no transpose, without the entries that keep rejects. Each
// vector of res is built in res's own free storage, in one of three ways:
//
// - Marked: a marker of the inner dimension locates each index in the vector being built, so a coefficient costs one
//   lookup however many products it sums; the distinct entries are sorted afterwards (sort_inner_vector), which is
//   free when they arrive in order. This is the faster way for short vectors and for repeated indices.
// - Direct: a direct_accumulator, which holds index i at slot i - lo, sums the products, with no work
//   proportional to the inner dimension. This is the faster way for long vectors that are dense in their range and
//   whose indices rarely repeat.
// - Appended: every product is written to the bucket of its index, in the order of arrival, and each bucket is then
//   sorted and its equal indices summed (merge_buckets). The buckets split [lo, hi] evenly, so they follow one
//   another in the result and a bucket's work is local to it. This is the faster way for long vectors that are
//   sparse in their range.
//
// A vector is marked unless the previous one had at least kUnmarkedMin distinct indices, with at most two products
// each; the first one is not marked. The marker is allocated when the first vector is marked.
//
// keep sees the finished sum of each index, so a vector is pruned once it is complete (prune_inner_vector), before
// the next one starts behind it. sparse_product_keep_all compiles the pass out.
//
// The storage of res grows by doubling and is not shrunk at the end, so res can hold more capacity than entries.
template <typename Lhs, typename Rhs, typename ResultType, bool ReverseProduct = false,
          typename Keep = sparse_product_keep_all>
static void conservative_sparse_sparse_product_sorted_impl(const Lhs& lhs, const Rhs& rhs, ResultType& res,
                                                           const Keep& keep = Keep()) {
  using LhsScalar = typename remove_all_t<Lhs>::Scalar;
  using RhsScalar = typename remove_all_t<Rhs>::Scalar;
  using ResScalar = typename remove_all_t<ResultType>::Scalar;
  using StorageIndex = typename remove_all_t<ResultType>::StorageIndex;
  using LhsIterator = typename evaluator<Lhs>::InnerIterator;
  using RhsIterator = typename evaluator<Rhs>::InnerIterator;
  const sparse_scalar_product_op<LhsScalar, RhsScalar, ReverseProduct> product;
  // A vector with fewer distinct indices than this is marked; setting up the other two ways costs more.
  constexpr Index kUnmarkedMin = 48;

  // make sure to call innerSize/outerSize since we fake the storage order.
  Index rows = lhs.innerSize();
  Index cols = rhs.outerSize();
  Index mid = lhs.outerSize();
  eigen_assert(mid == rhs.innerSize());

  evaluator<Lhs> lhsEval(lhs);
  evaluator<Rhs> rhsEval(rhs);

  // The size and the index range of each lhs vector bound those of each result vector. A product with few rhs
  // entries must not cost a pass over the whole lhs, so they are found when a vector is first used, unless the rhs
  // has an entry per lhs vector or more: then one pass in storage order is faster than first uses in rhs order.
  ei_declare_aligned_stack_constructed_variable(Index, lhsCount, mid, 0);
  ei_declare_aligned_stack_constructed_variable(StorageIndex, lhsLo, mid, 0);
  ei_declare_aligned_stack_constructed_variable(StorageIndex, lhsHi, mid, 0);
  const bool scanAll = rhsEval.nonZerosEstimate() >= mid;
  ei_declare_aligned_stack_constructed_variable(bool, lhsKnown, scanAll ? 0 : mid, 0);
  if (!scanAll) std::fill_n(lhsKnown, mid, false);
  auto scan_lhs = [&](Index k) {
    Index count = 0;
    StorageIndex lo = NumTraits<StorageIndex>::highest(), hi = 0;
    for (LhsIterator lhsIt(lhsEval, k); lhsIt; ++lhsIt, ++count) {
      lo = numext::mini(lo, StorageIndex(lhsIt.index()));
      hi = numext::maxi(hi, StorageIndex(lhsIt.index()));
    }
    lhsCount[k] = count;
    lhsLo[k] = lo;
    lhsHi[k] = hi;
  };
  if (scanAll)
    for (Index k = 0; k < mid; ++k) scan_lhs(k);
  auto lhs_count = [&](Index k) {
    if (!scanAll && !lhsKnown[k]) {
      scan_lhs(k);
      lhsKnown[k] = true;
    }
    return lhsCount[k];
  };

  res.setZero();
  res.data().reserve(lhsEval.nonZerosEstimate() + rhsEval.nonZerosEstimate());
  StorageIndex* outer = res.outerIndexPtr();
  // marker[i] >= start locates inner index i in the vector being built, which begins at start.
  Matrix<StorageIndex, Dynamic, 1> marker;
  using Workspace = inner_sort_workspace<ResScalar, StorageIndex>;
  Workspace workspace;
  direct_accumulator<ResScalar, StorageIndex>& acc = workspace.acc;
  Index nnz = 0;
  bool marked = false;
  for (Index j = 0; j < cols; ++j) {
    Index start = nnz;
    outer[j] = convert_index<StorageIndex>(start);
    Index maxKeys = 0;
    for (RhsIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt) maxKeys += lhs_count(rhsIt.index());
    if (maxKeys == 0) continue;

    Index keys;
    StorageIndex lo = NumTraits<StorageIndex>::highest(), hi = 0;
    if (!marked) {
      for (RhsIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt) {
        Index k = rhsIt.index();
        if (lhsCount[k] == 0) continue;
        lo = numext::mini(lo, lhsLo[k]);
        hi = numext::maxi(hi, lhsHi[k]);
      }
      if (acc.direct(lo, hi, maxKeys)) {
        Index size = acc.size(lo, hi);
        if (res.data().allocatedSize() < nnz + size) {
          res.data().resize(nnz);
          res.data().reserve(numext::maxi(size, nnz));
        }
        acc.init(res.data().indexPtr() + nnz, res.data().valuePtr() + nnz, lo, hi);
        for (RhsIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt) {
          RhsScalar y = rhsIt.value();
          acc.accumulate(LhsIterator(lhsEval, rhsIt.index()),
                         [&](ResScalar& value, bool inserted, const LhsIterator& lhsIt) {
                           if (inserted)
                             value = product(lhsIt.value(), y);
                           else
                             product.accumulate(value, lhsIt.value(), y);
                         });
        }
        nnz += acc.compact();
      } else {
        monotone_hash<StorageIndex> hash = bucket_hash<ResScalar>(lo, hi, maxKeys);
        Index buckets = hash.range();
        // cursor[b + 1] counts the products of bucket b; cursor[b] then becomes where its next product goes.
        Matrix<Index, Dynamic, 1>& cursor = workspace.cursor;
        if (cursor.size() < buckets + 1) cursor.resize(buckets + 1);
        cursor.head(buckets + 1).setZero();
        for (RhsIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt)
          for (LhsIterator lhsIt(lhsEval, rhsIt.index()); lhsIt; ++lhsIt)
            ++cursor[hash(StorageIndex(lhsIt.index())) + 1];
        for (Index b = 1; b < buckets; ++b) cursor[b + 1] += cursor[b];
        Workspace::reserve(workspace.buckets, maxKeys);
        Workspace::reserve(workspace.split, maxKeys);
        StorageIndex* bucketIndices = workspace.buckets.indexPtr();
        ResScalar* bucketValues = workspace.buckets.valuePtr();
        for (RhsIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt) {
          RhsScalar y = rhsIt.value();
          for (LhsIterator lhsIt(lhsEval, rhsIt.index()); lhsIt; ++lhsIt) {
            StorageIndex i = StorageIndex(lhsIt.index());
            Index position = cursor[hash(i)]++;
            bucketIndices[position] = i;
            bucketValues[position] = product(lhsIt.value(), y);
          }
        }
        if (res.data().allocatedSize() < nnz + maxKeys) {
          res.data().resize(nnz);
          res.data().reserve(numext::maxi(maxKeys, nnz));
        }
        nnz = merge_buckets(bucketIndices, bucketValues, workspace.split.indexPtr(), workspace.split.valuePtr(),
                            workspace, 0, buckets, res.data().indexPtr(), res.data().valuePtr(), nnz);
      }
      keys = nnz - start;
      nnz = start + prune_inner_vector(res.data().indexPtr() + start, res.data().valuePtr() + start, keys, keep);
    } else {
      if (marker.size() == 0) marker.setConstant(rows, StorageIndex(-1));
      Index bound = numext::mini(maxKeys, rows);
      if (res.data().allocatedSize() < nnz + bound) {
        res.data().resize(nnz);
        res.data().reserve(numext::maxi(bound, nnz));
      }
      StorageIndex* indices = res.data().indexPtr();
      ResScalar* values = res.data().valuePtr();
      Index descents = 0;
      StorageIndex prev = 0;
      for (RhsIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt) {
        RhsScalar y = rhsIt.value();
        for (LhsIterator lhsIt(lhsEval, rhsIt.index()); lhsIt; ++lhsIt) {
          StorageIndex i = StorageIndex(lhsIt.index());
          Index position = marker[i];
          if (position >= start) {
            product.accumulate(values[position], lhsIt.value(), y);
          } else {
            descents += i < prev;
            prev = i;
            lo = numext::mini(lo, i);
            hi = numext::maxi(hi, i);
            marker[i] = convert_index<StorageIndex>(nnz);
            indices[nnz] = i;
            values[nnz] = product(lhsIt.value(), y);
            ++nnz;
          }
        }
      }
      keys = nnz - start;
      // A removed or moved entry leaves a marker at or past the new end of the vector, which the next vector would
      // take for its own. Removing entries adds no descent, and [lo, hi] still bounds the rest.
      nnz = start + prune_inner_vector(indices + start, values + start, keys, keep, marker.data());
      sort_inner_vector(indices + start, values + start, nnz - start, lo, hi, descents, workspace);
    }
    marked = keys < kUnmarkedMin || 2 * keys < maxKeys;
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

// The sorted kernel yields sorted vectors in the storage order of its operands, so each selector below converts at
// most one operand. When both operands have the storage order the result lacks, the unsorted kernel runs and the
// storage-order change of the result sorts it.
template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, ColMajor, ColMajor, ColMajor> {
  template <typename Keep = sparse_product_keep_all>
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res, const Keep& keep = Keep()) {
    using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
    ColMajorRes resCol(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_sorted_impl<Lhs, Rhs, ColMajorRes>(lhs, rhs, resCol, keep);
    res = resCol.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, RowMajor, ColMajor, ColMajor> {
  template <typename Keep = sparse_product_keep_all>
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res, const Keep& keep = Keep()) {
    using ColMajorLhs = WithStorageOrder<Lhs, ColMajor>;
    ColMajorLhs lhsCol = lhs;
    using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
    ColMajorRes resCol(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_sorted_impl<ColMajorLhs, Rhs, ColMajorRes>(lhsCol, rhs, resCol, keep);
    res = resCol.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, ColMajor, RowMajor, ColMajor> {
  template <typename Keep = sparse_product_keep_all>
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res, const Keep& keep = Keep()) {
    using ColMajorRhs = WithStorageOrder<Rhs, ColMajor>;
    ColMajorRhs rhsCol = rhs;
    using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
    ColMajorRes resCol(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_sorted_impl<Lhs, ColMajorRhs, ColMajorRes>(lhs, rhsCol, resCol, keep);
    res = resCol.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, RowMajor, RowMajor, ColMajor> {
  template <typename Keep = sparse_product_keep_all>
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res, const Keep& keep = Keep()) {
    using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
    RowMajorRes resRow(lhs.rows(), rhs.cols());
    // The storage-order change sorts the result, so the product can stay unsorted.
    internal::conservative_sparse_sparse_product_impl<Rhs, Lhs, RowMajorRes, true>(rhs, lhs, resRow, keep);
    res = resRow;
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, ColMajor, ColMajor, RowMajor> {
  template <typename Keep = sparse_product_keep_all>
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res, const Keep& keep = Keep()) {
    using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
    ColMajorRes resCol(lhs.rows(), rhs.cols());
    // The storage-order change sorts the result, so the product can stay unsorted.
    internal::conservative_sparse_sparse_product_impl<Lhs, Rhs, ColMajorRes>(lhs, rhs, resCol, keep);
    res = resCol;
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, RowMajor, ColMajor, RowMajor> {
  template <typename Keep = sparse_product_keep_all>
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res, const Keep& keep = Keep()) {
    using RowMajorRhs = WithStorageOrder<Rhs, RowMajor>;
    RowMajorRhs rhsRow = rhs;
    using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
    RowMajorRes resRow(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_sorted_impl<RowMajorRhs, Lhs, RowMajorRes, true>(rhsRow, lhs, resRow,
                                                                                                  keep);
    res = resRow.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, ColMajor, RowMajor, RowMajor> {
  template <typename Keep = sparse_product_keep_all>
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res, const Keep& keep = Keep()) {
    using RowMajorLhs = WithStorageOrder<Lhs, RowMajor>;
    RowMajorLhs lhsRow = lhs;
    using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
    RowMajorRes resRow(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_sorted_impl<Rhs, RowMajorLhs, RowMajorRes, true>(rhs, lhsRow, resRow,
                                                                                                  keep);
    res = resRow.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, RowMajor, RowMajor, RowMajor> {
  template <typename Keep = sparse_product_keep_all>
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res, const Keep& keep = Keep()) {
    using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
    RowMajorRes resRow(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_sorted_impl<Rhs, Lhs, RowMajorRes, true>(rhs, lhs, resRow, keep);
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

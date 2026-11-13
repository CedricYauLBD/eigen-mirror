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

// The terms lhs(:, k) * rhs(k, j) of vector j of a product, for sparse_accumulator. lo and hi hold the index range of
// each lhs vector that rhs(:, j) refers to. The members are always inlined: left to itself, GCC 14 called the
// constructor of Term once per term in a unit with several products, which cost 15-45% on vectors of one or two terms.
template <typename Lhs, typename Rhs, typename ResScalar, typename StorageIndex, bool ReverseProduct>
class sparse_product_terms {
  using LhsEvaluator = evaluator<Lhs>;
  using RhsEvaluator = evaluator<Rhs>;
  using LhsIterator = typename LhsEvaluator::InnerIterator;
  using RhsIterator = typename RhsEvaluator::InnerIterator;
  using LhsScalar = typename remove_all_t<Lhs>::Scalar;
  using RhsScalar = typename remove_all_t<Rhs>::Scalar;
  using Product = sparse_scalar_product_op<LhsScalar, RhsScalar, ReverseProduct>;

 public:
  class Term {
   public:
    EIGEN_ALWAYS_INLINE Term(const LhsEvaluator& lhsEval, const RhsIterator& rhsIt)
        : m_it(lhsEval, rhsIt.index()), m_y(rhsIt.value()) {}
    EIGEN_ALWAYS_INLINE explicit operator bool() const { return bool(m_it); }
    EIGEN_ALWAYS_INLINE Term& operator++() {
      ++m_it;
      return *this;
    }
    EIGEN_ALWAYS_INLINE StorageIndex index() const { return StorageIndex(m_it.index()); }
    EIGEN_ALWAYS_INLINE void assignTo(ResScalar& value) const { value = Product()(m_it.value(), m_y); }
    EIGEN_ALWAYS_INLINE void addTo(ResScalar& value) const { Product().accumulate(value, m_it.value(), m_y); }

   private:
    LhsIterator m_it;
    RhsScalar m_y;
  };

  sparse_product_terms(const LhsEvaluator& lhsEval, const RhsEvaluator& rhsEval, Index j, Index terms, Index entries,
                       const StorageIndex* lo, const StorageIndex* hi)
      : m_lhsEval(lhsEval), m_rhsEval(rhsEval), m_j(j), m_terms(terms), m_entries(entries), m_lo(lo), m_hi(hi) {}

  Index count() const { return m_terms; }
  Index maxEntries() const { return m_entries; }
  void range(StorageIndex& lo, StorageIndex& hi) const {
    for (RhsIterator rhsIt(m_rhsEval, m_j); rhsIt; ++rhsIt) {
      Index k = rhsIt.index();
      // An empty lhs vector has lo = highest() and hi = 0, which change nothing.
      lo = numext::mini(lo, m_lo[k]);
      hi = numext::maxi(hi, m_hi[k]);
    }
  }
  EIGEN_ALWAYS_INLINE Term term(Index t) const {
    RhsIterator rhsIt(m_rhsEval, m_j);
    for (; t > 0; --t) ++rhsIt;
    return Term(m_lhsEval, rhsIt);
  }
  template <typename Func>
  EIGEN_ALWAYS_INLINE void forEachTerm(const Func& func) const {
    for (RhsIterator rhsIt(m_rhsEval, m_j); rhsIt; ++rhsIt) func(Term(m_lhsEval, rhsIt));
  }

 private:
  const LhsEvaluator& m_lhsEval;
  const RhsEvaluator& m_rhsEval;
  Index m_j, m_terms, m_entries;
  const StorageIndex* m_lo;
  const StorageIndex* m_hi;
};

// Computes res = lhs * rhs with sorted inner vectors and no transpose, without the entries that keep rejects. A
// sparse_accumulator builds each vector of res in res's own storage, which can keep spare capacity.
template <typename Lhs, typename Rhs, typename ResultType, bool ReverseProduct = false,
          typename Keep = sparse_product_keep_all>
static void conservative_sparse_sparse_product_sorted_impl(const Lhs& lhs, const Rhs& rhs, ResultType& res,
                                                           const Keep& keep = Keep()) {
  using ResScalar = typename remove_all_t<ResultType>::Scalar;
  using StorageIndex = typename remove_all_t<ResultType>::StorageIndex;
  using LhsIterator = typename evaluator<Lhs>::InnerIterator;
  using RhsIterator = typename evaluator<Rhs>::InnerIterator;
  using Terms = sparse_product_terms<Lhs, Rhs, ResScalar, StorageIndex, ReverseProduct>;

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
  sparse_accumulator<ResScalar, StorageIndex> acc(rows, res.data());
  for (Index j = 0; j < cols; ++j) {
    outer[j] = convert_index<StorageIndex>(acc.size());
    Index entries = 0, terms = 0;
    for (RhsIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt, ++terms) entries += lhs_count(rhsIt.index());
    acc.append(Terms(lhsEval, rhsEval, j, terms, entries, lhsLo, lhsHi), keep);
  }
  outer[cols] = convert_index<StorageIndex>(acc.size());
  acc.finish();
}

// Helper template to generate new sparse matrix types
template <class Source, int Order>
using WithStorageOrder = SparseMatrix<typename Source::Scalar, Order, typename Source::StorageIndex>;

template <typename Lhs, typename Rhs, typename ResultType,
          int LhsStorageOrder = (traits<Lhs>::Flags & RowMajorBit) ? RowMajor : ColMajor,
          int RhsStorageOrder = (traits<Rhs>::Flags & RowMajorBit) ? RowMajor : ColMajor,
          int ResStorageOrder = (traits<ResultType>::Flags & RowMajorBit) ? RowMajor : ColMajor>
struct conservative_sparse_sparse_product_selector;

// The kernel yields sorted vectors in the storage order of its operands, so each selector below converts the operands
// that lack the storage order of the result. Converting both costs a pass over the operands; multiplying first and
// converting the result costs a pass over the result, which usually has more entries.
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
    using ColMajorLhs = WithStorageOrder<Lhs, ColMajor>;
    ColMajorLhs lhsCol = lhs;
    using ColMajorRhs = WithStorageOrder<Rhs, ColMajor>;
    ColMajorRhs rhsCol = rhs;
    using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
    ColMajorRes resCol(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_sorted_impl<ColMajorLhs, ColMajorRhs, ColMajorRes>(lhsCol, rhsCol,
                                                                                                    resCol, keep);
    res = resCol.markAsRValue();
  }
};

template <typename Lhs, typename Rhs, typename ResultType>
struct conservative_sparse_sparse_product_selector<Lhs, Rhs, ResultType, ColMajor, ColMajor, RowMajor> {
  template <typename Keep = sparse_product_keep_all>
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res, const Keep& keep = Keep()) {
    using RowMajorLhs = WithStorageOrder<Lhs, RowMajor>;
    RowMajorLhs lhsRow = lhs;
    using RowMajorRhs = WithStorageOrder<Rhs, RowMajor>;
    RowMajorRhs rhsRow = rhs;
    using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
    RowMajorRes resRow(lhs.rows(), rhs.cols());
    internal::conservative_sparse_sparse_product_sorted_impl<RowMajorRhs, RowMajorLhs, RowMajorRes, true>(
        rhsRow, lhsRow, resRow, keep);
    res = resRow.markAsRValue();
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

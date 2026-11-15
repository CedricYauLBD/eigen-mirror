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

// The outer index and inner sizes of an expression with compressed access, null without it. A SparseVector has
// compressed access and no outer index: vectorSize() is then the size of its one inner vector, and -1 otherwise.
template <typename Xpr, bool = has_compressed_access<Xpr>::value>
struct sparse_outer_pointers {
  using StorageIndex = typename Xpr::StorageIndex;
  static const StorageIndex* outer(const Xpr&) { return nullptr; }
  static const StorageIndex* innerNonZeros(const Xpr&) { return nullptr; }
  static Index vectorSize(const Xpr&) { return -1; }
};
template <typename Xpr>
struct sparse_outer_pointers<Xpr, true> {
  using StorageIndex = typename Xpr::StorageIndex;
  static const StorageIndex* outer(const Xpr& xpr) { return xpr.outerIndexPtr(); }
  static const StorageIndex* innerNonZeros(const Xpr& xpr) { return xpr.innerNonZeroPtr(); }
  static Index vectorSize(const Xpr& xpr) {
    return xpr.outerIndexPtr() == nullptr && xpr.outerSize() == 1 ? Index(xpr.nonZeros()) : -1;
  }
};

// The size and the index range of each inner vector of the lhs of a product, in arrays the caller owns. A size is
// read from the outer index, or from a SparseVector, without a scan; only the direct and appended vectors ask for a
// range. A vector is scanned when first needed, so that few rhs entries do not cost a pass over the lhs, unless
// scanAll: the first need then scans all of them in storage order, which beats first uses in rhs order.
template <typename Lhs, typename StorageIndex>
class sparse_product_lhs_stats {
  using LhsIterator = typename evaluator<Lhs>::InnerIterator;
  using Pointers = sparse_outer_pointers<Lhs>;

 public:
  // known has outerSize entries, or none if scanAll.
  sparse_product_lhs_stats(const Lhs& lhs, const evaluator<Lhs>& lhsEval, bool scanAll, Index* count, StorageIndex* lo,
                           StorageIndex* hi, bool* known)
      : m_lhsEval(lhsEval),
        m_outer(Pointers::outer(lhs)),
        m_innerNonZeros(Pointers::innerNonZeros(lhs)),
        m_vectorSize(Pointers::vectorSize(lhs)),
        m_outerSize(lhs.outerSize()),
        m_scanAll(scanAll),
        m_count(count),
        m_lo(lo),
        m_hi(hi),
        m_known(known) {
    if (!scanAll) std::fill_n(known, m_outerSize, false);
  }

  EIGEN_ALWAYS_INLINE Index count(Index k) {
    if (m_outer != nullptr)
      return m_innerNonZeros != nullptr ? Index(m_innerNonZeros[k]) : Index(m_outer[k + 1]) - Index(m_outer[k]);
    if (m_vectorSize >= 0) return m_vectorSize;
    scan(k);
    return m_count[k];
  }

  // Widens [lo, hi] to hold the indices of vector k.
  EIGEN_ALWAYS_INLINE void range(Index k, StorageIndex& lo, StorageIndex& hi) {
    scan(k);
    // An empty vector has lo = highest() and hi = 0, which change nothing.
    lo = numext::mini(lo, m_lo[k]);
    hi = numext::maxi(hi, m_hi[k]);
  }

 private:
  EIGEN_ALWAYS_INLINE void scan(Index k) {
    if (m_allKnown || (!m_scanAll && m_known[k])) return;
    scanMissing(k);
  }
  EIGEN_DONT_INLINE void scanMissing(Index k) {
    Index begin = m_scanAll ? 0 : k, end = m_scanAll ? m_outerSize : k + 1;
    for (Index v = begin; v < end; ++v) {
      Index count = 0;
      StorageIndex lo = NumTraits<StorageIndex>::highest(), hi = 0;
      for (LhsIterator lhsIt(m_lhsEval, v); lhsIt; ++lhsIt, ++count) {
        lo = numext::mini(lo, StorageIndex(lhsIt.index()));
        hi = numext::maxi(hi, StorageIndex(lhsIt.index()));
      }
      m_count[v] = count;
      m_lo[v] = lo;
      m_hi[v] = hi;
    }
    if (m_scanAll)
      m_allKnown = true;
    else
      m_known[k] = true;
  }

  const evaluator<Lhs>& m_lhsEval;
  const typename Pointers::StorageIndex* m_outer;
  const typename Pointers::StorageIndex* m_innerNonZeros;
  Index m_vectorSize;
  Index m_outerSize;
  bool m_scanAll;
  bool m_allKnown = false;
  Index* m_count;
  StorageIndex* m_lo;
  StorageIndex* m_hi;
  bool* m_known;
};

// The terms lhs(:, k) * rhs(k, j) of vector j of a product, for sparse_accumulator. The members are always inlined:
// left to itself, GCC 14 called the constructor of Term once per term in a unit with several products, which cost
// 15-45% on vectors of one or two terms.
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

  using LhsStats = sparse_product_lhs_stats<Lhs, StorageIndex>;

  sparse_product_terms(const LhsEvaluator& lhsEval, const RhsEvaluator& rhsEval, Index j, Index terms, Index entries,
                       LhsStats& lhsStats)
      : m_lhsEval(lhsEval), m_rhsEval(rhsEval), m_j(j), m_terms(terms), m_entries(entries), m_lhsStats(lhsStats) {}

  Index count() const { return m_terms; }
  Index maxEntries() const { return m_entries; }
  void range(StorageIndex& lo, StorageIndex& hi) const {
    for (RhsIterator rhsIt(m_rhsEval, m_j); rhsIt; ++rhsIt) m_lhsStats.range(rhsIt.index(), lo, hi);
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
  LhsStats& m_lhsStats;
};

// Computes res = lhs * rhs with sorted inner vectors and no transpose, without the entries that keep rejects. A
// sparse_accumulator builds each vector of res in res's own storage, which can keep spare capacity.
template <typename Lhs, typename Rhs, typename ResultType, bool ReverseProduct = false,
          typename Keep = sparse_product_keep_all>
static void conservative_sparse_sparse_product_sorted_impl(const Lhs& lhs, const Rhs& rhs, ResultType& res,
                                                           const Keep& keep = Keep()) {
  using ResScalar = typename remove_all_t<ResultType>::Scalar;
  using StorageIndex = typename remove_all_t<ResultType>::StorageIndex;
  using RhsIterator = typename evaluator<Rhs>::InnerIterator;
  using Terms = sparse_product_terms<Lhs, Rhs, ResScalar, StorageIndex, ReverseProduct>;

  // make sure to call innerSize/outerSize since we fake the storage order.
  Index rows = lhs.innerSize();
  Index cols = rhs.outerSize();
  Index mid = lhs.outerSize();
  eigen_assert(mid == rhs.innerSize());

  evaluator<Lhs> lhsEval(lhs);
  evaluator<Rhs> rhsEval(rhs);

  ei_declare_aligned_stack_constructed_variable(Index, lhsCount, mid, 0);
  ei_declare_aligned_stack_constructed_variable(StorageIndex, lhsLo, mid, 0);
  ei_declare_aligned_stack_constructed_variable(StorageIndex, lhsHi, mid, 0);
  // With an rhs entry per lhs vector or more, a need for the lhs statistics is a need for all of them.
  const bool scanAll = rhsEval.nonZerosEstimate() >= mid;
  ei_declare_aligned_stack_constructed_variable(bool, lhsKnown, scanAll ? 0 : mid, 0);
  sparse_product_lhs_stats<Lhs, StorageIndex> lhsStats(lhs, lhsEval, scanAll, lhsCount, lhsLo, lhsHi, lhsKnown);

  res.setZero();
  res.data().reserve(lhsEval.nonZerosEstimate() + rhsEval.nonZerosEstimate());
  StorageIndex* outer = res.outerIndexPtr();
  sparse_accumulator<ResScalar, StorageIndex> acc(rows, res.data());
  for (Index j = 0; j < cols; ++j) {
    outer[j] = convert_index<StorageIndex>(acc.size());
    Index entries = 0, terms = 0;
    for (RhsIterator rhsIt(rhsEval, j); rhsIt; ++rhsIt, ++terms) entries += lhsStats.count(rhsIt.index());
    acc.append(Terms(lhsEval, rhsEval, j, terms, entries, lhsStats), keep);
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
// that lack the storage order of the result. Where both lack it, the selector can instead run the kernel in the
// storage order of the operands and convert the result: a change of storage order is a counting transpose, which
// leaves the vectors sorted.
//
// For the kernel run on (K, T), which sums one term per entry (k, j) of T, the term being vector k of K, converting
// the result beat converting the operands when
//
//   nnz(K) > 2 nnz(T)   and   flops < 2 (nnz(K) + nnz(T)),   flops = sum over entries (k, j) of T of nnz(K_k).
//
// The kernel in the storage order of the result has nnz(K) terms, and a term has a fixed cost, so fewer terms are
// cheaper. nnz(C) <= flops, and converting C costs about as much per entry as converting K and T, so the second
// condition bounds the conversion. The factor 2 in the first keeps products of similar operands, such as A * A, on the
// conversion of the operands. flops is counted only up to its bound. The rule cannot see how scattered the vectors of
// C are, which decides the cost of converting C on a machine with a small cache.
template <typename K, typename T>
bool sparse_product_converts_result(const K& kernelLhs, const T& kernelRhs) {
  using StorageIndex = typename K::StorageIndex;
  using TIterator = typename evaluator<T>::InnerIterator;
  evaluator<K> kEval(kernelLhs);
  evaluator<T> tEval(kernelRhs);
  const Index kNonZeros = kEval.nonZerosEstimate(), tNonZeros = tEval.nonZerosEstimate();
  if (!(kNonZeros > 2 * tNonZeros)) return false;
  const Index outer = kernelLhs.outerSize();
  ei_declare_aligned_stack_constructed_variable(Index, count, outer, 0);
  ei_declare_aligned_stack_constructed_variable(StorageIndex, lo, outer, 0);
  ei_declare_aligned_stack_constructed_variable(StorageIndex, hi, outer, 0);
  ei_declare_aligned_stack_constructed_variable(bool, known, outer, 0);
  sparse_product_lhs_stats<K, StorageIndex> kStats(kernelLhs, kEval, false, count, lo, hi, known);
  const Index bound = 2 * (kNonZeros + tNonZeros);
  Index flops = 0;
  for (Index j = 0; j < kernelRhs.outerSize(); ++j)
    for (TIterator tIt(tEval, j); tIt; ++tIt) {
      flops += kStats.count(tIt.index());
      if (flops >= bound) return false;
    }
  return true;
}

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
    if (sparse_product_converts_result(rhs, lhs)) {
      using RowMajorRes = WithStorageOrder<ResultType, RowMajor>;
      RowMajorRes resRow(lhs.rows(), rhs.cols());
      internal::conservative_sparse_sparse_product_sorted_impl<Rhs, Lhs, RowMajorRes, true>(rhs, lhs, resRow, keep);
      res = resRow;
      return;
    }
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
    if (sparse_product_converts_result(lhs, rhs)) {
      using ColMajorRes = WithStorageOrder<ResultType, ColMajor>;
      ColMajorRes resCol(lhs.rows(), rhs.cols());
      internal::conservative_sparse_sparse_product_sorted_impl<Lhs, Rhs, ColMajorRes>(lhs, rhs, resCol, keep);
      res = resCol;
      return;
    }
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

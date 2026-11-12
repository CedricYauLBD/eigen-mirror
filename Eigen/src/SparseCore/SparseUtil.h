// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSEUTIL_H
#define EIGEN_SPARSEUTIL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

#ifdef NDEBUG
#define EIGEN_DBG_SPARSE(X)
#else
#define EIGEN_DBG_SPARSE(X) X
#endif

#define EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(Derived, Op)                                    \
  template <typename OtherDerived>                                                               \
  EIGEN_STRONG_INLINE Derived& operator Op(const Eigen::SparseMatrixBase<OtherDerived>& other) { \
    return Base::operator Op(other.derived());                                                   \
  }                                                                                              \
  EIGEN_STRONG_INLINE Derived& operator Op(const Derived & other) { return Base::operator Op(other); }

#define EIGEN_SPARSE_INHERIT_SCALAR_ASSIGNMENT_OPERATOR(Derived, Op) \
  template <typename Other>                                          \
  EIGEN_STRONG_INLINE Derived& operator Op(const Other & scalar) {   \
    return Base::operator Op(scalar);                                \
  }

#define EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATORS(Derived) EIGEN_SPARSE_INHERIT_ASSIGNMENT_OPERATOR(Derived, =)

#define EIGEN_SPARSE_PUBLIC_INTERFACE(Derived) EIGEN_GENERIC_PUBLIC_INTERFACE(Derived)

const int CoherentAccessPattern = 0x1;
const int InnerRandomAccessPattern = 0x2 | CoherentAccessPattern;
const int OuterRandomAccessPattern = 0x4 | CoherentAccessPattern;
const int RandomAccessPattern = 0x8 | OuterRandomAccessPattern | InnerRandomAccessPattern;

template <typename Scalar_, int Flags_ = 0, typename StorageIndex_ = int>
class SparseMatrix;
template <typename Scalar_, int Flags_ = 0, typename StorageIndex_ = int>
class SparseVector;

template <typename MatrixType, unsigned int UpLo>
class SparseSelfAdjointView;
template <typename MatrixType>
class SparseView;

template <typename MatrixType, int UpLo>
class SparseSymmetricPermutationProduct;

namespace internal {

// Retain the existing evaluation order for products declared commutative.
template <typename LhsScalar, typename RhsScalar, bool Reverse,
          bool Commutative = functor_is_commutative<scalar_product_op<LhsScalar, RhsScalar>>::value>
struct sparse_scalar_product_op {
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE auto operator()(const LhsScalar& lhs, const RhsScalar& rhs) const {
    return lhs * rhs;
  }
  // Keep multiply and addition in one expression for -ffp-contract=on.
  template <typename Acc>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void accumulate(Acc& acc, const LhsScalar& lhs, const RhsScalar& rhs) const {
    acc += lhs * rhs;
  }
};

template <typename LhsScalar, typename RhsScalar>
struct sparse_scalar_product_op<LhsScalar, RhsScalar, true, false> {
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE auto operator()(const LhsScalar& lhs, const RhsScalar& rhs) const {
    return rhs * lhs;
  }
  template <typename Acc>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void accumulate(Acc& acc, const LhsScalar& lhs, const RhsScalar& rhs) const {
    acc += rhs * lhs;
  }
};

/** \internal
 * Compile-time test for whether a sparse expression exposes its storage directly
 * through outerIndexPtr() / innerIndexPtr() / valuePtr() / innerNonZeroPtr()
 * (the \ref CompressedAccessBit contract: SparseMatrix, SparseVector,
 * Ref<Sparse>, Map<SparseMatrix>, ...). Such expressions can drive raw-pointer
 * fast paths; anything else must be walked via InnerIterator.
 */
template <typename Derived>
struct has_compressed_access : bool_constant<(int(traits<Derived>::Flags) & CompressedAccessBit) != 0> {};

// Defined in SparseApproxComparisons.h; BlockSparseMatrix.h uses it before that.
template <typename Scalar, bool = use_scaled_comparison<Scalar>::value>
struct sparse_approx_comparison;

template <typename T, int Rows, int Cols, int Flags>
struct sparse_eval;

template <typename T>
struct eval<T, Sparse> : sparse_eval<T, traits<T>::RowsAtCompileTime, traits<T>::ColsAtCompileTime, traits<T>::Flags> {
};

template <typename T, int Cols, int Flags>
struct sparse_eval<T, 1, Cols, Flags> {
  using Scalar_ = typename traits<T>::Scalar;
  using StorageIndex_ = typename traits<T>::StorageIndex;

 public:
  using type = SparseVector<Scalar_, RowMajor, StorageIndex_>;
};

template <typename T, int Rows, int Flags>
struct sparse_eval<T, Rows, 1, Flags> {
  using Scalar_ = typename traits<T>::Scalar;
  using StorageIndex_ = typename traits<T>::StorageIndex;

 public:
  using type = SparseVector<Scalar_, ColMajor, StorageIndex_>;
};

// TODO: consider unifying with plain_matrix_type<T, Sparse>.
template <typename T, int Rows, int Cols, int Flags>
struct sparse_eval {
  using Scalar_ = typename traits<T>::Scalar;
  using StorageIndex_ = typename traits<T>::StorageIndex;
  enum { Options_ = ((Flags & RowMajorBit) == RowMajorBit) ? RowMajor : ColMajor };

 public:
  using type = SparseMatrix<Scalar_, Options_, StorageIndex_>;
};

template <typename T, int Flags>
struct sparse_eval<T, 1, 1, Flags> {
  using Scalar_ = typename traits<T>::Scalar;

 public:
  using type = Matrix<Scalar_, 1, 1>;
};

template <typename T>
struct plain_matrix_type<T, Sparse> {
  using Scalar_ = typename traits<T>::Scalar;
  using StorageIndex_ = typename traits<T>::StorageIndex;
  enum { Options_ = ((evaluator<T>::Flags & RowMajorBit) == RowMajorBit) ? RowMajor : ColMajor };

 public:
  using type = SparseMatrix<Scalar_, Options_, StorageIndex_>;
};

template <typename T>
struct plain_object_eval<T, Sparse>
    : sparse_eval<T, traits<T>::RowsAtCompileTime, traits<T>::ColsAtCompileTime, evaluator<T>::Flags> {};

template <typename Decomposition, typename RhsType>
struct solve_traits<Decomposition, RhsType, Sparse> {
  using PlainObject = typename sparse_eval<RhsType, RhsType::RowsAtCompileTime, RhsType::ColsAtCompileTime,
                                           traits<RhsType>::Flags>::type;
};

template <typename Derived>
struct generic_xpr_base<Derived, MatrixXpr, Sparse> {
  using type = SparseMatrixBase<Derived>;
};

struct SparseTriangularShape {
  static std::string debugName() { return "SparseTriangularShape"; }
};
struct SparseSelfAdjointShape {
  static std::string debugName() { return "SparseSelfAdjointShape"; }
};

template <>
struct glue_shapes<SparseShape, SelfAdjointShape> {
  using type = SparseSelfAdjointShape;
};
template <>
struct glue_shapes<SparseShape, TriangularShape> {
  using type = SparseTriangularShape;
};

// return type of SparseCompressedBase::lower_bound;
struct LowerBoundIndex {
  LowerBoundIndex() = default;
  LowerBoundIndex(Index val, bool ok) : value(val), found(ok) {}
  Index value = -1;
  bool found = false;
};

// Maps [lo, hi] onto [0, m) monotonically:
//
//   h(i) = floor((i - lo) s / 2^k),   s = floor(m 2^k / span),   span = hi - lo + 1,
//
// k the width of StorageIndex. For m < span, s < 2^k and h(i) <= (i - lo) m / span < m. For m >= span, h(i) = i - lo
// (direct), and the range is [0, span).
template <typename StorageIndex>
class monotone_hash {
  using Unsigned = std::make_unsigned_t<StorageIndex>;

 public:
  monotone_hash() = default;
  monotone_hash(StorageIndex lo, StorageIndex hi, Index m) { compute(lo, hi, m); }

  void compute(StorageIndex lo, StorageIndex hi, Index m) {
    Index span = Index(hi) - Index(lo) + 1;
    m_lo = lo;
    m_direct = m >= span;
    m_range = m_direct ? span : m;
    if (!m_direct) m_scale = int_div::div_shifted(Unsigned(m), Unsigned(span));
  }

  Index operator()(StorageIndex i) const {
    Unsigned x = Unsigned(i - m_lo);
    return m_direct ? Index(x) : Index(int_div::umulhi(x, m_scale));
  }

  // h maps into [0, range()); direct() means h is one-to-one.
  Index range() const { return m_range; }
  bool direct() const { return m_direct; }

 private:
  StorageIndex m_lo = 0;
  Unsigned m_scale = 0;
  Index m_range = 1;
  bool m_direct = true;
};

// Sums (i, v) pairs with lo <= i <= hi in storage the caller owns. Slot i - lo holds index i or nothing, so one scan
// yields the sorted, duplicate-free pattern. The table has a slot for every index of [lo, hi]: it suits a vector
// that is dense in its range, see direct().
//
// Empty slots hold -1. Every other value of StorageIndex can be an index: a matrix may have highest() + 1 rows.
template <typename Scalar, typename StorageIndex>
class direct_accumulator {
  // Used by value only: in C++14 a static constexpr member that is bound to a reference needs a definition.
  static constexpr StorageIndex kEmpty = StorageIndex(-1);
  // compact() tests this many slots at once for being all empty.
  static constexpr Index kBlock = 16;
  // direct() accepts up to this many slots per key: emptying the table is one fill and compact() skips its empty
  // blocks, so unused slots are cheap.
  static constexpr Index kDirectRatio = 16;

 public:
  // Slots needed for keys in [lo, hi].
  static Index size(StorageIndex lo, StorageIndex hi) { return Index(hi) - Index(lo) + 1; }

  // Whether the table is worth its slots for at most maxKeys distinct keys in [lo, hi].
  static bool direct(StorageIndex lo, StorageIndex hi, Index maxKeys) { return size(lo, hi) <= kDirectRatio * maxKeys; }

  // Binds the table to keys[0, size(lo, hi)) and values[0, size(lo, hi)) and empties it.
  void init(StorageIndex* keys, Scalar* values, StorageIndex lo, StorageIndex hi) {
    m_keys = keys;
    m_values = values;
    m_lo = lo;
    m_top = 0;
    std::fill_n(keys, size(lo, hi), StorageIndex(kEmpty));
  }

  // The value of index i. If i is absent, inserts it with an unspecified value and sets inserted.
  Scalar& findOrInsert(StorageIndex i, bool& inserted) {
    Index s = Index(i) - Index(m_lo);
    inserted = m_keys[s] == kEmpty;
    if (inserted) {
      m_keys[s] = i;
      m_top = numext::maxi(m_top, s);
    }
    return m_values[s];
  }

  // Calls func(value, inserted, it) for each entry of the inner iterator it, with the value of index it.index() as
  // findOrInsert returns it. The table state stays in locals over the whole vector; findOrInsert reloads it from the
  // members at each call, since the stores to the table may alias them.
  template <typename InnerIterator, typename Func>
  void accumulate(InnerIterator it, const Func& func) {
    StorageIndex* keys = m_keys;
    Scalar* values = m_values;
    Index top = m_top;
    Index lo = m_lo;
    for (; it; ++it) {
      StorageIndex i = StorageIndex(it.index());
      Index s = Index(i) - lo;
      bool inserted = keys[s] == kEmpty;
      if (inserted) {
        keys[s] = i;
        top = numext::maxi(top, s);
      }
      func(values[s], inserted, it);
    }
    m_top = top;
  }

  // Moves the n entries to keys[0, n) and values[0, n) in increasing index order and returns n. The write slot never
  // passes the read slot, so the table may sit where its output belongs. init() must run before the next use.
  Index compact() {
    // Entries before the first empty slot are already in place; after it, n < s, so no value moves onto itself.
    Index s = 0;
    while (s <= m_top && m_keys[s] != kEmpty) ++s;
    Index n = s;
    // A block of kBlock empty slots is skipped after one test of all of them; its fixed length lets the compiler
    // vectorize the test.
    for (; s + kBlock <= m_top + 1; s += kBlock) {
      StorageIndex occupied = 0;
      for (Index k = 0; k < kBlock; ++k) occupied |= m_keys[s + k] ^ kEmpty;
      if (occupied == 0) continue;
      for (Index k = s; k < s + kBlock; ++k)
        if (m_keys[k] != kEmpty) {
          m_keys[n] = m_keys[k];
          m_values[n] = move_if_assignable(m_values[k]);
          ++n;
        }
    }
    for (; s <= m_top; ++s)
      if (m_keys[s] != kEmpty) {
        m_keys[n] = m_keys[s];
        m_values[n] = move_if_assignable(m_values[s]);
        ++n;
      }
    return n;
  }

 private:
  StorageIndex* m_keys = nullptr;
  Scalar* m_values = nullptr;
  StorageIndex m_lo = 0;
  Index m_top = 0;
};

template <typename Scalar, typename StorageIndex>
class CompressedStorage;

// The scratch storage of sort_inner_vector and merge_buckets, kept between calls.
template <typename Scalar, typename StorageIndex>
struct inner_sort_workspace {
  // The average number of entries per bucket: with one entry per bucket the counters cost as much as the entries,
  // and a long bucket is slow to sort by insertion.
  static constexpr Index kBucketLoad = 4;
  // A bucket with more entries than this holds a cluster and is not sorted by insertion. Splitting it must make at
  // least two buckets, or the split would not end.
  static constexpr Index kBucketMax = 8;
  // Makes room for n entries in storage.
  static void reserve(CompressedStorage<Scalar, StorageIndex>& storage, Index n) {
    if (storage.allocatedSize() < n) storage.reserve(n);
  }
  // The entries in their buckets, and a second array of the same extent that an overfull bucket is split into.
  CompressedStorage<Scalar, StorageIndex> buckets, split;
  // A direct table.
  CompressedStorage<Scalar, StorageIndex> table;
  // cursor[base + b] is the end of bucket b, for the buckets of one split; the next split follows them.
  Matrix<Index, Dynamic, 1> cursor;
  direct_accumulator<Scalar, StorageIndex> acc;
};

// The hash that sends the indices of n entries in [lo, hi] to their buckets.
template <typename Scalar, typename StorageIndex>
monotone_hash<StorageIndex> bucket_hash(StorageIndex lo, StorageIndex hi, Index n) {
  Index buckets = n / inner_sort_workspace<Scalar, StorageIndex>::kBucketLoad;
  return monotone_hash<StorageIndex>(lo, hi, numext::maxi<Index>(1, buckets));
}

// Moves the n entries (srcIndices[k], srcValues[k]), with indices in [lo, hi], to their buckets in (dstIndices,
// dstValues), in the order of arrival within a bucket. Returns the number of buckets and sets cursor[base + b] to the
// end of bucket b; bucket 0 begins at 0.
template <typename Scalar, typename StorageIndex>
Index split_into_buckets(const StorageIndex* srcIndices, Scalar* srcValues, Index n, StorageIndex lo, StorageIndex hi,
                         StorageIndex* dstIndices, Scalar* dstValues, Matrix<Index, Dynamic, 1>& cursor, Index base) {
  monotone_hash<StorageIndex> hash = bucket_hash<Scalar>(lo, hi, n);
  Index buckets = hash.range();
  if (cursor.size() < base + buckets + 1) cursor.conservativeResize(base + buckets + 1);
  // ends[b + 1] counts bucket b, then ends[b] is where its next entry goes.
  Index* ends = cursor.data() + base;
  std::fill_n(ends, buckets + 1, Index(0));
  for (Index k = 0; k < n; ++k) ++ends[hash(srcIndices[k]) + 1];
  for (Index b = 1; b < buckets; ++b) ends[b + 1] += ends[b];
  for (Index k = 0; k < n; ++k) {
    Index position = ends[hash(srcIndices[k])]++;
    dstIndices[position] = srcIndices[k];
    dstValues[position] = move_if_assignable(srcValues[k]);
  }
  return buckets;
}

// Appends the entries of the buckets to (indices, values) from position count, sorted by index and with the values
// of equal indices summed, and returns the new count. Bucket b is [cursor[base + b - 1], cursor[base + b]) of
// (bucketIndices, bucketValues), bucket 0 beginning at 0, and the indices of a bucket are below those of the next.
// (splitIndices, splitValues) has the extent of the buckets; both are overwritten. cursor is workspace.cursor.
//
// A bucket is sorted by insertion. One with more than kBucketMax entries holds a cluster, which a hash over all the
// buckets could not spread. It goes through a direct table if it is dense in its own range, and is otherwise split
// again over that range. A split has at least two buckets, each with about half of the range or less, so n entries
// cost O(n d) for d nested splits, d <= log2(hi - lo); d is 1 unless clusters lie within clusters.
template <typename Scalar, typename StorageIndex>
Index merge_buckets(StorageIndex* bucketIndices, Scalar* bucketValues, StorageIndex* splitIndices, Scalar* splitValues,
                    inner_sort_workspace<Scalar, StorageIndex>& workspace, Index base, Index buckets,
                    StorageIndex* indices, Scalar* values, Index count) {
  using Workspace = inner_sort_workspace<Scalar, StorageIndex>;
  constexpr Index kBucketMax = Workspace::kBucketMax;
  EIGEN_STATIC_ASSERT(Workspace::kBucketMax >= 2 * Workspace::kBucketLoad, AN_OVERFULL_BUCKET_MUST_SPLIT_IN_TWO)
  Index begin = 0;
  for (Index b = 0; b < buckets; ++b) {
    Index end = workspace.cursor[base + b];
    Index n = end - begin;
    if (n > kBucketMax) {
      StorageIndex lo = bucketIndices[begin], hi = bucketIndices[begin];
      for (Index p = begin + 1; p < end; ++p) {
        lo = numext::mini(lo, bucketIndices[p]);
        hi = numext::maxi(hi, bucketIndices[p]);
      }
      direct_accumulator<Scalar, StorageIndex>& acc = workspace.acc;
      if (acc.direct(lo, hi, n)) {
        Workspace::reserve(workspace.table, acc.size(lo, hi));
        acc.init(workspace.table.indexPtr(), workspace.table.valuePtr(), lo, hi);
        for (Index p = begin; p < end; ++p) {
          bool inserted;
          Scalar& value = acc.findOrInsert(bucketIndices[p], inserted);
          if (inserted)
            value = move_if_assignable(bucketValues[p]);
          else
            value += bucketValues[p];
        }
        Index distinct = acc.compact();
        std::copy(workspace.table.indexPtr(), workspace.table.indexPtr() + distinct, indices + count);
        move_or_copy(workspace.table.valuePtr(), workspace.table.valuePtr() + distinct, values + count);
        count += distinct;
      } else {
        // The split has at least two buckets, the first with lo and the last with hi, so each is smaller than this
        // one.
        Index next = base + buckets + 1;
        Index parts = split_into_buckets(bucketIndices + begin, bucketValues + begin, n, lo, hi, splitIndices + begin,
                                         splitValues + begin, workspace.cursor, next);
        count = merge_buckets(splitIndices + begin, splitValues + begin, bucketIndices + begin, bucketValues + begin,
                              workspace, next, parts, indices, values, count);
      }
      begin = end;
      continue;
    }
    Index first = count;
    for (Index p = begin; p < end; ++p) {
      StorageIndex i = bucketIndices[p];
      Index q = count;
      while (q > first && indices[q - 1] > i) --q;
      if (q > first && indices[q - 1] == i) {
        values[q - 1] += bucketValues[p];
        continue;
      }
      for (Index t = count; t > q; --t) {
        indices[t] = indices[t - 1];
        values[t] = move_if_assignable(values[t - 1]);
      }
      indices[q] = i;
      values[q] = move_if_assignable(bucketValues[p]);
      ++count;
    }
    begin = end;
  }
  return count;
}

// Sorts the n distinct entries (indices[k], values[k]) by index, given the lowest and highest index and the number of
// descents, the positions k with indices[k] < indices[k - 1]: the entries form descents + 1 increasing runs.
//
// - No descent: sorted.
// - Up to kSmall entries: insertion sort, unless they fill half of their span.
// - Few runs: insertion sort, which moves an entry only past the larger ones before it, within a budget of moves.
//   Long interleaved runs exhaust it.
// - Otherwise, and for what the budget leaves unsorted: a direct table if the entries are dense in their span, which
//   sorts them without a comparison, and buckets if they are not.
template <typename Scalar, typename StorageIndex>
void sort_inner_vector(StorageIndex* indices, Scalar* values, Index n, StorageIndex lo, StorageIndex hi, Index descents,
                       inner_sort_workspace<Scalar, StorageIndex>& workspace) {
  using Workspace = inner_sort_workspace<Scalar, StorageIndex>;
  constexpr Index kSmall = 64;
  if (descents == 0) return;
  bool small = n <= kSmall && Index(hi) - Index(lo) + 1 > 2 * n;
  Index sortedEnd = 1;
  if (small || (n > kSmall && descents * 4 <= n)) {
    Index budget = small ? n * n : 4 * n;
    for (; sortedEnd < n && budget >= 0; ++sortedEnd) {
      Index q = sortedEnd;
      if (indices[q - 1] < indices[q]) continue;
      StorageIndex index = indices[q];
      Scalar value = std::move(values[q]);
      do {
        indices[q] = indices[q - 1];
        values[q] = move_if_assignable(values[q - 1]);
        --q;
      } while (q > 0 && indices[q - 1] > index);
      indices[q] = index;
      values[q] = move_if_assignable(value);
      budget -= sortedEnd - q;
    }
  }
  if (sortedEnd == n) return;
  direct_accumulator<Scalar, StorageIndex>& acc = workspace.acc;
  if (acc.direct(lo, hi, n)) {
    Workspace::reserve(workspace.table, acc.size(lo, hi));
    acc.init(workspace.table.indexPtr(), workspace.table.valuePtr(), lo, hi);
    for (Index k = 0; k < n; ++k) {
      bool inserted;
      acc.findOrInsert(indices[k], inserted) = move_if_assignable(values[k]);
    }
    acc.compact();
    std::copy(workspace.table.indexPtr(), workspace.table.indexPtr() + n, indices);
    move_or_copy(workspace.table.valuePtr(), workspace.table.valuePtr() + n, values);
    return;
  }
  Workspace::reserve(workspace.buckets, n);
  Workspace::reserve(workspace.split, n);
  Index buckets = split_into_buckets(indices, values, n, lo, hi, workspace.buckets.indexPtr(),
                                     workspace.buckets.valuePtr(), workspace.cursor, 0);
  merge_buckets(workspace.buckets.indexPtr(), workspace.buckets.valuePtr(), workspace.split.indexPtr(),
                workspace.split.valuePtr(), workspace, 0, buckets, indices, values, 0);
}

}  // end namespace internal

/** \ingroup SparseCore_Module
 *
 * \class Triplet
 *
 * \brief A small structure to hold a non zero as a triplet (i,j,value).
 *
 * \sa SparseMatrix::setFromTriplets()
 */
template <typename Scalar, typename StorageIndex = typename SparseMatrix<Scalar>::StorageIndex>
class Triplet {
 public:
  Triplet() = default;

  Triplet(const StorageIndex& i, const StorageIndex& j, const Scalar& v = Scalar(0)) : m_row(i), m_col(j), m_value(v) {}

  /** \returns the row index of the element */
  const StorageIndex& row() const { return m_row; }

  /** \returns the column index of the element */
  const StorageIndex& col() const { return m_col; }

  /** \returns the value of the element */
  const Scalar& value() const { return m_value; }

 protected:
  StorageIndex m_row = 0;
  StorageIndex m_col = 0;
  Scalar m_value = Scalar(0);
};

}  // end namespace Eigen

#endif  // EIGEN_SPARSEUTIL_H

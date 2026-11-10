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

// Sums (i, v) pairs with lo <= i <= hi in storage the caller owns, keeping the slots in index order, so one scan
// yields the sorted, duplicate-free pattern: a column of a sparse product, a row of an incomplete factor.
//
// Open addressing with monotone_hash and ordered linear probing: a probe stops at the first key >= i, and an insertion
// there shifts the run up to the next empty slot right by one, so slot order follows key order. Probes never wrap:
// with at most c = min(maxKeys, span) distinct keys, every run ends below m + c. m = 1.5 c (load 2/3); at load 1,
// runs grow long on clustered columns. A direct hash needs no tail.
//
// Empty slots hold highest(), above any index, so a probe tests one comparison per slot.
template <typename Scalar, typename StorageIndex>
class ordered_hash_accumulator {
  static constexpr StorageIndex kEmpty = NumTraits<StorageIndex>::highest();

 public:
  // Slots needed for keys in [lo, hi] with at most maxKeys distinct keys.
  static Index size(StorageIndex lo, StorageIndex hi, Index maxKeys) {
    return table_size(monotone_hash<StorageIndex>(lo, hi, buckets(lo, hi, maxKeys)), maxKeys);
  }

  // Binds the table to keys[0, size()) and values[0, size()) and empties it.
  void init(StorageIndex* keys, Scalar* values, StorageIndex lo, StorageIndex hi, Index maxKeys) {
    m_hash.compute(lo, hi, buckets(lo, hi, maxKeys));
    m_keys = keys;
    m_values = values;
    m_top = 0;
    std::fill_n(keys, table_size(m_hash, maxKeys), kEmpty);
  }

  // The value of index i, inserted as zero if absent. The reference is valid until the next insertion.
  Scalar& coeffRef(StorageIndex i) {
    Index s = m_hash(i);
    while (m_keys[s] < i) ++s;
    if (m_keys[s] != i) {
      Index q = s;
      while (m_keys[q] != kEmpty) ++q;
      m_top = numext::maxi(m_top, q);
      for (; q > s; --q) m_keys[q] = m_keys[q - 1], m_values[q] = m_values[q - 1];
      m_keys[s] = i;
      m_values[s] = Scalar(0);
    }
    return m_values[s];
  }

  // Moves the n entries to keys[0, n) and values[0, n) in increasing index order and returns n. The write slot never
  // passes the read slot, so the table may sit where its output belongs. init() must run before the next use.
  Index compact() {
    Index n = 0;
    for (Index s = 0; s <= m_top; ++s)
      if (m_keys[s] != kEmpty) m_keys[n] = m_keys[s], m_values[n] = m_values[s], ++n;
    return n;
  }

 private:
  static Index buckets(StorageIndex lo, StorageIndex hi, Index maxKeys) {
    Index c = numext::mini(maxKeys, Index(hi) - Index(lo) + 1);
    return numext::maxi<Index>(1, c + c / 2);
  }
  static Index table_size(const monotone_hash<StorageIndex>& h, Index maxKeys) {
    return h.direct() ? h.range() : h.range() + numext::mini(maxKeys, h.range());
  }

  monotone_hash<StorageIndex> m_hash;
  StorageIndex* m_keys = nullptr;
  Scalar* m_values = nullptr;
  Index m_top = 0;
};

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

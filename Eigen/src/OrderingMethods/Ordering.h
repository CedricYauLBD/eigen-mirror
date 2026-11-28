// SPDX-License-Identifier: MPL-2.0

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012  Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ORDERING_H
#define EIGEN_ORDERING_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"
#include "Eigen_Colamd.h"

namespace Eigen {

/** \ingroup OrderingMethods_Module
 * \class AMDOrdering
 *
 * Functor computing the \em approximate \em minimum \em degree ordering
 * If the matrix is not structurally symmetric, an ordering of A^T+A is computed.
 * Only the sparsity pattern of the input is read — scalar values are not.
 * \tparam  StorageIndex The type of indices of the matrix
 * \sa COLAMDOrdering
 */
template <typename StorageIndex>
class AMDOrdering {
 public:
  using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;

  /** Compute the permutation vector from a sparse matrix.
   * Only the sparsity pattern of \a mat is read; scalar values are not.
   * This routine is much faster if the input matrix is column-major.
   */
  template <typename MatrixType>
  void operator()(const MatrixType& mat, PermutationType& perm) const {
    // AMD only reads the sparsity pattern. Build a column-major view of mat,
    // then materialize \c pattern(mat + mat^T) directly into a
    // SparseMatrix<signed char> (1-byte placeholder values), bypassing
    // Eigen's generic transpose + sparse-sum evaluators.
    Matrix<StorageIndex, Dynamic, 1> outer_buf;
    Matrix<StorageIndex, Dynamic, 1> inner_buf;
    internal::SparsityPatternRef<StorageIndex> pat = internal::make_col_major_pattern_ref(mat, outer_buf, inner_buf);
    SparseMatrix<signed char, ColMajor, StorageIndex> symm;
    internal::materialize_at_plus_a_pattern(pat, symm);
    internal::minimum_degree_ordering(symm, perm);
  }

  /** Compute the permutation with a selfadjoint matrix.
   * Only the sparsity pattern is used; scalar values are not.
   */
  template <typename SrcType, unsigned int SrcUpLo>
  void operator()(const SparseSelfAdjointView<SrcType, SrcUpLo>& mat, PermutationType& perm) const {
    // Build a column-major pattern view of the underlying matrix and expand
    // its UpLo triangle to the full symmetric pattern in one pass, bypassing
    // Eigen's generic selfadjointView assignment evaluator.
    Matrix<StorageIndex, Dynamic, 1> outer_buf;
    Matrix<StorageIndex, Dynamic, 1> inner_buf;
    internal::SparsityPatternRef<StorageIndex> pat =
        internal::make_col_major_pattern_ref(mat.matrix(), outer_buf, inner_buf);
    SparseMatrix<signed char, ColMajor, StorageIndex> symm;
    internal::materialize_selfadjoint_pattern<SrcUpLo>(pat, symm);
    internal::minimum_degree_ordering(symm, perm);
  }
};

/** \ingroup OrderingMethods_Module
 * \class NaturalOrdering
 *
 * Functor computing the natural ordering (identity)
 *
 * \note Returns an empty permutation matrix
 * \tparam  StorageIndex The type of indices of the matrix
 */
template <typename StorageIndex>
class NaturalOrdering {
 public:
  using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;

  /** Compute the permutation vector from a column-major sparse matrix */
  template <typename MatrixType>
  void operator()(const MatrixType& /*mat*/, PermutationType& perm) const {
    perm.resize(0);
  }
};

/** \ingroup OrderingMethods_Module
 * \class COLAMDOrdering
 *
 * \tparam  StorageIndex The type of indices of the matrix
 *
 * Functor computing the \em column \em approximate \em minimum \em degree ordering.
 * Only the sparsity pattern of the input is read — scalar values are not.
 */
template <typename StorageIndex>
class COLAMDOrdering {
 public:
  using PermutationType = PermutationMatrix<Dynamic, Dynamic, StorageIndex>;
  using IndexVector = Matrix<StorageIndex, Dynamic, 1>;

  /** Compute the permutation vector \a perm from the sparse matrix \a mat.
   * Throws \c std::bad_alloc if the rows, columns or stored entries of \a mat, or COLAMD's workspace of about
   * 2.2 nnz + 7 cols + 4 rows indices, exceed the range of \c StorageIndex.
   */
  template <typename MatrixType>
  void operator()(const MatrixType& mat, PermutationType& perm) const {
    using PatternIndex = typename pattern_index<MatrixType>::type;
    Matrix<PatternIndex, Dynamic, 1> outer_buf, inner_buf;
    internal::SparsityPatternRef<PatternIndex> pat = internal::make_col_major_pattern_ref(mat, outer_buf, inner_buf);
    // Accumulate in Index — Eigen's contract is that any valid nnz fits there
    // (mat.nonZeros() returns Index), so the sum can't overflow.
    Index total_nnz = 0;
    for (Index j = 0; j < pat.outerSize; ++j) total_nnz += pat.nonZeros(j);
    // convert_index() checks the range only under EIGEN_INTERNAL_DEBUGGING, and with narrowed sizes Colamd rejects
    // valid row indices or is handed an undersized workspace.
    internal::check_sparse_solver_size_for_overflow<StorageIndex>(pat.innerSize, pat.outerSize);
    if (total_nnz > NumTraits<StorageIndex>::highest()) internal::throw_std_bad_alloc();
    const StorageIndex m = internal::convert_index<StorageIndex>(pat.innerSize);
    const StorageIndex n = internal::convert_index<StorageIndex>(pat.outerSize);
    const StorageIndex nnz = internal::convert_index<StorageIndex>(total_nnz);

    const StorageIndex Alen = internal::Colamd::recommended(nnz, m, n);
    if (Alen < 0) internal::throw_std_bad_alloc();
    // StorageIndex can be wider than Index, and the workspace below is allocated with an Index length.
    using WideIndex = std::common_type_t<Index, StorageIndex>;
    if (WideIndex(Alen) > WideIndex(NumTraits<Index>::highest())) internal::throw_std_bad_alloc();
    double knobs[internal::Colamd::NKnobs];
    StorageIndex stats[internal::Colamd::NStats];
    internal::Colamd::set_defaults(knobs);

    // Colamd writes into A[] in place and needs a contiguous CSC layout, so
    // always compact per column — handles both compressed and uncompressed
    // sources uniformly via SparsityPatternRef::nonZeros(j).
    IndexVector p(n + 1), A(Alen);
    p(0) = 0;
    for (StorageIndex j = 0; j < n; ++j) {
      const Index nz = pat.nonZeros(j);
      const PatternIndex* src = pat.inner + pat.outer[j];
      copy_colamd_indices(src, nz, A.data() + p(j), std::is_same<PatternIndex, StorageIndex>());
      p(j + 1) = p(j) + static_cast<StorageIndex>(nz);
    }

    StorageIndex info = internal::Colamd::compute_ordering(m, n, Alen, A.data(), p.data(), knobs, stats);
    eigen_assert(info && "COLAMD failed");
    // Colamd fails only on a malformed pattern. p then holds no ordering to invert.
    if (!info) {
      perm.setIdentity(n);
      return;
    }

    perm.resize(n);
    for (StorageIndex i = 0; i < n; i++) perm.indices()(p(i)) = i;
  }

 private:
  // Index type of the column-major pattern handed to COLAMD. A column-major SparseMatrix is read in place, in its own
  // index type. Any other source is materialized in StorageIndex rather than in the source's index type: the row
  // numbers of a row-major source are outer indices, which can exceed the range of that type.
  template <typename MatrixType>
  struct pattern_index {
    using type = StorageIndex;
  };
  template <typename Scalar, typename MatrixStorageIndex>
  struct pattern_index<SparseMatrix<Scalar, ColMajor, MatrixStorageIndex>> {
    using type = MatrixStorageIndex;
  };

  template <typename SrcStorageIndex>
  static void copy_colamd_indices(const SrcStorageIndex* src, Index nz, StorageIndex* dst, std::true_type) {
    std::copy_n(src, nz, dst);
  }

  template <typename SrcStorageIndex>
  static void copy_colamd_indices(const SrcStorageIndex* src, Index nz, StorageIndex* dst, std::false_type) {
    for (Index k = 0; k < nz; ++k) dst[k] = internal::convert_index<StorageIndex>(src[k]);
  }
};

}  // end namespace Eigen

#endif

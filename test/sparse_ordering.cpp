// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse.h"
#include <Eigen/OrderingMethods>

template <typename Idx>
void verify_is_permutation(const PermutationMatrix<Dynamic, Dynamic, Idx>& perm) {
  using Vec = Matrix<Idx, Dynamic, 1>;
  Vec sorted = perm.indices();
  std::sort(sorted.data(), sorted.data() + sorted.size());
  Vec expected = Vec::LinSpaced(perm.size(), 0, Idx(perm.size() - 1));
  VERIFY_IS_EQUAL(sorted, expected);
}

template <typename Scalar>
void test_colamd_uncompressed() {
  using Idx = int;
  using SpMat = SparseMatrix<Scalar, ColMajor, Idx>;
  const int n = 32;
  SpMat A(n, n);
  A.reserve(VectorXi::Constant(n, 5));
  for (int j = 0; j < n; ++j) {
    A.insert(j, j) = Scalar(2) + Scalar(j);
    if (j > 0) A.insert(j - 1, j) = Scalar(-1);
    if (j + 1 < n) A.insert(j + 1, j) = Scalar(-1);
  }
  A.insert(0, n - 1) = Scalar(0.5);
  A.insert(n - 1, 0) = Scalar(0.25);
  VERIFY(!A.isCompressed());

  // Regression for #649: COLAMD on uncompressed input used to silently corrupt
  // the permutation in release builds.
  PermutationMatrix<Dynamic, Dynamic, Idx> perm_u, perm_c;
  COLAMDOrdering<Idx> ord;
  ord(A, perm_u);
  verify_is_permutation(perm_u);

  SpMat B = A;
  B.makeCompressed();
  ord(B, perm_c);
  verify_is_permutation(perm_c);

  VERIFY_IS_EQUAL(perm_u.indices(), perm_c.indices());
}

void test_colamd_mixed_index_types() {
  using SpMat = SparseMatrix<double, ColMajor, int>;
  const int n = 8;
  SpMat A(n, n);
  A.reserve(VectorXi::Constant(n, 3));
  for (int j = 0; j < n; ++j) {
    A.insert(j, j) = 2.0 + j;
    if (j > 0) A.insert(j - 1, j) = -1.0;
    if (j + 1 < n) A.insert(j + 1, j) = -1.0;
  }
  A.makeCompressed();

  // COLAMDOrdering<StorageIndex> is a public API choice independent of the
  // source matrix's StorageIndex.  Keep accepting mixed index types.
  PermutationMatrix<Dynamic, Dynamic, long> perm;
  COLAMDOrdering<long> ord;
  ord(A, perm);
  verify_is_permutation(perm);
}

// The rows of a row-major matrix are its outer dimension and can exceed the range of its StorageIndex. COLAMD's
// column-major pattern used to be built in that type: the row numbers wrapped, COLAMD rejected them, and release
// builds scattered the permutation through its untouched column pointers.
void test_colamd_row_major_narrow_index() {
  const Index rows = 40000;
  VERIFY(rows > Index(NumTraits<short>::highest()));
  SparseMatrix<double, RowMajor, short> A(rows, 3);
  for (int i = 0; i < 40; ++i) A.insert(i, 0) = 1.0;
  A.insert(35000, 1) = 1.0;
  A.insert(2, 2) = 1.0;
  A.makeCompressed();

  PermutationMatrix<Dynamic, Dynamic, int> perm, perm_ref, perm_transposed;
  COLAMDOrdering<int> ord;
  ord(A, perm);
  VERIFY_IS_EQUAL(perm.size(), Index(3));
  verify_is_permutation(perm);

  // Same pattern in a matrix whose own index type holds the row numbers.
  SparseMatrix<double, ColMajor, int> A_ref(rows, 3);
  for (int i = 0; i < 40; ++i) A_ref.insert(i, 0) = 1.0;
  A_ref.insert(35000, 1) = 1.0;
  A_ref.insert(2, 2) = 1.0;
  ord(A_ref, perm_ref);
  VERIFY_IS_EQUAL(perm.indices(), perm_ref.indices());

  // Same pattern again, as the transpose of a column-major matrix with 16-bit indices.
  SparseMatrix<double, ColMajor, short> At(3, rows);
  for (int i = 0; i < 40; ++i) At.insert(0, i) = 1.0;
  At.insert(1, 35000) = 1.0;
  At.insert(2, 2) = 1.0;
  ord(At.transpose(), perm_transposed);
  VERIFY_IS_EQUAL(perm_transposed.indices(), perm_ref.indices());
}

template <typename Scalar>
void test_amd_uncompressed() {
  using Idx = int;
  using SpMat = SparseMatrix<Scalar, ColMajor, Idx>;
  const int n = 24;
  SpMat A(n, n);
  A.reserve(VectorXi::Constant(n, 4));
  for (int j = 0; j < n; ++j) {
    A.insert(j, j) = Scalar(2) + Scalar(j);
    if (j > 0) {
      A.insert(j - 1, j) = Scalar(-1);
      A.insert(j, j - 1) = Scalar(-1);
    }
  }
  VERIFY(!A.isCompressed());

  PermutationMatrix<Dynamic, Dynamic, Idx> perm;
  AMDOrdering<Idx> ord;
  ord(A, perm);
  verify_is_permutation(perm);
}

// The selfadjoint view of a matrix with unsorted inner indices must still expand to a symmetric pattern: AMD indexed
// out of bounds when one half of it was selected by value and the other by binary search.
template <typename Scalar>
void test_amd_selfadjoint_unsorted() {
  using Idx = int;
  using SpMat = SparseMatrix<Scalar, ColMajor, Idx>;
  const int n = 5;
  SpMat A(n, n);
  for (int j = 0; j < n; ++j) A.insert(j, j) = Scalar(2);
  const int edges[3][2] = {{0, 3}, {3, 4}, {2, 4}};
  for (const auto& e : edges) {
    A.insert(e[0], e[1]) = Scalar(-1);
    A.insert(e[1], e[0]) = Scalar(-1);
  }
  A.makeCompressed();
  // Column 3 holds the rows 0 3 4. List them as 4 0 3, as a deserialized matrix may.
  Idx* rows = A.innerIndexPtr() + A.outerIndexPtr()[3];
  std::rotate(rows, rows + 2, rows + 3);
  VERIFY(A.innerIndicesAreSorted() < A.outerSize());

  PermutationMatrix<Dynamic, Dynamic, Idx> perm;
  AMDOrdering<Idx> ord;
  ord(A.template selfadjointView<Lower>(), perm);
  verify_is_permutation(perm);
  ord(A.template selfadjointView<Upper>(), perm);
  verify_is_permutation(perm);
}

// Node 0 is adjacent to the `degree` highest-numbered nodes, every other node only to itself.
template <typename Idx>
PermutationMatrix<Dynamic, Dynamic, Idx> amd_of_star(Idx n, Idx degree) {
  std::vector<Triplet<double, Idx> > triplets;
  triplets.reserve(n + 2 * degree);
  for (Idx j = 0; j < n; ++j) triplets.emplace_back(j, j, 1.0);
  for (Idx j = n - degree; j < n; ++j) {
    triplets.emplace_back(0, j, 1.0);
    triplets.emplace_back(j, 0, 1.0);
  }
  SparseMatrix<double, ColMajor, Idx> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  PermutationMatrix<Dynamic, Dynamic, Idx> perm;
  AMDOrdering<Idx> ord;
  ord(A, perm);
  return perm;
}

void test_amd_hash_overflow() {
  // Node 0 stays below AMD's dense threshold 10*sqrt(n) (its column also holds the diagonal), and its supervariable
  // hash, the sum of its neighbours' indices, exceeds INT_MAX. With a signed int hash this overflowed, h % n was
  // negative and hhead[h] was written out of bounds. The ordering does not depend on the hash values, so the int and
  // the 64-bit orderings must agree.
  const int n = 400000;
  const int degree = int(10 * std::sqrt(double(n))) - 2;
  VERIFY(double(degree) * double(n - degree) > double(NumTraits<int>::highest()));
  PermutationMatrix<Dynamic, Dynamic, int> perm = amd_of_star<int>(n, degree);
  verify_is_permutation(perm);
  PermutationMatrix<Dynamic, Dynamic, std::int64_t> perm64 = amd_of_star<std::int64_t>(n, degree);
  VERIFY_IS_EQUAL(perm.indices().cast<std::int64_t>().eval(), perm64.indices());
}

// Symmetric pattern with a full diagonal and `pairs` entries on each side of it, on the nearest subdiagonals.
template <typename Idx>
SparseMatrix<double, ColMajor, Idx> banded_pattern(Index n, Index pairs) {
  std::vector<Triplet<double, Idx> > triplets;
  triplets.reserve(n + 2 * pairs);
  for (Index j = 0; j < n; ++j) triplets.emplace_back(Idx(j), Idx(j), 1.0);
  for (Index band = 1; pairs > 0; ++band) {
    for (Index j = 0; j + band < n && pairs > 0; ++j, --pairs) {
      triplets.emplace_back(Idx(j), Idx(j + band), 1.0);
      triplets.emplace_back(Idx(j + band), Idx(j), 1.0);
    }
  }
  SparseMatrix<double, ColMajor, Idx> A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

#ifdef EIGEN_EXCEPTIONS
// Orders a rows x 3 pattern with 7 entries, for which COLAMD's workspace is
// 2*7 + 6*(3+1) + 4*(rows+1) + 3 + 7/5 = 4*rows + 46 indices.
template <typename MatIdx, typename OrdIdx>
bool colamd_throws_bad_alloc(Index rows) {
  SparseMatrix<double, ColMajor, MatIdx> A(rows, 3);
  for (Index i : {0, 5, 6, 7}) A.insert(i, 0) = 1.0;
  for (Index i : {1, 5}) A.insert(i, 1) = 1.0;
  A.insert(1, 2) = 1.0;
  A.makeCompressed();
  PermutationMatrix<Dynamic, Dynamic, OrdIdx> perm;
  COLAMDOrdering<OrdIdx> ord;
  try {
    ord(A, perm);
  } catch (const std::bad_alloc&) {
    return true;
  }
  verify_is_permutation(perm);
  return false;
}

template <typename Idx>
bool amd_throws_bad_alloc(const SparseMatrix<double, ColMajor, Idx>& A) {
  PermutationMatrix<Dynamic, Dynamic, Idx> perm;
  AMDOrdering<Idx> ord;
  try {
    ord(A, perm);
  } catch (const std::bad_alloc&) {
    return true;
  }
  return false;
}
#endif

// COLAMD counts rows, columns, entries and its workspace in OrdIdx. A matrix that does not fit used to be ordered
// with wrapped sizes: an undersized workspace was written past, and in release builds COLAMD's failure on the
// narrowed row count was ignored and the column pointers were inverted as if they were the ordering.
template <typename MatIdx, typename OrdIdx>
void test_colamd_size_overflow() {
#ifdef EIGEN_EXCEPTIONS
  if (sizeof(OrdIdx) > sizeof(Index)) return;
  const Index highest = static_cast<Index>(NumTraits<OrdIdx>::highest());
  VERIFY(!(colamd_throws_bad_alloc<MatIdx, OrdIdx>(8)));
  // The row count fits, the workspace does not: 4*rows + 46 wraps to a negative, to a small positive (4*2^30 + 46
  // to 46 for int) and to an overflowing rows + 1.
  VERIFY((colamd_throws_bad_alloc<MatIdx, OrdIdx>((highest - 46) / 4 + 1)));
  VERIFY((colamd_throws_bad_alloc<MatIdx, OrdIdx>(highest / 2 + 1)));
  VERIFY((colamd_throws_bad_alloc<MatIdx, OrdIdx>(highest)));
  // The row count does not fit: 2^32 + 2 narrows to 2 for int, below the stored row indices.
  if (sizeof(OrdIdx) < sizeof(Index)) {
    const numext::uint64_t rows = 2 * (static_cast<numext::uint64_t>(highest) + 1) + 2;
    VERIFY((colamd_throws_bad_alloc<MatIdx, OrdIdx>(static_cast<Index>(rows))));
  }
#endif
}

// The row-major source has wider indices than the ordering. Its column offsets are summed in the ordering's narrow
// index type; 257 * 257 entries wrapped to 513, and the pattern was then scattered into a buffer of that size. (COLAMD
// is not instantiated for 16-bit indices, so AMD, which uses the same materialization, covers this.)
// The expanded pattern of a triangle can also exceed the range although the triangle itself fits: 250 * 251 / 2
// entries become 250 * 250 = 62500, which does not fit in 16 bits.
void test_pattern_offsets_overflow() {
#ifdef EIGEN_EXCEPTIONS
  using Narrow = short;
  const Index rows = 257, cols = 257;
  VERIFY(rows * cols > 2 * Index(NumTraits<Narrow>::highest()) + 1);
  SparseMatrix<double, RowMajor, int> A(rows, cols);
  A.reserve(VectorXi::Constant(rows, cols));
  for (Index i = 0; i < rows; ++i)
    for (Index j = 0; j < cols; ++j) A.insert(i, j) = 1.0;
  PermutationMatrix<Dynamic, Dynamic, Narrow> perm;
  AMDOrdering<Narrow> amd;
  bool thrown = false;
  try {
    amd(A, perm);
  } catch (const std::bad_alloc&) {
    thrown = true;
  }
  VERIFY(thrown);

  const Index n = 250;
  SparseMatrix<double, ColMajor, Narrow> L(n, n);
  L.reserve(VectorXi::LinSpaced(n, n, 1));
  for (Index j = 0; j < n; ++j)
    for (Index i = j; i < n; ++i) L.insert(i, j) = 1.0;
  VERIFY(L.nonZeros() <= Index(NumTraits<Narrow>::highest()));
  VERIFY(n * n > Index(NumTraits<Narrow>::highest()));
  thrown = false;
  try {
    amd(L.template selfadjointView<Lower>(), perm);
  } catch (const std::bad_alloc&) {
    thrown = true;
  }
  VERIFY(thrown);
#endif
}

// With a 32-bit Index and 64-bit ordering indices the workspace length fits the ordering's index type but is
// narrowed by the Index length of the allocation. The matrix has few entries, so it is cheap to build.
void test_colamd_workspace_exceeds_index() {
#ifdef EIGEN_EXCEPTIONS
  if (sizeof(std::int64_t) <= sizeof(Index)) return;
  const Index rows = NumTraits<Index>::highest() / 2;  // 4 * rows + 46 exceeds the range of Index.
  VERIFY((colamd_throws_bad_alloc<int, std::int64_t>(rows)));
#endif
}

void test_amd_index_range() {
  // AMD stores as StorageIndex the nodes 0..n and the positions in the pattern of A + A^T, which it grows to
  // t = nnz + nnz / 5 + 2 * n entries. It computed t, and the size 8 * (n + 1) of its workspace, in StorageIndex
  // without a range check: with int and n = 2^29, 8 * (n + 1) wrapped to 8 and the workspace was overrun. A short
  // index reaches the bound on t at a size that fits a test.
  using Idx = short;
  using SpMat = SparseMatrix<double, ColMajor, Idx>;
  const Index highest = NumTraits<Idx>::highest();
  const Index n = 3000;
  const auto grown_size = [](Index nnz, Index cols) { return nnz + nnz / 5 + 2 * cols; };

  SpMat A = banded_pattern<Idx>(n, 9653);
  VERIFY_IS_EQUAL(grown_size(A.nonZeros(), n), highest);
  PermutationMatrix<Dynamic, Dynamic, Idx> perm;
  AMDOrdering<Idx> ord;
  ord(A, perm);
  verify_is_permutation(perm);
  // The ordering does not depend on the index type.
  PermutationMatrix<Dynamic, Dynamic, int> perm_int;
  AMDOrdering<int> ord_int;
  ord_int(banded_pattern<int>(n, 9653), perm_int);
  VERIFY_IS_EQUAL(perm.indices().cast<int>().eval(), perm_int.indices());

#ifdef EIGEN_EXCEPTIONS
  A = banded_pattern<Idx>(n, 9654);
  VERIFY(grown_size(A.nonZeros(), n) > highest);
  VERIFY(amd_throws_bad_alloc(A));

  // The outer dimension of a matrix is not bounded by its StorageIndex, but the nodes of AMD are.
  VERIFY(amd_throws_bad_alloc(SpMat(highest + 1000, highest + 1000)));
#endif
}

EIGEN_DECLARE_TEST(sparse_ordering) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(test_colamd_uncompressed<double>());
    CALL_SUBTEST_2(test_colamd_uncompressed<std::complex<double> >());
    CALL_SUBTEST_3(test_amd_uncompressed<double>());
    CALL_SUBTEST_3(test_amd_selfadjoint_unsorted<double>());
    CALL_SUBTEST_4(test_colamd_mixed_index_types());
  }
  CALL_SUBTEST_4(test_colamd_row_major_narrow_index());
  CALL_SUBTEST_4(test_pattern_offsets_overflow());
  CALL_SUBTEST_4(test_colamd_workspace_exceeds_index());
  CALL_SUBTEST_5(test_amd_hash_overflow());
  CALL_SUBTEST_5(test_amd_index_range());
  CALL_SUBTEST_6((test_colamd_size_overflow<int, int>()));
  CALL_SUBTEST_6((test_colamd_size_overflow<std::int64_t, int>()));
  CALL_SUBTEST_6((test_colamd_size_overflow<int, std::int64_t>()));
}

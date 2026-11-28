// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Index is int here, as on 32-bit targets, so that a StorageIndex can be wider than Index.
#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#endif
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "main.h"

#include <Eigen/SparseCore>

// Random insertion past the reserved room must grow the storage. Its growth limit is NumTraits<StorageIndex>::highest()
// clamped to Index; a StorageIndex wider than Index used to wrap the limit to -1, so the insertion wrote past the end
// of the storage instead.
template <typename StorageIndex, int Options>
void sparse_random_insertion_growth(bool reserveFirst) {
  using SpMat = SparseMatrix<double, Options, StorageIndex>;
  using DenseMat = Matrix<double, Dynamic, Dynamic>;
  const Index rows = internal::random<Index>(2, 40);
  const Index cols = internal::random<Index>(2, 40);

  // Every coefficient, in random order, so that most insertions find no free slot in their inner vector.
  std::vector<Index> order(rows * cols);
  for (Index k = 0; k < rows * cols; ++k) order[k] = k;
  for (Index k = rows * cols - 1; k > 0; --k) std::swap(order[k], order[internal::random<Index>(0, k)]);

  SpMat A(rows, cols);
  if (reserveFirst) A.reserve(VectorXi::Constant(A.outerSize(), 1));
  DenseMat ref = DenseMat::Zero(rows, cols);
  for (Index k : order) {
    const Index i = k % rows, j = k / rows;
    A.insert(i, j) = ref(i, j) = double(k + 1);
  }
  VERIFY_IS_EQUAL(A.nonZeros(), rows * cols);
  VERIFY_IS_EQUAL(A.innerIndicesAreSorted(), A.outerSize());
  VERIFY_IS_EQUAL(DenseMat(A), ref);

  A.makeCompressed();
  VERIFY(A.isCompressed());
  VERIFY_IS_EQUAL(DenseMat(A), ref);
}

template <typename StorageIndex>
void sparse_random_insertion_growth() {
  for (bool reserveFirst : {false, true}) {
    sparse_random_insertion_growth<StorageIndex, ColMajor>(reserveFirst);
    sparse_random_insertion_growth<StorageIndex, RowMajor>(reserveFirst);
  }
}

// A range of triplets that is generated while it is read: triplet k is the k-th position of a 65536 x 32768 matrix in
// column-major order. `visited` records how far the range was read.
struct GeneratedTriplet {
  long long k;
  long long* visited;
  const GeneratedTriplet* operator->() const { return this; }
  int row() const { return int(k & 0xffff); }
  int col() const { return int(k >> 16); }
  float value() const { return 1.f; }
  GeneratedTriplet& operator++() {
    *visited = ++k;
    return *this;
  }
  bool operator==(const GeneratedTriplet& other) const { return k == other.k; }
  bool operator!=(const GeneratedTriplet& other) const { return k != other.k; }
};

// setFromTriplets() and setFromSortedTriplets() count the entries in an Index, so the count is limited by Index's
// range as well as StorageIndex's. They checked StorageIndex's alone; with a StorageIndex wider than Index the count
// wrapped, and the entries were written past the end of a storage sized by the wrapped count.
void sparse_triplet_count_limit() {
  // With EIGEN_DEBUG_ASSERTS, VERIFY_RAISES_ASSERT records every assertion that passes: several per triplet here.
#if defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_NO_ASSERTION_CHECKING) && !defined(EIGEN_DEBUG_ASSERTS)
  using SpMat = SparseMatrix<float, ColMajor, long long>;
  // One triplet more than an Index can count: 2^31 iterator steps per call, and no storage for them.
  const long long limit = NumTraits<Index>::highest();
  long long visited = 0;
  const GeneratedTriplet begin{0, &visited}, end{limit + 1, &visited};
  SpMat A(65536, 32768);

  // Both must stop at the triplet that exceeds the limit. Reading the range to its end means that the count wrapped.
  VERIFY_RAISES_ASSERT(A.setFromTriplets(begin, end));
  VERIFY_IS_EQUAL(visited, limit);

  visited = 0;
  VERIFY_RAISES_ASSERT(A.setFromSortedTriplets(begin, end));
  VERIFY_IS_EQUAL(visited, limit);
#endif
}

#ifdef EIGEN_HAS_OPENMP
// The OpenMP kernel for column-major sparse * vector gives each thread a scratch stripe of rows() scalars at offset
// t * rows(), and runs only if nonZeros() >= threads * rows(). That product used to be formed in Index. Here it is
// 2^31, which wrapped to a negative value and opened the gate; with more rows the stripe offsets wrap as well.
template <typename Scalar>
void sparse_dense_product_openmp_scratch_gate() {
  using DenseMat = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVec = Matrix<Scalar, Dynamic, 1>;
  const int threads = 1 << 16;
  const Index rows = 1 << 15, cols = 64;
  // cols * filledRows entries: more than the 20000 the kernel asks for, far fewer than threads * rows.
  const Index filledRows = 400;

  const DenseMat refBlock = DenseMat::Random(filledRows, cols);
  SparseMatrix<Scalar, ColMajor> A(rows, cols);
  A.reserve(VectorXi::Constant(cols, int(filledRows)));
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < filledRows; ++i) A.insert(i, j) = refBlock(i, j);
  A.makeCompressed();
  VERIFY_IS_EQUAL(A.nonZeros(), cols * filledRows);

  const DenseVec x = DenseVec::Random(cols);
  DenseVec ref = DenseVec::Zero(rows);
  ref.head(filledRows) = refBlock * x;

  // Not setNbThreads(), which clamps to omp_get_max_threads().
  const int oldThreads = omp_get_max_threads();
  omp_set_num_threads(threads);
  const DenseVec y = A * x;
  omp_set_num_threads(oldThreads);
  VERIFY_IS_APPROX(y, ref);
  VERIFY((y.tail(rows - filledRows).array() == Scalar(0)).all());
}
#endif

EIGEN_DECLARE_TEST(sparse_int_index) {
  STATIC_CHECK(sizeof(Index) == sizeof(int));
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(sparse_random_insertion_growth<long long>());
    CALL_SUBTEST_2(sparse_random_insertion_growth<int>());
  }
  CALL_SUBTEST_3(sparse_triplet_count_limit());
#ifdef EIGEN_HAS_OPENMP
  CALL_SUBTEST_2(sparse_dense_product_openmp_scratch_gate<double>());
#endif
}

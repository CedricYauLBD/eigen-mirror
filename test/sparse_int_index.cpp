// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Index is int here, as on 32-bit targets, so that a StorageIndex can be wider than Index.
#ifdef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#endif
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "main.h"

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

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

// SparseLU sizes its buffers from the number of nonzeros: initially as fillratio * (nonZeros + 1), then growing them by
// a ratio. Both products used to overflow Index, and the first one left factorize() with a buffer it had not allocated.
struct sparselu_buffer_sizes : internal::SparseLUImpl<float, int> {
  static void run() {
    sparselu_buffer_sizes lu;
    GlobalLU_t glu;
    const Index highest = NumTraits<Index>::highest();
    const Index n = 20000, fillratio = 20, panel_size = 16;
    // From highest / fillratio nonzeros on, fillratio * (annz + 1) does not fit Index.
    for (Index annz : {highest / fillratio - 1, highest / fillratio, highest}) {
      // lwork = -1 computes the initial sizes without allocating the buffers.
      lu.memInit(n, n, annz, -1, fillratio, panel_size, glu);
      VERIFY(glu.nzlmax > highest / 4 - fillratio);
      VERIFY_IS_EQUAL(glu.nzumax, n * n);
      VERIFY_IS_EQUAL(glu.nzlumax, n * n);
    }

    VERIFY_IS_EQUAL(internal::LUexpandedLength(1, 1.5f), Index(2));
    VERIFY_IS_EQUAL(internal::LUexpandedLength(1000, 1.5f), Index(1500));
    VERIFY_IS_EQUAL(internal::LUexpandedLength(highest / 3 * 2 + 1000, 1.5f), highest);
    VERIFY_IS_EQUAL(internal::LUexpandedLength(highest, 1.5f), highest);
    // A buffer of the largest length cannot grow: that is a failed expansion, which leaves the buffer as it is.
    IndexVector vec(3);
    Index length = highest, num_expansions = 1;
    VERIFY(lu.expand(vec, length, 0, 0, num_expansions) != 0);
    VERIFY_IS_EQUAL(length, highest);
    VERIFY_IS_EQUAL(vec.size(), Index(3));
  }
};

EIGEN_DECLARE_TEST(sparse_int_index) {
  STATIC_CHECK(sizeof(Index) == sizeof(int));
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(sparse_random_insertion_growth<long long>());
    CALL_SUBTEST_2(sparse_random_insertion_growth<int>());
  }
  CALL_SUBTEST_3(sparselu_buffer_sizes::run());
}

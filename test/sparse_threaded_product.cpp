// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_THREADS 1
#include "sparse.h"

// Pulls in the ThreadedSparseProduct header via the SparseCore module.
// (Eigen/Sparse -> Eigen/SparseCore conditional include under EIGEN_USE_THREADS.)
#include <Eigen/SparseCore>

template <typename SparseMatrixType, typename DenseVectorType>
void verify_threaded_spmv(Index rows, Index cols, double density) {
  using Scalar = typename SparseMatrixType::Scalar;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;

  DenseMatrix refMat(rows, cols);
  SparseMatrixType A(rows, cols);
  initSparse<Scalar>(density, refMat, A);
  A.makeCompressed();

  DenseVectorType x_fwd = DenseVectorType::Random(cols);
  DenseVectorType x_adj = DenseVectorType::Random(rows);

  // Reference results via the existing dense path.
  DenseVectorType y_fwd_ref = refMat * x_fwd;
  DenseVectorType y_adj_ref = refMat.adjoint() * x_adj;

  ThreadedSparseProduct<SparseMatrixType> op(A);
  {
    DenseVectorType wrong_x = DenseVectorType::Zero(cols + 1);
    DenseVectorType wrong_adjoint_x = DenseVectorType::Zero(rows + 1);
    DenseVectorType y(rows), adjoint_y(cols);
    VERIFY_RAISES_ASSERT(op.apply(wrong_x, y));
    VERIFY_RAISES_ASSERT(op.applyAdjoint(wrong_adjoint_x, adjoint_y));
  }
  VERIFY(op.rows() == A.rows());
  VERIFY(op.cols() == A.cols());

  // Overwriting forms.
  {
    DenseVectorType y(rows);
    y.setRandom();
    op.apply(x_fwd, y);
    VERIFY_IS_APPROX(y, y_fwd_ref);
  }
  {
    DenseVectorType y(cols);
    y.setRandom();
    op.applyAdjoint(x_adj, y);
    VERIFY_IS_APPROX(y, y_adj_ref);
  }

  // Accumulating forms: y += alpha * A * x, with both alpha = 1 and a
  // non-trivial alpha.
  for (Scalar alpha : {Scalar(1), Scalar(internal::random<Scalar>())}) {
    {
      DenseVectorType y0 = DenseVectorType::Random(rows);
      DenseVectorType y_ref = y0 + alpha * y_fwd_ref;
      DenseVectorType y = y0;
      op.applyAddTo(x_fwd, y, alpha);
      VERIFY_IS_APPROX(y, y_ref);
    }
    {
      DenseVectorType y0 = DenseVectorType::Random(cols);
      DenseVectorType y_ref = y0 + alpha * y_adj_ref;
      DenseVectorType y = y0;
      op.applyAdjointAddTo(x_adj, y, alpha);
      VERIFY_IS_APPROX(y, y_ref);
    }
  }

  // Calling adjoint must have materialized the mirror; calling it again must
  // not rebuild (no easy assertion for "not rebuilt", but verify the path is
  // consistent across repeated calls).
  VERIFY(op.hasMirror());
  {
    DenseVectorType y(cols);
    op.applyAdjoint(x_adj, y);
    VERIFY_IS_APPROX(y, y_adj_ref);
  }

  // Re-running analyzePattern must invalidate the mirror.
  op.analyzePattern(A);
  VERIFY(!op.hasMirror());
  {
    DenseVectorType y(cols);
    op.applyAdjoint(x_adj, y);
    VERIFY_IS_APPROX(y, y_adj_ref);
  }

  // Coefficient-only invalidation: mutate the bound matrix's stored values
  // (same sparsity pattern), call refreshValues(), and verify the next
  // mirror-using direction picks up the new coefficients. The mirror is
  // materialized lazily by the direction that needs the transposed view --
  // forward for ColMajor A, adjoint for RowMajor A -- so call both to warm
  // it regardless of storage order.
  {
    DenseVectorType y_fwd(rows);
    DenseVectorType y_adj(cols);
    op.apply(x_fwd, y_fwd);
    op.applyAdjoint(x_adj, y_adj);
    VERIFY(op.hasMirror());
  }
  for (Index k = 0; k < A.nonZeros(); ++k) A.valuePtr()[k] *= Scalar(2);
  refMat *= Scalar(2);
  DenseVectorType y_fwd_ref_updated = refMat * x_fwd;
  DenseVectorType y_adj_ref_updated = refMat.adjoint() * x_adj;
  op.refreshValues();
  VERIFY(!op.hasMirror());
  {
    DenseVectorType y(rows);
    op.apply(x_fwd, y);
    VERIFY_IS_APPROX(y, y_fwd_ref_updated);
  }
  {
    DenseVectorType y(cols);
    op.applyAdjoint(x_adj, y);
    VERIFY_IS_APPROX(y, y_adj_ref_updated);
  }
}

// IEEE-754 propagation through the ColMajor SpMV OMP scatter/reduce path
// (Eigen/src/SparseCore/SparseDenseProduct.h). With OpenMP enabled, the
// kernel sums per-thread scratch buffers into y; NaN/Inf in any thread's
// contribution must propagate to the reduced sum. Without OpenMP this still
// exercises the serial scatter, where the same propagation must hold.
template <typename Scalar>
void verify_threaded_scatter_ieee754() {
  using SpMat = SparseMatrix<Scalar, ColMajor>;
  using Vec = Matrix<Scalar, Dynamic, 1>;
  using Real = typename NumTraits<Scalar>::Real;

  // Pin the OMP team size so the scatter/reduce gate (nnz >= threads * m)
  // is deterministic across runs.
  const int prev_threads = Eigen::nbThreads();
  Eigen::setNbThreads(4);

  // 300x300 with diagonal + (i+j)%4==0 entries -> ~22600 nnz, above the
  // 20000-nnz threshold and well above threads*m = 4*300 for the gate.
  const Index n = 300;
  std::vector<Triplet<Scalar> > triplets;
  triplets.reserve(static_cast<std::size_t>(n) * (n / 4 + 1));
  // Simple deterministic LCG so we don't depend on <random> being included
  // transitively. Modulo-bounded so we don't rely on a specific integer width.
  unsigned rng = 0xC0FFEEu;
  for (int j = 0; j < int(n); ++j) {
    for (int i = 0; i < int(n); ++i) {
      if (i == j || (i + j) % 4 == 0) {
        rng = (rng * 1664525u + 1013904223u) & 0xFFFFu;
        const Real v = Real(0.1) + Real(0.9) * Real(rng) / Real(0xFFFF);
        triplets.push_back(Triplet<Scalar>(i, j, Scalar(v)));
      }
    }
  }
  SpMat A(n, n);
  A.setFromTriplets(triplets.begin(), triplets.end());
  A.makeCompressed();
  VERIFY(A.nonZeros() > 20000);

  // Inject NaN/Inf into three rows of column 0 by overwriting valuePtr().
  // Pick rows that are guaranteed in column 0's pattern: row 0 (diagonal),
  // row 4 ((4+0)%4==0), row 8 ((8+0)%4==0).
  const Real nan_r = std::numeric_limits<Real>::quiet_NaN();
  const Real inf_r = std::numeric_limits<Real>::infinity();
  const Index i_nan = 0, i_pinf = 4, i_ninf = 8;
  bool set_nan = false, set_pinf = false, set_ninf = false;
  for (Index k = A.outerIndexPtr()[0]; k < A.outerIndexPtr()[1]; ++k) {
    const Index i = A.innerIndexPtr()[k];
    if (i == i_nan) {
      A.valuePtr()[k] = Scalar(nan_r);
      set_nan = true;
    } else if (i == i_pinf) {
      A.valuePtr()[k] = Scalar(inf_r);
      set_pinf = true;
    } else if (i == i_ninf) {
      A.valuePtr()[k] = Scalar(-inf_r);
      set_ninf = true;
    }
  }
  VERIFY(set_nan && set_pinf && set_ninf);

  Vec x = Vec::Random(n);
  // Ensure x[0] is finite and strictly positive (real part), so the injected
  // ±Inf contributions don't cancel against a zero-multiplier or flip sign.
  x[0] = Scalar(1);

  Vec y = A * x;

  VERIFY((numext::isnan)(numext::real(y[i_nan])));
  VERIFY((numext::isinf)(numext::real(y[i_pinf])) && numext::real(y[i_pinf]) > Real(0));
  VERIFY((numext::isinf)(numext::real(y[i_ninf])) && numext::real(y[i_ninf]) < Real(0));

  Eigen::setNbThreads(prev_threads);
}

// As strictly aligned as Eigen's dynamic storage: with AVX or wider packets this is more than std::allocator provides
// before C++17. Construction checks it. The 128-byte variant is beyond Eigen's own alignment, so the stack accumulator
// of the reduction needs the scalar's alignment, which an alignas() of Eigen's weaker alignment would contradict; the
// scalar's storage in Eigen's containers is then not checked.
template <int Align>
struct alignas(Align) OverAlignedScalarT {
  double value;
  static void check(const OverAlignedScalarT* p) {
    if (Align <= EIGEN_DEFAULT_ALIGN_BYTES) VERIFY(std::uintptr_t(p) % alignof(OverAlignedScalarT) == 0);
  }
  OverAlignedScalarT(double v = 0) : value(v) { check(this); }
  OverAlignedScalarT(const OverAlignedScalarT& other) : value(other.value) { check(this); }
  OverAlignedScalarT& operator=(const OverAlignedScalarT&) = default;
  OverAlignedScalarT& operator+=(const OverAlignedScalarT& other) {
    value += other.value;
    return *this;
  }
  friend OverAlignedScalarT operator*(const OverAlignedScalarT& a, const OverAlignedScalarT& b) {
    return a.value * b.value;
  }
};
using OverAlignedScalar = OverAlignedScalarT<EIGEN_DEFAULT_ALIGN_BYTES>;
using WiderAlignedScalar = OverAlignedScalarT<128>;

namespace Eigen {
// The threaded kernel is reserved for scalars whose product is known to commute.
template <int Align>
struct ScalarBinaryOpTraits<OverAlignedScalarT<Align>, OverAlignedScalarT<Align>> {
  using ReturnType = OverAlignedScalarT<Align>;
  static constexpr bool IsCommutative = true;
};
}  // namespace Eigen

// The per-thread accumulators of the ColMajor SpMV OMP scatter/reduce path
// hold result scalars, so they must be aligned as the result itself is.
template <typename Scalar>
void verify_threaded_scatter_overaligned() {
  STATIC_CHECK((internal::functor_is_commutative<internal::scalar_product_op<Scalar> >::value));
  const int prev_threads = Eigen::nbThreads();
  Eigen::setNbThreads(4);

  // 4 entries per column -> 32768 nnz: above the 20000-nnz threshold and, for
  // up to 4 threads, at least threads*m for the gate. The accumulators are
  // then large enough for glibc to map them separately, 16 bytes past a page
  // boundary: misaligned unless the allocator sees to it.
  const Index n = 8192, nnz_per_col = 4;
  SparseMatrix<Scalar, ColMajor> A(n, n);
  A.reserve(VectorXi::Constant(n, int(nnz_per_col)));
  VectorXd x_ref(n), y_ref = VectorXd::Zero(n);
  for (Index j = 0; j < n; ++j) {
    x_ref[j] = double(j % 7 + 1);
    for (Index k = 0; k < nnz_per_col; ++k) {
      const Index i = (j + k * k) % n;
      A.insert(i, j) = Scalar(double(k + 1));
      y_ref[i] += double(k + 1) * x_ref[j];
    }
  }
  A.makeCompressed();
  VERIFY(A.nonZeros() > 20000);

  Matrix<Scalar, Dynamic, 1> x(n);
  for (Index j = 0; j < n; ++j) x[j] = Scalar(x_ref[j]);
  // Twice: the second product reuses the accumulators, which the first one
  // must have left zeroed.
  for (int pass = 0; pass < 2; ++pass) {
    const Matrix<Scalar, Dynamic, 1> y = A * x;
    for (Index i = 0; i < n; ++i) VERIFY_IS_EQUAL(y[i].value, y_ref[i]);
  }

  Eigen::setNbThreads(prev_threads);
}

template <typename Scalar>
void run_grid() {
  using Vec = Matrix<Scalar, Dynamic, 1>;
  // Cover both storage orders, square and rectangular, small (serial path,
  // nnz < threshold) and larger (threaded path).
  for (int order = 0; order < 2; ++order) {
    if (order == 0) {
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(8, 8, 0.5);
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(13, 21, 0.3);
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(500, 500, 0.15);
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(800, 400, 0.10);
      // Large enough that nnz > kThreadingThreshold (20000) to exercise the
      // multi-threaded dispatch path.
      verify_threaded_spmv<SparseMatrix<Scalar, ColMajor>, Vec>(1500, 1500, 0.02);
    } else {
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(8, 8, 0.5);
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(21, 13, 0.3);
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(500, 500, 0.15);
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(400, 800, 0.10);
      verify_threaded_spmv<SparseMatrix<Scalar, RowMajor>, Vec>(1500, 1500, 0.02);
    }
  }
}

// The partition targets are floor(t * nnz / chunks), computed without forming the product t * nnz, which can overflow
// Index. Compare with the product formed in long long, on an outer index array with one entry per outer vector.
void verify_nnz_balanced_partition() {
  for (Index totalNnz : {Index(1), Index(7), Index(1000), Index(123457), Index(1) << 30}) {
    for (int chunks : {2, 3, 7, 64, 1000, 46341}) {
      // outer[i] = i * totalNnz / outerSize: the entries are spread evenly over the outer vectors.
      const Index outerSize = numext::mini<Index>(totalNnz, 200000);
      std::vector<long long> outer(outerSize + 1);
      for (Index i = 0; i <= outerSize; ++i) outer[i] = ((long long)i * totalNnz) / outerSize;
      std::vector<Index> boundaries;
      internal::compute_nnz_balanced_partition(outer.data(), outerSize, totalNnz, chunks, boundaries);
      VERIFY_IS_EQUAL(Index(boundaries.size()), Index(chunks) + 1);
      VERIFY_IS_EQUAL(boundaries[0], Index(0));
      VERIFY_IS_EQUAL(boundaries[chunks], outerSize);
      for (int t = 1; t < chunks; ++t) {
        const long long target = ((long long)t * totalNnz) / chunks;
        const Index expected = std::lower_bound(outer.begin(), outer.end(), target) - outer.begin();
        VERIFY_IS_EQUAL(boundaries[t], expected);
      }
    }
  }
}

EIGEN_DECLARE_TEST(sparse_threaded_product) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(run_grid<float>());
    CALL_SUBTEST_2(run_grid<double>());
    CALL_SUBTEST_3(run_grid<std::complex<double> >());
    CALL_SUBTEST_4(verify_threaded_scatter_ieee754<double>());
    CALL_SUBTEST_4(verify_threaded_scatter_ieee754<float>());
    CALL_SUBTEST_4(verify_threaded_scatter_overaligned<OverAlignedScalar>());
    CALL_SUBTEST_4(verify_threaded_scatter_overaligned<WiderAlignedScalar>());
    CALL_SUBTEST_5(verify_nnz_balanced_partition());
  }
}

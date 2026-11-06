// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "lapack_common.h"
#include <Eigen/Eigenvalues>

namespace {

// Returns INFO. LWORK must be at least max(1, k*N - 1), the minimum of the reference xSYEV (k = 3) and xHEEV (k = 2),
// and a workspace query returns that bound. Eigen allocates its own workspace, so WORK is referenced only for WORK(1).
EIGEN_BLAS_INT check_ev_args(char jobz, char uplo, EIGEN_BLAS_INT n, EIGEN_BLAS_INT lda, EIGEN_BLAS_INT lwork,
                             EIGEN_BLAS_INT k) {
  if (jobz != 'N' && jobz != 'V') return -1;
  if (UPLO(uplo) == INVALID) return -2;
  if (n < 0) return -3;
  if (lda < std::max<EIGEN_BLAS_INT>(1, n)) return -5;
  if (lwork != -1 && lwork < std::max<EIGEN_BLAS_INT>(1, k * n - 1)) return -8;
  return 0;
}

// Returns INFO; INFO = 1 on NoConvergence stands for "at least one", where the reference counts the unconverged
// off-diagonals. Reads only the UPLO triangle of A, whose diagonal must be real. For N = 1 the reference returns
// W(1) = real(A(1,1)) with INFO = 0 even when it is not finite, which SelfAdjointEigenSolver reports as NoConvergence.
EIGEN_BLAS_INT solve_selfadjoint_ev(bool compute_vectors, char uplo, EIGEN_BLAS_INT n, Scalar* a, EIGEN_BLAS_INT lda,
                                    RealScalar* w) {
  if (n == 0) return 0;
  if (n == 1) {
    w[0] = Eigen::numext::real(a[0]);
    if (compute_vectors) a[0] = Scalar(1);
    return 0;
  }

  Eigen::SelfAdjointEigenSolver<PlainMatrixType> eig(n);
  const EIGEN_BLAS_INT options = compute_vectors ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly;
  if (UPLO(uplo) == UP)
    eig.compute(matrix(a, n, n, lda).adjoint(), options);
  else
    eig.compute(matrix(a, n, n, lda), options);

  if (eig.info() == Eigen::NoConvergence) {
    make_vector(w, n).setZero();
    if (compute_vectors) matrix(a, n, n, lda).setIdentity();
    return 1;
  }

  make_vector(w, n) = eig.eigenvalues();
  if (compute_vectors) matrix(a, n, n, lda) = eig.eigenvectors();
  return 0;
}

}  // namespace

#if !ISCOMPLEX

EIGEN_LAPACK_FUNC(syev)
(const char* jobz, const char* uplo, EIGEN_BLAS_INT* n, RealScalar* a, EIGEN_BLAS_INT* lda, RealScalar* w,
 RealScalar* work, EIGEN_BLAS_INT* lwork, EIGEN_BLAS_INT* info) {
  const char jobz_upper = static_cast<char>(std::toupper(static_cast<unsigned char>(*jobz)));
  *info = check_ev_args(jobz_upper, *uplo, *n, *lda, *lwork, 3);
  // As in the reference, WORK(1) is set once every argument before LWORK checks out.
  if (*info == 0 || *info == -8) work[0] = roundup_lwork(std::max<EIGEN_BLAS_INT>(1, 3 * *n - 1));
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "SYEV ", &e, kBlasNameLength);
  }
  if (*lwork == -1) return;

  *info = solve_selfadjoint_ev(jobz_upper == 'V', *uplo, *n, a, *lda, w);
}

#else

EIGEN_LAPACK_FUNC(heev)
(const char* jobz, const char* uplo, EIGEN_BLAS_INT* n, RealScalar* ra, EIGEN_BLAS_INT* lda, RealScalar* w,
 RealScalar* work, EIGEN_BLAS_INT* lwork, RealScalar* /*rwork*/, EIGEN_BLAS_INT* info) {
  const char jobz_upper = static_cast<char>(std::toupper(static_cast<unsigned char>(*jobz)));
  *info = check_ev_args(jobz_upper, *uplo, *n, *lda, *lwork, 2);
  if (*info == 0 || *info == -8)
    *reinterpret_cast<Scalar*>(work) = Scalar(roundup_lwork(std::max<EIGEN_BLAS_INT>(1, 2 * *n - 1)));
  if (*info != 0) {
    EIGEN_BLAS_INT e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "HEEV ", &e, kBlasNameLength);
  }
  if (*lwork == -1) return;

  // xHETRD replaces A(i,i) by real(A(i,i)). SelfAdjointEigenSolver ignores imag(A(i,i)) except in its scaling, where a
  // NaN or huge imaginary part would corrupt the result, so it is dropped here first; the reference destroys A anyway.
  Scalar* a = reinterpret_cast<Scalar*>(ra);
  for (EIGEN_BLAS_INT i = 0; i < *n; ++i) a[i + i * *lda] = Scalar(Eigen::numext::real(a[i + i * *lda]));
  *info = solve_selfadjoint_ev(jobz_upper == 'V', *uplo, *n, a, *lda, w);
}

#endif  // !ISCOMPLEX

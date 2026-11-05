// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "lapack_common.h"
#include <Eigen/SVD>

#if ISCOMPLEX
#define EIGEN_LAPACK_ARG_IF_COMPLEX(X) X,
#else
#define EIGEN_LAPACK_ARG_IF_COMPLEX(X)
#endif

// GESDD computes the singular values and vectors of a general M-by-N matrix A by divide and conquer.
EIGEN_LAPACK_FUNC(gesdd)
(const char *jobz, int *m, int *n, RealScalar *ra, int *lda, RealScalar *s, RealScalar *ru, int *ldu, RealScalar *rvt,
 int *ldvt, RealScalar *work, int *lwork, EIGEN_LAPACK_ARG_IF_COMPLEX(RealScalar * /*rwork*/) int * /*iwork*/,
 int *info) {
  // TODO exploit the work buffer
  bool query_size = *lwork == -1;
  const char jobz_upper = static_cast<char>(std::toupper(static_cast<unsigned char>(*jobz)));
  jobz = &jobz_upper;
  int diag_size = (std::min)(*m, *n);

  *info = 0;
  if (*jobz != 'A' && *jobz != 'S' && *jobz != 'O' && *jobz != 'N')
    *info = -1;
  else if (*m < 0)
    *info = -2;
  else if (*n < 0)
    *info = -3;
  else if (*lda < std::max(1, *m))
    *info = -5;
  else if (*ldu < 1 || ((*jobz == 'A' || *jobz == 'S') && *ldu < *m) || (*jobz == 'O' && *m < *n && *ldu < *m))
    *info = -8;
  else if (*ldvt < 1 || (*jobz == 'A' && *ldvt < *n) || (*jobz == 'S' && *ldvt < diag_size) ||
           (*jobz == 'O' && *m >= *n && *ldvt < *n))
    *info = -10;

  if (*info != 0) {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "GESDD ", &e, kBlasNameLength);
  }

  if (query_size) {
    work[0] = RealScalar(1);
    return;
  }

  if (*n == 0 || *m == 0) return;

  Scalar *a = reinterpret_cast<Scalar *>(ra);
  Scalar *u = reinterpret_cast<Scalar *>(ru);
  Scalar *vt = reinterpret_cast<Scalar *>(rvt);

  PlainMatrixType mat = matrix(a, *m, *n, *lda);

  // As reference xGESDD documents: INFO = -4 if A has a NaN entry, without a call to XERBLA.
  if (mat.hasNaN()) {
    *info = -4;
    return;
  }

  if (*jobz == 'A') {
    Eigen::BDCSVD<PlainMatrixType, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(mat);
    make_vector(s, diag_size) = svd.singularValues().head(diag_size);
    matrix(u, *m, *m, *ldu) = svd.matrixU();
    matrix(vt, *n, *n, *ldvt) = svd.matrixV().adjoint();
  } else if (*jobz == 'N') {
    Eigen::BDCSVD<PlainMatrixType> svd(mat);
    make_vector(s, diag_size) = svd.singularValues().head(diag_size);
  } else {
    // JOBZ = 'S' or 'O'; 'O' overwrites A with U if M >= N and with V^H otherwise.
    Eigen::BDCSVD<PlainMatrixType, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(mat);
    make_vector(s, diag_size) = svd.singularValues().head(diag_size);
    if (*jobz == 'S') {
      matrix(u, *m, diag_size, *ldu) = svd.matrixU();
      matrix(vt, diag_size, *n, *ldvt) = svd.matrixV().adjoint();
    } else if (*m >= *n) {
      matrix(a, *m, *n, *lda) = svd.matrixU();
      matrix(vt, *n, *n, *ldvt) = svd.matrixV().adjoint();
    } else {
      matrix(u, *m, *m, *ldu) = svd.matrixU();
      matrix(a, diag_size, *n, *lda) = svd.matrixV().adjoint();
    }
  }
}

// GESVD computes the singular values and vectors of a general M-by-N matrix A by the two-sided Jacobi algorithm.
EIGEN_LAPACK_FUNC(gesvd)
(const char *jobu, const char *jobv, int *m, int *n, RealScalar *ra, int *lda, RealScalar *s, RealScalar *ru, int *ldu,
 RealScalar *rvt, int *ldvt, RealScalar *work, int *lwork,
 EIGEN_LAPACK_ARG_IF_COMPLEX(RealScalar * /*rwork*/) int *info) {
  // TODO exploit the work buffer
  bool query_size = *lwork == -1;
  const char jobu_upper = static_cast<char>(std::toupper(static_cast<unsigned char>(*jobu)));
  jobu = &jobu_upper;
  const char jobv_upper = static_cast<char>(std::toupper(static_cast<unsigned char>(*jobv)));
  jobv = &jobv_upper;
  int diag_size = (std::min)(*m, *n);

  *info = 0;
  if (*jobu != 'A' && *jobu != 'S' && *jobu != 'O' && *jobu != 'N')
    *info = -1;
  else if ((*jobv != 'A' && *jobv != 'S' && *jobv != 'O' && *jobv != 'N') || (*jobu == 'O' && *jobv == 'O'))
    *info = -2;
  else if (*m < 0)
    *info = -3;
  else if (*n < 0)
    *info = -4;
  else if (*lda < std::max(1, *m))
    *info = -6;
  else if (*ldu < 1 || ((*jobu == 'A' || *jobu == 'S') && *ldu < *m))
    *info = -9;
  else if (*ldvt < 1 || (*jobv == 'A' && *ldvt < *n) || (*jobv == 'S' && *ldvt < diag_size))
    *info = -11;

  if (*info != 0) {
    int e = -*info;
    return xerbla_(SCALAR_SUFFIX_UP "GESVD ", &e, kBlasNameLength);
  }

  if (query_size) {
    work[0] = RealScalar(1);
    return;
  }

  if (*n == 0 || *m == 0) return;

  Scalar *a = reinterpret_cast<Scalar *>(ra);
  Scalar *u = reinterpret_cast<Scalar *>(ru);
  Scalar *vt = reinterpret_cast<Scalar *>(rvt);

  PlainMatrixType mat = matrix(a, *m, *n, *lda);

  int option = 0;
  if (*jobu == 'A')
    option |= Eigen::ComputeFullU;
  else if (*jobu == 'S' || *jobu == 'O')
    option |= Eigen::ComputeThinU;
  if (*jobv == 'A')
    option |= Eigen::ComputeFullV;
  else if (*jobv == 'S' || *jobv == 'O')
    option |= Eigen::ComputeThinV;

  Eigen::JacobiSVD<PlainMatrixType> svd(mat, option);

  make_vector(s, diag_size) = svd.singularValues().head(diag_size);
  if (*jobu == 'A')
    matrix(u, *m, *m, *ldu) = svd.matrixU();
  else if (*jobu == 'S')
    matrix(u, *m, diag_size, *ldu) = svd.matrixU();
  else if (*jobu == 'O')
    matrix(a, *m, diag_size, *lda) = svd.matrixU();
  if (*jobv == 'A')
    matrix(vt, *n, *n, *ldvt) = svd.matrixV().adjoint();
  else if (*jobv == 'S')
    matrix(vt, diag_size, *n, *ldvt) = svd.matrixV().adjoint();
  else if (*jobv == 'O')
    matrix(a, diag_size, *n, *lda) = svd.matrixV().adjoint();
}

#undef EIGEN_LAPACK_ARG_IF_COMPLEX

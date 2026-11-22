// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Stand-in for a vendor Fortran BLAS header where Accelerate is unavailable; declared only, never called.
#ifndef EIGEN_TEST_BLAS_VENDOR_HEADER_H
#define EIGEN_TEST_BLAS_VENDOR_HEADER_H

#include <complex>

#if defined(ACCELERATE_LAPACK_ILP64)
using blas_vendor_int = long;
#else
using blas_vendor_int = int;
#endif

extern "C" {
#if !defined(ACCELERATE_NEW_LAPACK)
// Legacy form: int result, no const, void* for complex.
int dgemm_(char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);
int zgemm_(char*, char*, int*, int*, int*, void*, void*, int*, void*, int*, void*, void*, int*);
int ztrsm_(char*, char*, char*, char*, int*, int*, void*, void*, int*, void*, int*);
#else
// ACCELERATE_NEW_LAPACK form: const inputs, std::complex* for complex, blas_vendor_int for integers.
void dgemm_(const char*, const char*, const blas_vendor_int*, const blas_vendor_int*, const blas_vendor_int*,
            const double*, const double*, const blas_vendor_int*, const double*, const blas_vendor_int*, const double*,
            double*, const blas_vendor_int*);
void zgemm_(const char*, const char*, const blas_vendor_int*, const blas_vendor_int*, const blas_vendor_int*,
            const std::complex<double>*, const std::complex<double>*, const blas_vendor_int*,
            const std::complex<double>*, const blas_vendor_int*, const std::complex<double>*, std::complex<double>*,
            const blas_vendor_int*);
void ztrsm_(const char*, const char*, const char*, const char*, const blas_vendor_int*, const blas_vendor_int*,
            const std::complex<double>*, const std::complex<double>*, const blas_vendor_int*, std::complex<double>*,
            const blas_vendor_int*);
#endif
}

#endif  // EIGEN_TEST_BLAS_VENDOR_HEADER_H

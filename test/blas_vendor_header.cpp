// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// #2420: parts 1-2 legacy vendor prototypes, 3-4 ACCELERATE_NEW_LAPACK ones (4: ILP64); odd parts include them first.
#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif
// Declares Eigen's sbgemm_ prototype; no bfloat16 product is formed, so the test needs no sbgemm_ to link.
#ifndef EIGEN_USE_OPENBLAS_BFLOAT16
#define EIGEN_USE_OPENBLAS_BFLOAT16 1
#endif
// OpenBLAS f77blas.h form, which declares sbgemm_ whether or not the library was built with bfloat16.
#define EIGEN_TEST_OPENBLAS_SBGEMM                                                                              \
  extern "C" void sbgemm_(char*, char*, int*, int*, int*, float*, unsigned short*, int*, unsigned short*, int*, \
                          float*, float*, int*);

#if defined(__APPLE__) && defined(__clang__) && defined(__has_include)
#if __has_include(<Accelerate/Accelerate.h>)
#define EIGEN_TEST_ACCELERATE
#endif
#endif

#if defined(EIGEN_TEST_PART_3) || defined(EIGEN_TEST_PART_4)
#define ACCELERATE_NEW_LAPACK
#endif
#if defined(EIGEN_TEST_PART_4)
#define ACCELERATE_LAPACK_ILP64
#endif

#ifdef EIGEN_TEST_ACCELERATE
#define EIGEN_TEST_VENDOR_HEADER <Accelerate/Accelerate.h>
#else
#define EIGEN_TEST_VENDOR_HEADER "blas_vendor_header.h"
#endif

#if defined(EIGEN_TEST_PART_1) || defined(EIGEN_TEST_PART_3)
#include EIGEN_TEST_VENDOR_HEADER
EIGEN_TEST_OPENBLAS_SBGEMM
#endif

#include "main.h"

#if defined(EIGEN_TEST_PART_2) || defined(EIGEN_TEST_PART_4)
// main.h defines DEBUG empty, which CoreServices (pulled in by Accelerate) tests with #if.
#pragma push_macro("DEBUG")
#undef DEBUG
#include EIGEN_TEST_VENDOR_HEADER
#pragma pop_macro("DEBUG")
EIGEN_TEST_OPENBLAS_SBGEMM
#endif

// Integer entries in [-3, 3] keep every sum exact, so the ?gemm-backed product must equal the coefficient-based one.
template <typename Scalar>
void blas_vendor_header() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  using RealScalar = typename NumTraits<Scalar>::Real;
  // Above EIGEN_GEMM_TO_COEFFBASED_THRESHOLD, so the products reach the BLAS adapters.
  const Index size = internal::random<Index>(24, 64);
  Mat a = RandomIntegerMatrix<Mat>(size, size, -3, 3);
  const Mat b = RandomIntegerMatrix<Mat>(size, size, -3, 3);
  VERIFY_IS_CWISE_EQUAL(Mat(a * b), Mat(a.lazyProduct(b)));
  // ?trsm, on a diagonally dominant factor
  a.diagonal().setConstant(Scalar(RealScalar(4 * size)));
  const Mat x = a.template triangularView<Lower>().solve(b);
  VERIFY_IS_APPROX(Mat(a.template triangularView<Lower>() * x), b);
}

EIGEN_DECLARE_TEST(blas_vendor_header) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(blas_vendor_header<double>());
    CALL_SUBTEST_1(blas_vendor_header<std::complex<double>>());
    CALL_SUBTEST_2(blas_vendor_header<double>());
    CALL_SUBTEST_2(blas_vendor_header<std::complex<double>>());
    CALL_SUBTEST_3(blas_vendor_header<double>());
    CALL_SUBTEST_3(blas_vendor_header<std::complex<double>>());
    CALL_SUBTEST_4(blas_vendor_header<double>());
    CALL_SUBTEST_4(blas_vendor_header<std::complex<double>>());
  }
}

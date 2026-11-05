// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef LAPACK_H
#define LAPACK_H

#include "../blas/blas.h"

#if defined(_WIN32)
#if defined(EIGEN_LAPACK_BUILD_DLL)
#define EIGEN_LAPACK_API __declspec(dllexport)
#elif defined(EIGEN_LAPACK_LINK_DLL)
#define EIGEN_LAPACK_API __declspec(dllimport)
#else
#define EIGEN_LAPACK_API
#endif
#elif ((defined(__GNUC__) && __GNUC__ >= 4) || defined(__clang__)) && defined(EIGEN_LAPACK_BUILD_DLL)
#define EIGEN_LAPACK_API __attribute__((visibility("default")))
#else
#define EIGEN_LAPACK_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float r, i;
} eigen_lapack_complex_float;

typedef struct {
  double r, i;
} eigen_lapack_complex_double;

// Exactly the routines this library defines, each also listed in eigen_lapack.def. EIGEN_LAPACK_API on these
// declarations exports the definitions from the shared library.

// Cholesky.
EIGEN_LAPACK_API void BLASFUNC(spotrf)(const char *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dpotrf)(const char *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(cpotrf)(const char *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zpotrf)(const char *, int *, double *, int *, int *);

EIGEN_LAPACK_API void BLASFUNC(spotrs)(const char *, int *, int *, float *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dpotrs)(const char *, int *, int *, double *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(cpotrs)(const char *, int *, int *, float *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zpotrs)(const char *, int *, int *, double *, int *, double *, int *, int *);

EIGEN_LAPACK_API void BLASFUNC(spstrf)(const char *, int *, float *, int *, int *, int *, float *, float *, int *);
EIGEN_LAPACK_API void BLASFUNC(dpstrf)(const char *, int *, double *, int *, int *, int *, double *, double *, int *);
EIGEN_LAPACK_API void BLASFUNC(cpstrf)(const char *, int *, float *, int *, int *, int *, float *, float *, int *);
EIGEN_LAPACK_API void BLASFUNC(zpstrf)(const char *, int *, double *, int *, int *, int *, double *, double *, int *);

// Symmetric and Hermitian indefinite (Bunch-Kaufman).
EIGEN_LAPACK_API void BLASFUNC(ssytrf)(const char *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dsytrf)(const char *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(chetrf)(const char *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zhetrf)(const char *, int *, double *, int *, int *, double *, int *, int *);

EIGEN_LAPACK_API void BLASFUNC(ssytrs)(const char *, int *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dsytrs)(const char *, int *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(chetrs)(const char *, int *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zhetrs)(const char *, int *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(spotri)(const char *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dpotri)(const char *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(cpotri)(const char *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zpotri)(const char *, int *, double *, int *, int *);

// Eigenvalues.
EIGEN_LAPACK_API void BLASFUNC(ssyev)(const char *, const char *, int *, float *, int *, float *, float *, int *,
                                      int *);
EIGEN_LAPACK_API void BLASFUNC(dsyev)(const char *, const char *, int *, double *, int *, double *, double *, int *,
                                      int *);

// LU.
EIGEN_LAPACK_API void BLASFUNC(sgetrf)(int *, int *, float *, int *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dgetrf)(int *, int *, double *, int *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(cgetrf)(int *, int *, float *, int *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zgetrf)(int *, int *, double *, int *, int *, int *);

EIGEN_LAPACK_API void BLASFUNC(sgetrs)(const char *, int *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dgetrs)(const char *, int *, int *, double *, int *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(cgetrs)(const char *, int *, int *, float *, int *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zgetrs)(const char *, int *, int *, double *, int *, int *, double *, int *, int *);

EIGEN_LAPACK_API void BLASFUNC(sgetri)(int *, float *, int *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dgetri)(int *, double *, int *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(cgetri)(int *, float *, int *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zgetri)(int *, double *, int *, int *, double *, int *, int *);

// Triangular.
EIGEN_LAPACK_API void BLASFUNC(strtri)(const char *, const char *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dtrtri)(const char *, const char *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(ctrtri)(const char *, const char *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(ztrtri)(const char *, const char *, int *, double *, int *, int *);

EIGEN_LAPACK_API void BLASFUNC(slauum)(const char *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dlauum)(const char *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(clauum)(const char *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zlauum)(const char *, int *, double *, int *, int *);

EIGEN_LAPACK_API void BLASFUNC(strtrs)(const char *, const char *, const char *, int *, int *, float *, int *, float *,
                                       int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dtrtrs)(const char *, const char *, const char *, int *, int *, double *, int *,
                                       double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(ctrtrs)(const char *, const char *, const char *, int *, int *, float *, int *, float *,
                                       int *, int *);
EIGEN_LAPACK_API void BLASFUNC(ztrtrs)(const char *, const char *, const char *, int *, int *, double *, int *,
                                       double *, int *, int *);

// SVD.
EIGEN_LAPACK_API void BLASFUNC(sgesdd)(const char *, int *, int *, float *, int *, float *, float *, int *, float *,
                                       int *, float *, int *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dgesdd)(const char *, int *, int *, double *, int *, double *, double *, int *, double *,
                                       int *, double *, int *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(cgesdd)(const char *, int *, int *, float *, int *, float *, float *, int *, float *,
                                       int *, float *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(zgesdd)(const char *, int *, int *, double *, int *, double *, double *, int *, double *,
                                       int *, double *, int *, double *, int *, int *);

EIGEN_LAPACK_API void BLASFUNC(sgesvd)(const char *, const char *, int *, int *, float *, int *, float *, float *,
                                       int *, float *, int *, float *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(dgesvd)(const char *, const char *, int *, int *, double *, int *, double *, double *,
                                       int *, double *, int *, double *, int *, int *);
EIGEN_LAPACK_API void BLASFUNC(cgesvd)(const char *, const char *, int *, int *, float *, int *, float *, float *,
                                       int *, float *, int *, float *, int *, float *, int *);
EIGEN_LAPACK_API void BLASFUNC(zgesvd)(const char *, const char *, int *, int *, double *, int *, double *, double *,
                                       int *, double *, int *, double *, int *, double *, int *);

// Time.
EIGEN_LAPACK_API float BLASFUNC(second)();
EIGEN_LAPACK_API double BLASFUNC(dsecnd)();

// Householder.
EIGEN_LAPACK_API void BLASFUNC(slarfg)(int *, float *, float *, int *, float *);
EIGEN_LAPACK_API void BLASFUNC(dlarfg)(int *, double *, double *, int *, double *);
EIGEN_LAPACK_API void BLASFUNC(clarfg)(int *, float *, float *, int *, float *);
EIGEN_LAPACK_API void BLASFUNC(zlarfg)(int *, double *, double *, int *, double *);

EIGEN_LAPACK_API void BLASFUNC(slarf)(const char *, int *, int *, float *, int *, float *, float *, int *, float *);
EIGEN_LAPACK_API void BLASFUNC(dlarf)(const char *, int *, int *, double *, int *, double *, double *, int *, double *);
EIGEN_LAPACK_API void BLASFUNC(clarf)(const char *, int *, int *, float *, int *, float *, float *, int *, float *);
EIGEN_LAPACK_API void BLASFUNC(zlarf)(const char *, int *, int *, double *, int *, double *, double *, int *, double *);

EIGEN_LAPACK_API void BLASFUNC(slarft)(const char *, const char *, int *, int *, float *, int *, float *, float *,
                                       int *);
EIGEN_LAPACK_API void BLASFUNC(dlarft)(const char *, const char *, int *, int *, double *, int *, double *, double *,
                                       int *);
EIGEN_LAPACK_API void BLASFUNC(clarft)(const char *, const char *, int *, int *, float *, int *, float *, float *,
                                       int *);
EIGEN_LAPACK_API void BLASFUNC(zlarft)(const char *, const char *, int *, int *, double *, int *, double *, double *,
                                       int *);

EIGEN_LAPACK_API void BLASFUNC(slarfb)(const char *, const char *, const char *, const char *, int *, int *, int *,
                                       float *, int *, float *, int *, float *, int *, float *, int *);
EIGEN_LAPACK_API void BLASFUNC(dlarfb)(const char *, const char *, const char *, const char *, int *, int *, int *,
                                       double *, int *, double *, int *, double *, int *, double *, int *);
EIGEN_LAPACK_API void BLASFUNC(clarfb)(const char *, const char *, const char *, const char *, int *, int *, int *,
                                       float *, int *, float *, int *, float *, int *, float *, int *);
EIGEN_LAPACK_API void BLASFUNC(zlarfb)(const char *, const char *, const char *, const char *, int *, int *, int *,
                                       double *, int *, double *, int *, double *, int *, double *, int *);

// Utilities.
EIGEN_LAPACK_API float BLASFUNC(slapy2)(float *, float *);
EIGEN_LAPACK_API double BLASFUNC(dlapy2)(double *, double *);

EIGEN_LAPACK_API float BLASFUNC(slapy3)(float *, float *, float *);
EIGEN_LAPACK_API double BLASFUNC(dlapy3)(double *, double *, double *);

EIGEN_LAPACK_API void BLASFUNC(clacgv)(int *, float *, int *);
EIGEN_LAPACK_API void BLASFUNC(zlacgv)(int *, double *, int *);

EIGEN_LAPACK_API float BLASFUNC(slamch)(const char *);
EIGEN_LAPACK_API double BLASFUNC(dlamch)(const char *);
EIGEN_LAPACK_API float BLASFUNC(slamc3)(float *, float *);
EIGEN_LAPACK_API double BLASFUNC(dlamc3)(double *, double *);

EIGEN_LAPACK_API void BLASFUNC(sladiv)(const float *, const float *, const float *, const float *, float *, float *);
EIGEN_LAPACK_API void BLASFUNC(dladiv)(const double *, const double *, const double *, const double *, double *,
                                       double *);

// Note: Returning complex structs by value adheres to the SysV AMD64 ABI (gfortran
// on Linux/x86_64), consistent with the return-by-value convention used for cdotc/zdotc
// in blas/complexdots.cpp.
EIGEN_LAPACK_API eigen_lapack_complex_float BLASFUNC(cladiv)(const float *, const float *);
EIGEN_LAPACK_API eigen_lapack_complex_double BLASFUNC(zladiv)(const double *, const double *);

EIGEN_LAPACK_API int BLASFUNC(ilaslc)(int *, int *, float *, int *);
EIGEN_LAPACK_API int BLASFUNC(iladlc)(int *, int *, double *, int *);
EIGEN_LAPACK_API int BLASFUNC(ilaclc)(int *, int *, float *, int *);
EIGEN_LAPACK_API int BLASFUNC(ilazlc)(int *, int *, double *, int *);

EIGEN_LAPACK_API int BLASFUNC(ilaslr)(int *, int *, float *, int *);
EIGEN_LAPACK_API int BLASFUNC(iladlr)(int *, int *, double *, int *);
EIGEN_LAPACK_API int BLASFUNC(ilaclr)(int *, int *, float *, int *);
EIGEN_LAPACK_API int BLASFUNC(ilazlr)(int *, int *, double *, int *);

EIGEN_LAPACK_API void BLASFUNC(ilaver)(int *, int *, int *);

#ifdef __cplusplus
}
#endif

#endif

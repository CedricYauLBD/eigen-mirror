// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_MISC_BLAS_H
#define EIGEN_MISC_BLAS_H

/* EIGEN_BLAS_SYM applies EIGEN_BLAS_SYMBOL_SUFFIX; both come from Core/util/MKL_support.h, this header's only
 * includer. The declarations here must stay in step with the BLAS call sites under Core/products.
 */

// Asm-labelled C++ declarations in Eigen::internal cannot conflict with differently typed vendor prototypes (#2420).
#if (EIGEN_COMP_GNUC || EIGEN_COMP_CLANG) && defined(__USER_LABEL_PREFIX__)
#define EIGEN_BLAS_DECL(FUNC, PARAMS) \
  void EIGEN_BLAS_SYM(FUNC)           \
      PARAMS __asm__(EIGEN_MAKESTRING(__USER_LABEL_PREFIX__) EIGEN_MAKESTRING(EIGEN_BLAS_SYM(FUNC)))
#else
#define EIGEN_BLAS_DECL(FUNC, PARAMS) extern "C" void EIGEN_BLAS_SYM(FUNC) PARAMS
#endif

namespace Eigen {
namespace internal {

/* Level 1 routines */

EIGEN_BLAS_DECL(saxpy, (const EIGEN_BLAS_INT*, const float*, const float*, const EIGEN_BLAS_INT*, float*,
                        const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(daxpy, (const EIGEN_BLAS_INT*, const double*, const double*, const EIGEN_BLAS_INT*, double*,
                        const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(caxpy, (const EIGEN_BLAS_INT*, const float*, const float*, const EIGEN_BLAS_INT*, float*,
                        const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(zaxpy, (const EIGEN_BLAS_INT*, const double*, const double*, const EIGEN_BLAS_INT*, double*,
                        const EIGEN_BLAS_INT*));

/* Level 2 routines */

EIGEN_BLAS_DECL(sgemv, (const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const float*, const float*,
                        const EIGEN_BLAS_INT*, const float*, const EIGEN_BLAS_INT*, const float*, float*,
                        const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(dgemv, (const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const double*, const double*,
                        const EIGEN_BLAS_INT*, const double*, const EIGEN_BLAS_INT*, const double*, double*,
                        const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(cgemv, (const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const float*, const float*,
                        const EIGEN_BLAS_INT*, const float*, const EIGEN_BLAS_INT*, const float*, float*,
                        const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(zgemv, (const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const double*, const double*,
                        const EIGEN_BLAS_INT*, const double*, const EIGEN_BLAS_INT*, const double*, double*,
                        const EIGEN_BLAS_INT*));

EIGEN_BLAS_DECL(strmv, (const char*, const char*, const char*, const EIGEN_BLAS_INT*, const float*,
                        const EIGEN_BLAS_INT*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(dtrmv, (const char*, const char*, const char*, const EIGEN_BLAS_INT*, const double*,
                        const EIGEN_BLAS_INT*, double*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(ctrmv, (const char*, const char*, const char*, const EIGEN_BLAS_INT*, const float*,
                        const EIGEN_BLAS_INT*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(ztrmv, (const char*, const char*, const char*, const EIGEN_BLAS_INT*, const double*,
                        const EIGEN_BLAS_INT*, double*, const EIGEN_BLAS_INT*));

EIGEN_BLAS_DECL(ssymv, (const char*, const EIGEN_BLAS_INT*, const float*, const float*, const EIGEN_BLAS_INT*,
                        const float*, const EIGEN_BLAS_INT*, const float*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(dsymv, (const char*, const EIGEN_BLAS_INT*, const double*, const double*, const EIGEN_BLAS_INT*,
                        const double*, const EIGEN_BLAS_INT*, const double*, double*, const EIGEN_BLAS_INT*));

EIGEN_BLAS_DECL(chemv, (const char*, const EIGEN_BLAS_INT*, const float*, const float*, const EIGEN_BLAS_INT*,
                        const float*, const EIGEN_BLAS_INT*, const float*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(zhemv, (const char*, const EIGEN_BLAS_INT*, const double*, const double*, const EIGEN_BLAS_INT*,
                        const double*, const EIGEN_BLAS_INT*, const double*, double*, const EIGEN_BLAS_INT*));

/* Level 3 routines */

EIGEN_BLAS_DECL(sgemm, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                        const float*, const float*, const EIGEN_BLAS_INT*, const float*, const EIGEN_BLAS_INT*,
                        const float*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(dgemm, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                        const double*, const double*, const EIGEN_BLAS_INT*, const double*, const EIGEN_BLAS_INT*,
                        const double*, double*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(cgemm, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                        const float*, const float*, const EIGEN_BLAS_INT*, const float*, const EIGEN_BLAS_INT*,
                        const float*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(zgemm, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                        const double*, const double*, const EIGEN_BLAS_INT*, const double*, const EIGEN_BLAS_INT*,
                        const double*, double*, const EIGEN_BLAS_INT*));

EIGEN_BLAS_DECL(strsm,
                (const char*, const char*, const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                 const float*, const float*, const EIGEN_BLAS_INT*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(dtrsm,
                (const char*, const char*, const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                 const double*, const double*, const EIGEN_BLAS_INT*, double*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(ctrsm,
                (const char*, const char*, const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                 const float*, const float*, const EIGEN_BLAS_INT*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(ztrsm,
                (const char*, const char*, const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                 const double*, const double*, const EIGEN_BLAS_INT*, double*, const EIGEN_BLAS_INT*));

EIGEN_BLAS_DECL(strmm,
                (const char*, const char*, const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                 const float*, const float*, const EIGEN_BLAS_INT*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(dtrmm,
                (const char*, const char*, const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                 const double*, const double*, const EIGEN_BLAS_INT*, double*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(ctrmm,
                (const char*, const char*, const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                 const float*, const float*, const EIGEN_BLAS_INT*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(ztrmm,
                (const char*, const char*, const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*,
                 const double*, const double*, const EIGEN_BLAS_INT*, double*, const EIGEN_BLAS_INT*));

EIGEN_BLAS_DECL(ssymm, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const float*,
                        const float*, const EIGEN_BLAS_INT*, const float*, const EIGEN_BLAS_INT*, const float*, float*,
                        const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(dsymm, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const double*,
                        const double*, const EIGEN_BLAS_INT*, const double*, const EIGEN_BLAS_INT*, const double*,
                        double*, const EIGEN_BLAS_INT*));

EIGEN_BLAS_DECL(ssyrk, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const float*,
                        const float*, const EIGEN_BLAS_INT*, const float*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(dsyrk, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const double*,
                        const double*, const EIGEN_BLAS_INT*, const double*, double*, const EIGEN_BLAS_INT*));

EIGEN_BLAS_DECL(chemm, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const float*,
                        const float*, const EIGEN_BLAS_INT*, const float*, const EIGEN_BLAS_INT*, const float*, float*,
                        const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(zhemm, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const double*,
                        const double*, const EIGEN_BLAS_INT*, const double*, const EIGEN_BLAS_INT*, const double*,
                        double*, const EIGEN_BLAS_INT*));

EIGEN_BLAS_DECL(cherk, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const float*,
                        const float*, const EIGEN_BLAS_INT*, const float*, float*, const EIGEN_BLAS_INT*));
EIGEN_BLAS_DECL(zherk, (const char*, const char*, const EIGEN_BLAS_INT*, const EIGEN_BLAS_INT*, const double*,
                        const double*, const EIGEN_BLAS_INT*, const double*, double*, const EIGEN_BLAS_INT*));

#undef EIGEN_BLAS_DECL

}  // namespace internal
}  // namespace Eigen

#endif

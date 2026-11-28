// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#include "sparse_solver.h"

#include <Eigen/KLUSupport>

template <typename T>
void test_klu_support_T() {
  KLU<SparseMatrix<T, ColMajor> > klu_colmajor;
  KLU<SparseMatrix<T, RowMajor> > klu_rowmajor;

  check_sparse_square_solving(klu_colmajor);
  check_sparse_square_solving(klu_rowmajor);

  // check_sparse_square_determinant(umfpack_colmajor);
  // check_sparse_square_determinant(umfpack_rowmajor);
}

// klu_analyze() rejects a matrix without entry storage and a 0 x 0 matrix. A solver that has never succeeded must
// report that through info() as well.
template <typename T>
void test_klu_rejected_matrix() {
  using Mat = SparseMatrix<T, ColMajor>;

  KLU<Mat> klu_no_entries;
  check_sparse_zero_matrix(klu_no_entries);

  Mat empty(0, 0);
  KLU<Mat> klu_empty;
  klu_empty.compute(empty);
  VERIFY_IS_EQUAL(klu_empty.info(), NumericalIssue);

  KLU<Mat> klu_constructed(empty);
  VERIFY_IS_EQUAL(klu_constructed.info(), NumericalIssue);

  KLU<Mat> klu_analyzed;
  klu_analyzed.analyzePattern(empty);
  VERIFY_IS_EQUAL(klu_analyzed.info(), InvalidInput);

  // A solver whose factorization failed still refuses to solve.
  Matrix<T, Dynamic, 1> b(0);
  VERIFY_RAISES_ASSERT(klu_empty.solve(b).eval());
}

EIGEN_DECLARE_TEST(klu_support) {
  CALL_SUBTEST_1(test_klu_support_T<double>());
  CALL_SUBTEST_2(test_klu_support_T<std::complex<double> >());
  CALL_SUBTEST_1(test_klu_rejected_matrix<double>());
  CALL_SUBTEST_2(test_klu_rejected_matrix<std::complex<double> >());
}

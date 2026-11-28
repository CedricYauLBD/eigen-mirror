// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_solver.h"
#include <Eigen/SparseLU>

extern "C" {
#include <metis.h>
}

// While this is set, MetisOrdering sees METIS report an error, as METIS does when it cannot allocate its workspace.
static bool g_metis_fails = false;

static int metis_nodend_or_error(idx_t* nvtxs, idx_t* xadj, idx_t* adjncy, idx_t* vwgt, idx_t* options, idx_t* perm,
                                 idx_t* iperm) {
  if (g_metis_fails) return METIS_ERROR_MEMORY;
  return METIS_NodeND(nvtxs, xadj, adjncy, vwgt, options, perm, iperm);
}

// <metis.h> is included already, so this only redirects the call that MetisOrdering makes.
#define METIS_NodeND metis_nodend_or_error

#include <Eigen/MetisSupport>
#include <contrib/Eigen/SparseExtra>

template <typename T>
void test_metis_T() {
  SparseLU<SparseMatrix<T, ColMajor>, MetisOrdering<int> > sparselu_metis;

  check_sparse_square_solving(sparselu_metis);
}

// When METIS fails, MetisOrdering must return the empty permutation (the natural ordering), not leave the
// permutation of an earlier call in place: a solver that is reused for a matrix of another size would apply it.
template <typename T>
void test_metis_failure() {
  using SpMat = SparseMatrix<T, ColMajor>;
  using Vector = Matrix<T, Dynamic, 1>;

  MetisOrdering<int> ordering;
  MetisOrdering<int>::PermutationType perm;
  SparseLU<SpMat, MetisOrdering<int> > sparselu_metis;

  // METIS fails for the second and the fourth matrix: one is smaller than its predecessor, the other larger.
  const int sizes[] = {200, 8, 8, 200};
  for (int k = 0; k < 4; ++k) {
    const int n = sizes[k];
    SpMat A(n, n);
    A.reserve(VectorXi::Constant(n, 3));
    for (int i = 0; i < n; ++i) {
      A.insert(i, i) = T(4);
      if (i > 0) A.insert(i, i - 1) = T(-1);
      if (i + 1 < n) A.insert(i, i + 1) = T(-2);
    }
    A.makeCompressed();
    const Vector b = Vector::Random(n);

    g_metis_fails = (k % 2 == 1);
    ordering(A, perm);
    VERIFY_IS_EQUAL(perm.size(), g_metis_fails ? Index(0) : Index(n));

    sparselu_metis.compute(A);
    VERIFY_IS_EQUAL(sparselu_metis.info(), Success);
    const Vector x = sparselu_metis.solve(b);
    VERIFY_IS_APPROX(A * x, b);
    g_metis_fails = false;
  }
}

EIGEN_DECLARE_TEST(metis_support) {
  CALL_SUBTEST_1(test_metis_T<double>());
  CALL_SUBTEST_2(test_metis_failure<double>());
}

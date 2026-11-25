// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "sparse_solver.h"

template <typename Solver>
void verify_determinant_asserts(const Solver& solver) {
  VERIFY_RAISES_ASSERT(solver.determinant());
  VERIFY_RAISES_ASSERT(solver.absDeterminant());
  VERIFY_RAISES_ASSERT(solver.logAbsDeterminant());
  VERIFY_RAISES_ASSERT(solver.signDeterminant());
}

// factorize_preordered() breaks out of its column loop on a bad pivot but still sets m_factorizationIsOk,
// leaving the tail of D -- and of L's diagonal -- unwritten. The determinant accessors read exactly those,
// so they check info() too, as the dense LLT ones do.
template <typename SparseMatrixType>
void test_simplicial_cholesky_determinant_asserts() {
  using Scalar = typename SparseMatrixType::Scalar;
  const Index n = 6;

  // A zero pivot stops all four classes. A negative one stops the LLT classes for a real scalar; for a complex scalar
  // the non-Hermitian LLT takes a complex square root of it, and only the Hermitian LLT rejects it. The LDLT classes
  // accept it, since D may be indefinite.
  SparseMatrixType zero_pivot(n, n), negative_pivot(n, n);
  for (Index i = 0; i < n; ++i) {
    zero_pivot.insert(i, i) = (i == 0) ? Scalar(0) : Scalar(1);
    negative_pivot.insert(i, i) = (i == n - 1) ? Scalar(-1) : Scalar(1);
  }
  zero_pivot.makeCompressed();
  negative_pivot.makeCompressed();

  SimplicialLLT<SparseMatrixType> llt(zero_pivot);
  VERIFY(llt.info() == NumericalIssue);
  verify_determinant_asserts(llt);

  SimplicialLLT<SparseMatrixType> indefinite_llt(negative_pivot);
  VERIFY(indefinite_llt.info() == NumericalIssue);
  verify_determinant_asserts(indefinite_llt);

  SimplicialLDLT<SparseMatrixType> ldlt(zero_pivot);
  VERIFY(ldlt.info() == NumericalIssue);
  verify_determinant_asserts(ldlt);

  SimplicialNonHermitianLLT<SparseMatrixType> nhllt(zero_pivot);
  VERIFY(nhllt.info() == NumericalIssue);
  verify_determinant_asserts(nhllt);

  SimplicialNonHermitianLDLT<SparseMatrixType> nhldlt(zero_pivot);
  VERIFY(nhldlt.info() == NumericalIssue);
  verify_determinant_asserts(nhldlt);

  SimplicialNonHermitianLLT<SparseMatrixType> nh_indefinite_llt(negative_pivot);
  if (NumTraits<Scalar>::IsComplex) {
    VERIFY(nh_indefinite_llt.info() == Success);
    // L L^T = P A P^T, whichever position the ordering gives the pivot -1.
    using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
    const DenseMatrix L = nh_indefinite_llt.matrixL();
    const DenseMatrix A = DenseMatrix(negative_pivot);
    const DenseMatrix PAPt = nh_indefinite_llt.permutationP() * A * nh_indefinite_llt.permutationP().transpose();
    VERIFY_IS_APPROX(L * L.transpose(), PAPt);
  } else {
    VERIFY(nh_indefinite_llt.info() == NumericalIssue);
    verify_determinant_asserts(nh_indefinite_llt);
  }

  // The deprecated SimplicialCholesky exposes determinant() alone, and reads the same tail.
  SimplicialCholesky<SparseMatrixType> deprecated(zero_pivot);
  VERIFY(deprecated.info() == NumericalIssue);
  VERIFY_RAISES_ASSERT(deprecated.determinant());

  // info() alone would not reject an unfactorized decomposition: it is constructed Success.
  SimplicialLDLT<SparseMatrixType> unfactorized;
  verify_determinant_asserts(unfactorized);
}

template <typename T, typename I_, int flag>
void test_simplicial_cholesky_T() {
  using SparseMatrixType = SparseMatrix<T, flag, I_>;
  SimplicialCholesky<SparseMatrixType, Lower> chol_colmajor_lower_amd;
  SimplicialCholesky<SparseMatrixType, Upper> chol_colmajor_upper_amd;
  SimplicialLLT<SparseMatrixType, Lower> llt_colmajor_lower_amd;
  SimplicialLLT<SparseMatrixType, Upper> llt_colmajor_upper_amd;
  SimplicialLDLT<SparseMatrixType, Lower> ldlt_colmajor_lower_amd;
  SimplicialLDLT<SparseMatrixType, Upper> ldlt_colmajor_upper_amd;
  SimplicialLDLT<SparseMatrixType, Lower, NaturalOrdering<I_> > ldlt_colmajor_lower_nat;
  SimplicialLDLT<SparseMatrixType, Upper, NaturalOrdering<I_> > ldlt_colmajor_upper_nat;
  SimplicialNonHermitianLLT<SparseMatrixType, Lower> nhllt_colmajor_lower_amd;
  SimplicialNonHermitianLLT<SparseMatrixType, Upper> nhllt_colmajor_upper_amd;
  SimplicialNonHermitianLDLT<SparseMatrixType, Lower> nhldlt_colmajor_lower_amd;
  SimplicialNonHermitianLDLT<SparseMatrixType, Upper> nhldlt_colmajor_upper_amd;
  SimplicialNonHermitianLDLT<SparseMatrixType, Lower, NaturalOrdering<I_> > nhldlt_colmajor_lower_nat;
  SimplicialNonHermitianLDLT<SparseMatrixType, Upper, NaturalOrdering<I_> > nhldlt_colmajor_upper_nat;

  check_sparse_spd_solving(chol_colmajor_lower_amd);
  check_sparse_spd_solving(chol_colmajor_upper_amd);
  check_sparse_spd_solving(llt_colmajor_lower_amd);
  check_sparse_spd_solving(llt_colmajor_upper_amd);
  check_sparse_spd_solving(ldlt_colmajor_lower_amd);
  check_sparse_spd_solving(ldlt_colmajor_upper_amd);
  check_sparse_nonhermitian_solving(nhllt_colmajor_lower_amd);
  check_sparse_nonhermitian_solving(nhllt_colmajor_upper_amd);
  check_sparse_nonhermitian_solving(nhldlt_colmajor_lower_amd);
  check_sparse_nonhermitian_solving(nhldlt_colmajor_upper_amd);

  check_sparse_spd_determinant(chol_colmajor_lower_amd);
  check_sparse_spd_determinant(chol_colmajor_upper_amd);
  check_sparse_spd_determinant(llt_colmajor_lower_amd);
  check_sparse_spd_determinant(llt_colmajor_upper_amd);
  check_sparse_spd_determinant(ldlt_colmajor_lower_amd);
  check_sparse_spd_determinant(ldlt_colmajor_upper_amd);
  check_sparse_nonhermitian_determinant(nhllt_colmajor_lower_amd);
  check_sparse_nonhermitian_determinant(nhllt_colmajor_upper_amd);
  check_sparse_nonhermitian_determinant(nhldlt_colmajor_lower_amd);
  check_sparse_nonhermitian_determinant(nhldlt_colmajor_upper_amd);

  check_sparse_spd_log_abs_determinant(llt_colmajor_lower_amd);
  check_sparse_spd_log_abs_determinant(llt_colmajor_upper_amd);
  check_sparse_spd_log_abs_determinant(ldlt_colmajor_lower_amd);
  check_sparse_spd_log_abs_determinant(ldlt_colmajor_upper_amd);
  check_sparse_nonhermitian_log_abs_determinant(nhllt_colmajor_lower_amd);
  check_sparse_nonhermitian_log_abs_determinant(nhllt_colmajor_upper_amd);
  check_sparse_nonhermitian_log_abs_determinant(nhldlt_colmajor_lower_amd);
  check_sparse_nonhermitian_log_abs_determinant(nhldlt_colmajor_upper_amd);

  test_simplicial_cholesky_determinant_asserts<SparseMatrixType>();

  check_sparse_spd_solving(ldlt_colmajor_lower_nat, (std::min)(300, EIGEN_TEST_MAX_SIZE), 1000);
  check_sparse_spd_solving(ldlt_colmajor_upper_nat, (std::min)(300, EIGEN_TEST_MAX_SIZE), 1000);
  check_sparse_nonhermitian_solving(nhldlt_colmajor_lower_nat, (std::min)(300, EIGEN_TEST_MAX_SIZE), 1000);
  check_sparse_nonhermitian_solving(nhldlt_colmajor_upper_nat, (std::min)(300, EIGEN_TEST_MAX_SIZE), 1000);
}

template <typename Scalar>
void test_simplicial_cholesky_scatter(Index bandwidth) {
  using RealScalar = typename NumTraits<Scalar>::Real;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using SparseMatrixType = SparseMatrix<Scalar, ColMajor, long long>;
  const Index n = 67;
  DenseMatrix matrix = DenseMatrix::Zero(n, n);
  for (Index j = 0; j < n; ++j) {
    matrix(j, j) = Scalar(2 * bandwidth + 1);
    for (Index i = j + 1; i < (std::min)(n, j + bandwidth + 1); ++i) {
      matrix(i, j) = Scalar(0.125) + Scalar(0.0625) * internal::random<Scalar>();
      matrix(j, i) = numext::conj(matrix(i, j));
    }
  }
  SparseMatrixType sparse = matrix.sparseView();
  SimplicialLLT<SparseMatrixType, Lower, NaturalOrdering<long long>> solver;
  SimplicialLDLT<SparseMatrixType, Lower, NaturalOrdering<long long>> ldlt;
  solver.analyzePattern(sparse);
  ldlt.analyzePattern(sparse);
  // Reconstruction error scales with the O(n * epsilon) factorization backward error.
  const RealScalar tolerance = RealScalar(16 * n) * NumTraits<RealScalar>::epsilon();
  DenseMatrix lower;
  for (int repeat = 0; repeat < 2; ++repeat) {
    solver.factorize(sparse);
    VERIFY_IS_EQUAL(solver.info(), Success);
    lower = solver.matrixL();
    VERIFY((lower * lower.adjoint() - matrix).norm() <= tolerance * matrix.norm());
    ldlt.factorize(sparse);
    VERIFY_IS_EQUAL(ldlt.info(), Success);
    lower = ldlt.matrixL();
    VERIFY((lower * ldlt.vectorD().asDiagonal() * lower.adjoint() - matrix).norm() <= tolerance * matrix.norm());
    matrix.diagonal().array() += Scalar(1);
    sparse.diagonal().array() += Scalar(1);
  }
}

// The factor, the permutations and the elimination tree number rows and columns with StorageIndex. A size beyond
// its range used to wrap, so that a 300 x 300 matrix with signed char indices got a 44 x 44 factor.
template <template <typename, int, typename> class Solver, int UpLo>
void test_simplicial_cholesky_narrow_index() {
  using SpMat = SparseMatrix<double, ColMajor, signed char>;
  const Index highest = NumTraits<signed char>::highest();
  // Every stored index fits signed char, also when the size does not.
  const auto diagonal = [=](Index n) {
    SpMat A(n, n);
    for (Index i = 0; i < (std::min)(n, highest); ++i) A.insert(i, i) = double(i + 1);
    A.makeCompressed();
    return A;
  };

  Solver<SpMat, UpLo, NaturalOrdering<signed char>> solver(diagonal(highest));
  VERIFY_IS_EQUAL(solver.info(), Success);
  const VectorXd b = VectorXd::LinSpaced(highest, 1, double(highest));
  VERIFY_IS_APPROX(solver.solve(b), VectorXd::Ones(highest));
#ifdef EIGEN_EXCEPTIONS
  for (Index n : {highest + 1, Index(300)}) {
    bool threw = false;
    try {
      solver.compute(diagonal(n));
    } catch (const std::bad_alloc&) {
      threw = true;
    }
    VERIFY(threw);
    // The solver still holds the factorization of the first matrix.
    VERIFY_IS_EQUAL(solver.rows(), highest);
    VERIFY_IS_APPROX(solver.solve(b), VectorXd::Ones(highest));
  }
#endif
}

// In an arrowhead matrix the last unknown is coupled to all the others: the root of the elimination tree has n - 1
// children and L has no fill-in. The analysis must be linear in the number of children of a node. If it is quadratic
// it takes about n^2 / 2 = 5e11 steps here, so a regression shows up as a timeout, not as a failed VERIFY.
// The shared harnesses compare against a dense reference, which does not fit at this size.
template <typename Ordering>
void test_simplicial_cholesky_many_children() {
  using SparseMatrixType = SparseMatrix<double>;
  const int n = 1000000;
  std::vector<Triplet<double>> triplets;
  triplets.reserve(2 * n - 1);
  for (int j = 0; j < n - 1; ++j) {
    triplets.emplace_back(j, j, 2.0);
    triplets.emplace_back(n - 1, j, 1.0);
  }
  triplets.emplace_back(n - 1, n - 1, double(n));
  SparseMatrixType arrowhead(n, n);
  arrowhead.setFromTriplets(triplets.begin(), triplets.end());

  SimplicialLLT<SparseMatrixType, Lower, Ordering> llt(arrowhead);
  VERIFY_IS_EQUAL(llt.info(), Success);
  VERIFY_IS_EQUAL(llt.matrixL().nestedExpression().nonZeros(), arrowhead.nonZeros());
  VectorXd x = VectorXd::Ones(n);
  VectorXd b = arrowhead.selfadjointView<Lower>() * x;
  VERIFY_IS_APPROX(llt.solve(b), x);
}

EIGEN_DECLARE_TEST(simplicial_cholesky) {
  CALL_SUBTEST_31(test_simplicial_cholesky_scatter<float>(3));
  CALL_SUBTEST_31(test_simplicial_cholesky_scatter<float>(37));
  CALL_SUBTEST_32(test_simplicial_cholesky_scatter<std::complex<float>>(3));
  CALL_SUBTEST_32(test_simplicial_cholesky_scatter<std::complex<float>>(37));
  CALL_SUBTEST_31((test_simplicial_cholesky_narrow_index<SimplicialLLT, Lower>()));
  CALL_SUBTEST_31((test_simplicial_cholesky_narrow_index<SimplicialLDLT, Upper>()));
  CALL_SUBTEST_11((test_simplicial_cholesky_T<double, int, ColMajor>()));
  CALL_SUBTEST_11(test_simplicial_cholesky_many_children<AMDOrdering<int>>());
  CALL_SUBTEST_11(test_simplicial_cholesky_many_children<NaturalOrdering<int>>());
  CALL_SUBTEST_12((test_simplicial_cholesky_T<std::complex<double>, int, ColMajor>()));
  CALL_SUBTEST_13((test_simplicial_cholesky_T<double, long int, ColMajor>()));
  CALL_SUBTEST_21((test_simplicial_cholesky_T<double, int, RowMajor>()));
  CALL_SUBTEST_22((test_simplicial_cholesky_T<std::complex<double>, int, RowMajor>()));
  CALL_SUBTEST_23((test_simplicial_cholesky_T<double, long int, RowMajor>()));
}

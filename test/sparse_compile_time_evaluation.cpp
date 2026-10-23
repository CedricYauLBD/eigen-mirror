// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of sparse matrices and the sparse solvers. Like compile_time_evaluation, this test is strictly
// compile-time: main.h adds instrumentation that constant evaluation rejects.

#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>
#include <vector>

using namespace Eigen;

// Sparse storage is allocated on the heap, which constant evaluation allows from C++20.
#if EIGEN_COMP_CXXVER >= 20 && defined(__cpp_lib_constexpr_dynamic_alloc)

// The tridiagonal matrix with 4 on the diagonal and -1 off it, which is symmetric positive definite.
template <typename SparseMatrixType>
constexpr SparseMatrixType laplacian(Index n) {
  std::vector<Triplet<double>> triplets;
  for (Index i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, 4.0);
    if (i > 0) triplets.emplace_back(i, i - 1, -1.0);
    if (i + 1 < n) triplets.emplace_back(i, i + 1, -1.0);
  }
  SparseMatrixType a(n, n);
  a.setFromTriplets(triplets.begin(), triplets.end());
  return a;
}

constexpr bool sparse_basics() {
  SparseMatrix<double> a(3, 3);
  a.insert(0, 0) = 2;
  a.insert(1, 2) = 3;
  a.coeffRef(2, 1) += 4;
  a.makeCompressed();
  bool ok = a.nonZeros() == 3 && a.coeff(1, 2) == 3 && a.coeff(2, 2) == 0 && a.sum() == 9;
  const Vector3d v(1, 2, 3);
  ok = ok && (a * v).isApprox(Vector3d(2, 9, 8));
  const SparseMatrix<double> at = a.transpose();
  ok = ok && at.coeff(2, 1) == 3 && at.coeff(1, 2) == 4;
  const SparseMatrix<double> sum = a + at;
  ok = ok && sum.coeff(1, 2) == 7;
  const SparseMatrix<double> product = a * at;
  ok = ok && MatrixXd(product).isApprox(MatrixXd(a) * MatrixXd(at));
  SparseMatrix<double, RowMajor> row_major = a;
  ok = ok && row_major.coeff(2, 1) == 4;
  SparseVector<double> sv(5);
  sv.insert(3) = 1.5;
  sv.coeffRef(1) = -2;
  ok = ok && sv.nonZeros() == 2 && sv.sum() == -0.5 && sv.dot(sv) == 6.25;
  return ok;
}

static_assert(sparse_basics());

constexpr bool direct_solvers() {
  const SparseMatrix<double> a = laplacian<SparseMatrix<double>>(6);
  const VectorXd b = VectorXd::LinSpaced(6, 1, 6);
  const SimplicialLLT<SparseMatrix<double>> llt(a);
  bool ok = llt.info() == Success && (a * llt.solve(b)).isApprox(b);
  const SimplicialLDLT<SparseMatrix<double>> ldlt(a);
  ok = ok && ldlt.info() == Success && (a * ldlt.solve(b)).isApprox(b);
  SparseLU<SparseMatrix<double>> lu(a);
  ok = ok && lu.info() == Success && (a * lu.solve(b)).isApprox(b);
  SparseMatrix<double> c = a;
  c.makeCompressed();
  const SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> qr(c);
  ok = ok && qr.info() == Success && (a * qr.solve(b)).isApprox(b);
  return ok;
}

static_assert(direct_solvers());

// A dense n x n matrix with a_ij = 1/(1+|i-j|) + n delta_ij, which is symmetric positive definite.
constexpr SparseMatrix<double> dense_spd(Index n) {
  SparseMatrix<double> a(n, n);
  a.reserve(VectorXi::Constant(n, int(n)));
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < n; ++i) a.insert(i, j) = 1.0 / double(1 + (i > j ? i - j : j - i)) + (i == j ? n : 0);
  a.makeCompressed();
  return a;
}

// The first column of the factor has 32 entries below the diagonal, enough for the packet scatter, which constant
// evaluation must skip.
constexpr bool long_columns() {
  const SparseMatrix<double> a = dense_spd(33);
  const VectorXd b = VectorXd::LinSpaced(33, 1, 2);
  const SimplicialLLT<SparseMatrix<double>, Lower, NaturalOrdering<int>> llt(a);
  return llt.info() == Success && (a * llt.solve(b)).isApprox(b);
}

static_assert(long_columns());

// Dense supernodes take SparseLU through its panel and kernel block updates.
constexpr bool supernodes() {
  const SparseMatrix<double> a = dense_spd(12);
  const VectorXd b = VectorXd::LinSpaced(12, 1, 2);
  SparseLU<SparseMatrix<double>, NaturalOrdering<int>> lu(a);
  return lu.info() == Success && (a * lu.solve(b)).isApprox(b);
}

static_assert(supernodes());

// An uncompressed matrix makes Ref<const SparseMatrix, StandardCompressedFormat> hold a compressed copy, and a block
// of rows of a column-major matrix iterates across its outer vectors.
constexpr bool refs_and_blocks() {
  SparseMatrix<double> a = laplacian<SparseMatrix<double>>(5);
  a.uncompress();
  const Ref<const SparseMatrix<double>, StandardCompressedFormat> ref(a);
  bool ok = ref.isCompressed() && ref.nonZeros() == 13 && ref.coeff(3, 2) == -1;
  const SparseMatrix<double, RowMajor> rows = a.middleRows(1, 2);
  ok = ok && rows.nonZeros() == 6 && rows.coeff(0, 1) == 4;
  double sum = 0;
  for (SparseMatrix<double>::InnerIterator it(a, 2); it; ++it) sum += it.value();
  return ok && sum == 2;
}

static_assert(refs_and_blocks());

// Iterators compare by position, and assigning a map rebinds it.
constexpr bool iterator_equality_and_map_assignment() {
  const SparseMatrix<double> a = laplacian<SparseMatrix<double>>(4);
  SparseMatrix<double>::InnerIterator it(a, 1), other(a, 1);
  bool ok = it == other && !(it != other);
  ++other;
  ok = ok && it != other && !(it == other);
  SparseMatrix<double>::ReverseInnerIterator rit(a, 1), rother(a, 1);
  ok = ok && rit == rother;
  --rother;
  ok = ok && rit != rother;

  const SparseMatrix<double> b = 2 * a;
  Map<const SparseMatrix<double>> map(a.rows(), a.cols(), a.nonZeros(), a.outerIndexPtr(), a.innerIndexPtr(),
                                      a.valuePtr());
  const Map<const SparseMatrix<double>> map_b(b.rows(), b.cols(), b.nonZeros(), b.outerIndexPtr(), b.innerIndexPtr(),
                                              b.valuePtr());
  map = map_b;
  return ok && map.valuePtr() == b.valuePtr() && map.coeff(1, 1) == 8;
}

static_assert(iterator_equality_and_map_assignment());

// The sizes keep each evaluation within the compilers' default constexpr step limits.
template <typename Solver, Index n = 8>
constexpr bool iterative_solver() {
  const SparseMatrix<double> a = laplacian<SparseMatrix<double>>(n);
  const VectorXd b = VectorXd::LinSpaced(n, 1, double(n));
  Solver solver;
  solver.setTolerance(1e-12);
  solver.compute(a);
  const VectorXd x = solver.solve(b);
  return solver.info() == Success && (a * x).isApprox(b, 1e-10);
}

static_assert(iterative_solver<ConjugateGradient<SparseMatrix<double>, Lower | Upper>>());
static_assert(iterative_solver<BiCGSTAB<SparseMatrix<double>>>());
static_assert(iterative_solver<LeastSquaresConjugateGradient<SparseMatrix<double>>>());
static_assert(iterative_solver<GMRES<SparseMatrix<double>>>());
static_assert(iterative_solver<DGMRES<SparseMatrix<double>>>());
static_assert(iterative_solver<IDRS<SparseMatrix<double>>, 4>());
static_assert(iterative_solver<IDRSTABL<SparseMatrix<double>>>());
static_assert(iterative_solver<BiCGSTABL<SparseMatrix<double>>>());
static_assert(iterative_solver<MINRES<SparseMatrix<double>, Lower | Upper>>());
static_assert(iterative_solver<BiCGSTAB<SparseMatrix<double>, IncompleteLUT<double>>>());
static_assert(iterative_solver<ConjugateGradient<SparseMatrix<double>, Lower, IncompleteCholesky<double>>>());

#endif

int main() { return 0; }

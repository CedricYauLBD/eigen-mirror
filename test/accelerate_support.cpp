// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#if defined(DEBUG)
#undef DEBUG
#endif

#define EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS

#pragma clang diagnostic push
// The following "warning" causes a build failure on macOS with the latest
// version of clang:
// error: non-defining declaration of enumeration with a fixed underlying
//        type is only permitted as a standalone declaration
#pragma clang diagnostic ignored "-Welaborated-enum-base"
#include <Eigen/AccelerateSupport>
#pragma clang diagnostic pop

#include "sparse_solver.h"

template <typename MatrixType, typename DenseMat>
int generate_sparse_rectangular_problem(MatrixType& A, DenseMat& dA, int maxRows = 300, int maxCols = 300) {
  using Scalar = typename MatrixType::Scalar;
  int rows = internal::random<int>(1, maxRows);
  int cols = internal::random<int>(1, maxCols);
  double density = (std::max)(8.0 / (rows * cols), 0.01);

  A.resize(rows, cols);
  dA.resize(rows, cols);
  initSparse<Scalar>(density, dA, A, ForceNonZeroDiag);
  A.makeCompressed();
  return rows;
}

template <typename MatrixType, typename DenseMat>
int generate_sparse_square_symmetric_problem(MatrixType& A, DenseMat& dA, int maxSize = 300) {
  using Scalar = typename MatrixType::Scalar;
  int rows = internal::random<int>(1, maxSize);
  int cols = rows;
  double density = (std::max)(8.0 / (rows * cols), 0.01);

  A.resize(rows, cols);
  dA.resize(rows, cols);
  initSparse<Scalar>(density, dA, A, ForceNonZeroDiag);
  dA = dA * dA.transpose();
  A = A * A.transpose();
  A.makeCompressed();
  return rows;
}

template <typename Scalar, typename Solver>
void test_accelerate_ldlt() {
  using MatrixType = SparseMatrix<Scalar>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  MatrixType A;
  Matrix<Scalar, Dynamic, Dynamic> dA;

  generate_sparse_square_symmetric_problem(A, dA);

  DenseVector b = DenseVector::Random(A.rows());

  Solver solver;
  solver.compute(A);

  if (solver.info() != Success) {
    std::cerr << "sparse LDLT factorization failed\n";
    exit(0);
    return;
  }

  DenseVector x = solver.solve(b);

  if (solver.info() != Success) {
    std::cerr << "sparse LDLT factorization failed\n";
    exit(0);
    return;
  }

  // Compare with a dense solver
  DenseVector refX = dA.ldlt().solve(b);
  VERIFY((A * x).isApprox(A * refX, test_precision<Scalar>()));
}

template <typename Scalar, typename Solver>
void test_accelerate_llt() {
  using MatrixType = SparseMatrix<Scalar>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  MatrixType A;
  Matrix<Scalar, Dynamic, Dynamic> dA;

  generate_sparse_square_symmetric_problem(A, dA);

  DenseVector b = DenseVector::Random(A.rows());

  Solver solver;
  solver.compute(A);

  if (solver.info() != Success) {
    std::cerr << "sparse LLT factorization failed\n";
    exit(0);
    return;
  }

  DenseVector x = solver.solve(b);

  if (solver.info() != Success) {
    std::cerr << "sparse LLT factorization failed\n";
    exit(0);
    return;
  }

  // Compare with a dense solver
  DenseVector refX = dA.llt().solve(b);
  VERIFY((A * x).isApprox(A * refX, test_precision<Scalar>()));
}

template <typename Scalar, typename Solver>
void test_accelerate_qr() {
  using MatrixType = SparseMatrix<Scalar>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  MatrixType A;
  Matrix<Scalar, Dynamic, Dynamic> dA;

  generate_sparse_rectangular_problem(A, dA);

  DenseVector b = DenseVector::Random(A.rows());

  Solver solver;
  solver.compute(A);

  if (solver.info() != Success) {
    std::cerr << "sparse QR factorization failed\n";
    exit(0);
    return;
  }

  DenseVector x = solver.solve(b);

  if (solver.info() != Success) {
    std::cerr << "sparse QR factorization failed\n";
    exit(0);
    return;
  }

  // Compare with a dense solver
  DenseVector refX = dA.colPivHouseholderQr().solve(b);
  VERIFY((A * x).isApprox(A * refX, test_precision<Scalar>()));
}

// Accelerate must be given the stored entries only, not the free slots of an uncompressed matrix.
template <typename Scalar, typename Solver>
void test_accelerate_uncompressed() {
  using MatrixType = SparseMatrix<Scalar>;
  using DenseVector = Matrix<Scalar, Dynamic, 1>;

  int size = internal::random<int>(1, 300);

  MatrixType A(size, size);
  A.reserve(VectorXi::Constant(size, 5));
  for (int j = 0; j < size; ++j) {
    if (j > 0) A.insert(j - 1, j) = Scalar(-1);
    A.insert(j, j) = Scalar(4);
    if (j + 1 < size) A.insert(j + 1, j) = Scalar(-1);
  }
  VERIFY(!A.isCompressed());

  // Make the free slots look like diagonal entries, which would change the solution.
  for (int j = 0; j < size; ++j) {
    for (int k = A.outerIndexPtr()[j] + A.innerNonZeroPtr()[j]; k < A.outerIndexPtr()[j + 1]; ++k) {
      A.innerIndexPtr()[k] = j;
      A.valuePtr()[k] = Scalar(100);
    }
  }

  DenseVector b = DenseVector::Random(size);

  Solver solver;
  solver.compute(A);
  VERIFY(solver.info() == Success);
  DenseVector x = solver.solve(b);
  VERIFY(solver.info() == Success);
  VERIFY((A * x).isApprox(b, test_precision<Scalar>()));

  Solver solver2;
  solver2.analyzePattern(A);
  VERIFY(solver2.info() == Success);
  solver2.factorize(A);
  VERIFY(solver2.info() == Success);
  x = solver2.solve(b);
  VERIFY(solver2.info() == Success);
  VERIFY((A * x).isApprox(b, test_precision<Scalar>()));
}

// Solves with `view`, an n x nrhs view of `store` that has data(), as the right-hand side and then as the destination.
template <typename Solver, typename View, typename Store, typename DenseMat>
void check_accelerate_view(const Solver& solver, View&& view, Store& store, const DenseMat& B, const DenseMat& refX) {
  using Scalar = typename DenseMat::Scalar;

  view = B;
  const DenseMat X = solver.solve(view);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_APPROX(X, refX);

  // solve() must write the coefficients of the view and no other coefficient of the store.
  store.setConstant(Scalar(7));
  view = solver.solve(B);
  VERIFY_IS_EQUAL(solver.info(), Success);
  VERIFY_IS_APPROX(view, refX);
  view.setConstant(Scalar(7));
  VERIFY((store.array() == Scalar(7)).all());
}

template <typename Scalar, typename Solver>
void test_accelerate_strided(Index nrhs) {
  using MatrixType = SparseMatrix<Scalar>;
  using DenseMat = Matrix<Scalar, Dynamic, Dynamic, ColMajor>;
  using RowMajorMat = Matrix<Scalar, Dynamic, Dynamic, RowMajor>;
  using StridedMap = Map<DenseMat, 0, Stride<Dynamic, Dynamic> >;

  // Tridiagonal and strictly diagonally dominant, so that the factorization cannot fail.
  const Index n = internal::random<Index>(2, 50);
  DenseMat dA = DenseMat::Zero(n, n);
  dA.diagonal().setConstant(Scalar(4));
  dA.diagonal(1).setConstant(Scalar(-1));
  dA.diagonal(-1).setConstant(Scalar(-1));
  const MatrixType A = dA.sparseView();

  Solver solver(A);
  VERIFY_IS_EQUAL(solver.info(), Success);

  const DenseMat B = DenseMat::Random(n, nrhs);
  const DenseMat refX = dA.llt().solve(B);

  DenseMat plain(n, nrhs);
  check_accelerate_view(solver, plain, plain, B, refX);

  // Negative strides: data() is the last coefficient of the first column, or the first one of the last column.
  check_accelerate_view(solver, plain(seqN(n - 1, n, -1), placeholders::all), plain, B, refX);
  check_accelerate_view(solver, plain(placeholders::all, seqN(nrhs - 1, nrhs, -1)), plain, B, refX);

  // An inner stride other than 1, and an outer stride other than the number of rows.
  DenseMat padded(2 * n, 2 * nrhs);
  check_accelerate_view(solver, StridedMap(padded.data(), n, nrhs, Stride<Dynamic, Dynamic>(2 * padded.rows(), 2)),
                        padded, B, refX);
  check_accelerate_view(solver, padded.block(1, 1, n, nrhs), padded, B, refX);

  RowMajorMat row_major(n, nrhs);
  check_accelerate_view(solver, row_major, row_major, B, refX);

  if (nrhs == 1) {
    DenseMat table(3, n);
    check_accelerate_view(solver, table.row(1).transpose(), table, B, refX);
  }
}

template <typename Scalar>
void run_tests() {
  using MatrixType = SparseMatrix<Scalar>;

  test_accelerate_strided<Scalar, AccelerateLLT<MatrixType, Lower> >(1);
  test_accelerate_strided<Scalar, AccelerateLLT<MatrixType, Lower> >(3);

  test_accelerate_ldlt<Scalar, AccelerateLDLT<MatrixType, Lower> >();
  test_accelerate_ldlt<Scalar, AccelerateLDLTUnpivoted<MatrixType, Lower> >();
  test_accelerate_ldlt<Scalar, AccelerateLDLTSBK<MatrixType, Lower> >();
  test_accelerate_ldlt<Scalar, AccelerateLDLTTPP<MatrixType, Lower> >();

  test_accelerate_ldlt<Scalar, AccelerateLDLT<MatrixType, Upper> >();
  test_accelerate_ldlt<Scalar, AccelerateLDLTUnpivoted<MatrixType, Upper> >();
  test_accelerate_ldlt<Scalar, AccelerateLDLTSBK<MatrixType, Upper> >();
  test_accelerate_ldlt<Scalar, AccelerateLDLTTPP<MatrixType, Upper> >();

  test_accelerate_llt<Scalar, AccelerateLLT<MatrixType, Lower> >();

  test_accelerate_llt<Scalar, AccelerateLLT<MatrixType, Upper> >();

  test_accelerate_qr<Scalar, AccelerateQR<MatrixType> >();

  test_accelerate_uncompressed<Scalar, AccelerateLLT<MatrixType, Lower> >();
  test_accelerate_uncompressed<Scalar, AccelerateLDLT<MatrixType, Upper> >();
  test_accelerate_uncompressed<Scalar, AccelerateQR<MatrixType> >();
}

EIGEN_DECLARE_TEST(accelerate_support) {
  CALL_SUBTEST_1(run_tests<float>());
  CALL_SUBTEST_2(run_tests<double>());
}

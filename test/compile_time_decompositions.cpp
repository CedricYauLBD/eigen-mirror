// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of the dense decompositions, split from compile_time_evaluation because instantiating them
// dominates its compile time. Like compile_time_evaluation, this test is strictly compile-time: main.h adds
// instrumentation that constant evaluation rejects.

#include <Eigen/Dense>

using namespace Eigen;

// Dynamic sizes need heap allocation, which constant evaluation allows from C++20.
#if EIGEN_COMP_CXXVER >= 20 && defined(__cpp_lib_constexpr_dynamic_alloc)
// Dynamic outer products evaluate their operands into a local buffer, and the divide-and-conquer SVD applies plane
// rotations to strided columns.
constexpr bool dynamic_decompositions() {
  VectorXd u(3);
  u << 1, 2, 3;
  const MatrixXd outer = u * u.transpose();
  bool ok = outer(1, 2) == 6 && outer(2, 2) == 9;

  Matrix<double, 6, 6> a;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) a(i, j) = 1.0 / (i + j + 1) + (i == j ? 2 : 0);
  BDCSVD<Matrix<double, 6, 6>, ComputeFullU | ComputeFullV> bdc;
  bdc.setSwitchSize(3);
  bdc.compute(a);
  ok = ok && (bdc.matrixU() * bdc.singularValues().asDiagonal() * bdc.matrixV().transpose()).isApprox(a);
  const JacobiSVD<Matrix<double, 6, 6>> jacobi(a);
  ok = ok && jacobi.singularValues().isApprox(bdc.singularValues());
  return ok;
}

static_assert(dynamic_decompositions());

// A nonsingular test matrix with a_ij = 1/(i+j+1) + 2 delta_ij.
template <typename MatrixType>
constexpr MatrixType shifted_hilbert(Index n) {
  MatrixType a(n, n);
  for (Index i = 0; i < n; ++i)
    for (Index j = 0; j < n; ++j) a(i, j) = 1.0 / double(i + j + 1) + (i == j ? 2 : 0);
  return a;
}

// Fixed sizes up to 12 take the unrolled LU; dynamic sizes above 16 take the blocked LU.
constexpr bool lu_decompositions() {
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const PartialPivLU<Matrix3d> lu(g);
  bool ok = (g * lu.solve(Vector3d::Ones())).isApprox(Vector3d::Ones());
  const Matrix<double, 5, 5> h = shifted_hilbert<Matrix<double, 5, 5>>(5);
  ok = ok && (h * h.inverse()).isIdentity(1e-12);
  const MatrixXd d = shifted_hilbert<MatrixXd>(20);
  const PartialPivLU<MatrixXd> dlu(d);
  ok = ok && (d * dlu.solve(VectorXd::Ones(20))).isApprox(VectorXd::Ones(20));
  return ok;
}

static_assert(lu_decompositions());

constexpr bool householder_qr() {
  // The non-const lvalue selects the in-place constructor, which must keep the copied input.
  Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const HouseholderQR<Matrix3d> qr(g);
  const Matrix3d q = qr.householderQ();
  bool ok = (q * q.transpose()).isIdentity(1e-12) && (g * qr.solve(Vector3d::Ones())).isApprox(Vector3d::Ones());
  MatrixXd d = shifted_hilbert<MatrixXd>(6);
  const HouseholderQR<MatrixXd> dqr(d);
  ok = ok && (d * dqr.solve(VectorXd::Ones(6))).isApprox(VectorXd::Ones(6));
  return ok;
}

static_assert(householder_qr());

constexpr bool eigen_decompositions() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const SelfAdjointEigenSolver<Matrix3d> es(a);
  bool ok = (a * es.eigenvectors()).isApprox(es.eigenvectors() * es.eigenvalues().asDiagonal());
  SelfAdjointEigenSolver<Matrix2d> direct;
  direct.computeDirect(Matrix2d{{2, 1}, {1, 2}});
  ok = ok && direct.eigenvalues().isApprox(Vector2d(1, 3));
  const RealSchur<Matrix3d> schur(g);
  ok = ok && (schur.matrixU() * schur.matrixT() * schur.matrixU().transpose()).isApprox(g);
  const GeneralizedEigenSolver<Matrix3d> ges(a, Matrix3d::Identity());
  ok = ok && ges.info() == Success;
  const BunchKaufman<Matrix3d> bk(a);
  ok = ok && (a * bk.solve(Vector3d::Ones())).isApprox(Vector3d::Ones());
  // Bisection for the eigenvalues, inverse iteration for the eigenvectors.
  const TridiagonalEigenSolver<double> tes(Vector3d(2, 2, 2), Vector2d(1, 1));
  const Matrix3d tri{{2, 1, 0}, {1, 2, 1}, {0, 1, 2}};
  ok = ok && tes.info() == Success &&
       (tri * tes.eigenvectors()).isApprox(tes.eigenvectors() * tes.eigenvalues().asDiagonal());
  return ok;
}

static_assert(eigen_decompositions());

// The determinant accessors and the wrapper classes of the complete orthogonal decompositions. Constant evaluation
// takes the unblocked path of RandColPivHouseholderQR regardless of the block size.
constexpr bool decomposition_wrappers() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const double det = 44;
  const LLT<Matrix3d> llt(a);
  bool ok = numext::abs(llt.determinant() - det) < 1e-12 && numext::abs(llt.absDeterminant() - det) < 1e-12 &&
            llt.signDeterminant() == 1;
  const LDLT<Matrix3d> ldlt(a);
  ok = ok && numext::abs(ldlt.determinant() - det) < 1e-12 && numext::abs(ldlt.absDeterminant() - det) < 1e-12 &&
       ldlt.signDeterminant() == 1;
  const Matrix3d na = -a;
  const BunchKaufman<Matrix3d> bk(na);
  ok = ok && numext::abs(bk.determinant() + det) < 1e-12 && numext::abs(bk.absDeterminant() - det) < 1e-12 &&
       bk.signDeterminant() == -1;
  const FullPivLU<Matrix3d> lu(na);
  ok = ok && numext::abs(lu.absDeterminant() - det) < 1e-12 && lu.signDeterminant() == -1;

  Matrix3d singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  CompleteOrthogonalDecomposition<Matrix3d> cod;
  cod.setThreshold(Default).compute(singular);
  const Matrix3d pinv = cod.pseudoInverse();
  ok = ok && cod.rank() == 2 && (singular * pinv * singular).isApprox(singular);
  // The non-const lvalue selects the in-place constructors.
  const CompleteOrthogonalDecomposition<Matrix3d> cod_inplace(singular);
  ok = ok && cod_inplace.rank() == 2;

  RandColPivHouseholderQR<Matrix3d> rqr;
  rqr.setSeed(1).setBlockSize(1).compute(singular);
  ok = ok && rqr.rank() == 2 &&
       (rqr.householderQ() * rqr.matrixR().triangularView<Upper>().toDenseMatrix() * rqr.colsPermutation().transpose())
           .isApprox(singular);
  RandColPivHouseholderQR<Matrix3d> rqr_inplace(singular);
  ok = ok && rqr_inplace.rank() == 2;
  const RandCompleteOrthogonalDecomposition<Matrix3d> rcod = singular.randCompleteOrthogonalDecomposition();
  const Matrix3d rpinv = rcod.pseudoInverse();
  ok = ok && rcod.rank() == 2 && (singular * rpinv * singular).isApprox(singular);
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  ok = ok && numext::abs(g.randColPivHouseholderQr().absDeterminant() - 34) < 1e-12;
  return ok;
}

static_assert(decomposition_wrappers());

// Products of a Householder sequence with diagonal, triangular and permutation operands, from either side.
constexpr bool householder_products() {
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  const HouseholderQR<Matrix3d> qr(g);
  const auto h = qr.householderQ();
  const Matrix3d hd = h;
  const Vector3d dv(1, 2, 4);
  const auto d = dv.asDiagonal();
  const Matrix3d u = g.triangularView<Upper>();
  PermutationMatrix<3> p;
  p.indices() << 2, 0, 1;
  const Matrix3d pd = p;
  bool ok = Matrix3d(h * d).isApprox(hd * d) && Matrix3d(d * h).isApprox(d * hd);
  ok = ok && Matrix3d(h * g.triangularView<Upper>()).isApprox(hd * u) &&
       Matrix3d(g.triangularView<Upper>() * h).isApprox(u * hd);
  ok = ok && Matrix3d(h * p).isApprox(hd * pd) && Matrix3d(p * h).isApprox(pd * hd);
  ok = ok && Matrix3d(h * p.inverse()).isApprox(hd * pd.transpose()) &&
       Matrix3d(p.inverse() * h).isApprox(pd.transpose() * hd);
  return ok;
}

static_assert(householder_products());

// The non-const lvalue constructors decompose in place; the SVDs take triangular and bidiagonal input.
constexpr bool inplace_and_structured_inputs() {
  const Matrix3d a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
  const Matrix3d g{{1, 2, 0}, {3, -1, 4}, {0, 5, 2}};
  Matrix3d work = g;
  const HessenbergDecomposition<Matrix3d> hess(work);
  bool ok = (hess.matrixQ() * hess.matrixH() * hess.matrixQ().adjoint()).isApprox(g);
  work = a;
  const Tridiagonalization<Matrix3d> tri(work);
  ok = ok && (tri.matrixQ() * tri.matrixT() * tri.matrixQ().adjoint()).isApprox(a);
  work = g;
  const RealSchur<Ref<Matrix3d>> schur(work);
  ok = ok && (schur.matrixU() * schur.matrixT() * schur.matrixU().transpose()).isApprox(g);
  work = a;
  const SelfAdjointEigenSolver<Ref<Matrix3d>> es(work);
  ok = ok && (a * es.eigenvectors()).isApprox(es.eigenvectors() * es.eigenvalues().asDiagonal());
  Matrix3d b = Matrix3d::Identity();
  work = a;
  const GeneralizedEigenSolver<Matrix3d> ges(work, b);
  ok = ok && ges.info() == Success;

  const Matrix3d u = g.triangularView<Upper>();
  const JacobiSVD<Matrix3d> dense_svd(u);
  JacobiSVD<Matrix3d> svd(g.triangularView<Upper>());
  ok = ok && svd.singularValues().isApprox(dense_svd.singularValues());
  svd.compute(g.triangularView<Upper>());
  ok = ok && svd.singularValues().isApprox(dense_svd.singularValues());
  const Matrix3d bidiagonal{{1, 2, 0}, {0, 3, 4}, {0, 0, 5}};
  const BDCSVD<Matrix3d> bdc(Vector3d(1, 3, 5), Vector2d(2, 4));
  ok = ok && bdc.singularValues().isApprox(JacobiSVD<Matrix3d>(bidiagonal).singularValues());

  const Matrix3d singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const FullPivLU<Matrix3d> lu(singular);
  const MatrixXd kernel = lu.kernel();
  ok = ok && kernel.cols() == 1 && (singular * kernel).isZero(1e-12);
  return ok;
}

static_assert(inplace_and_structured_inputs());

#if EIGEN_COMP_GNUC_STRICT
// Only GCC treats std::exp as a constant expression.
constexpr bool selfadjoint_operator_exp() {
  const Matrix2d a{{0, 0}, {0, 0}};
  const SelfAdjointEigenSolver<Matrix2d> es(a);
  return es.operatorExp().isApprox(Matrix2d::Identity());
}

static_assert(selfadjoint_operator_exp());
#endif
#endif

#if EIGEN_COMP_CXXVER >= 20
constexpr bool rank_revealing() {
  const Matrix3d singular{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const ColPivHouseholderQR<Matrix3d> qr(singular);
  return qr.rank() == 2 && qr.dimensionOfKernel() == 1 && !qr.isInvertible();
}

static_assert(rank_revealing());

constexpr bool selfadjoint_eigensolver() {
  const Matrix2d a{{2, 1}, {1, 2}};
  const SelfAdjointEigenSolver<Matrix2d> es(a);
  return es.eigenvalues().isApprox(Vector2d(1, 3)) &&
         (a * es.eigenvectors()).isApprox(es.eigenvectors() * es.eigenvalues().asDiagonal());
}

static_assert(selfadjoint_eigensolver());

// Before C++26, constant evaluation cannot form realView() of std::complex coefficients, which the complex
// decompositions use.
#if defined(__cpp_lib_tuple_like) && __cpp_lib_tuple_like >= 202311L
constexpr bool complex_eigen_solver() {
  // The non-const lvalue selects the in-place constructor.
  using C = std::complex<double>;
  Matrix2cd c{{C(2, 0), C(1, 0)}, {C(1, 0), C(2, 0)}};
  const ComplexEigenSolver<Matrix2cd> ces(c);
  return (c * ces.eigenvectors()).isApprox(ces.eigenvectors() * ces.eigenvalues().asDiagonal());
}

static_assert(complex_eigen_solver());
#endif
#endif

int main() { return 0; }

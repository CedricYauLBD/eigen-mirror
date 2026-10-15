// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_GENERALIZED_SCHUR_REORDERING_H
#define EIGEN_GENERALIZED_SCHUR_REORDERING_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal
 * Size of the diagonal block of the generalized (quasi-)triangular matrix \p S starting at row/column \p k.
 * Complex Schur forms only have 1x1 blocks; real ones also have 2x2 blocks for complex conjugate eigenvalue pairs,
 * which are recognized by a nonzero subdiagonal entry of \p S (block boundaries are exact zeros).
 */
template <typename MatrixType>
Index generalized_schur_block_size(const MatrixType& S, Index k) {
  if (NumTraits<typename MatrixType::Scalar>::IsComplex) return 1;
  return (k + 1 < S.rows() && !numext::is_exactly_zero(S.coeff(k + 1, k))) ? 2 : 1;
}

/** \internal
 * Unitary 2x2 matrix whose first column is the unit vector in the direction of \p w.
 * \returns false if \p w is zero.
 */
template <typename Scalar, typename LocalMatrix>
bool generalized_schur_rotation(Matrix<Scalar, 2, 1> w, LocalMatrix& G) {
  const typename NumTraits<Scalar>::Real norm = w.stableNorm();
  if (numext::is_exactly_zero(norm)) return false;
  w /= norm;
  G.resize(2, 2);
  G << w(0), -numext::conj(w(1)), w(1), numext::conj(w(0));
  return true;
}

/** \internal
 * Transformations \p Ql, \p Zl that swap the 1x1 blocks of the upper triangular 2x2 pencil (\p A, \p B).
 *
 * As in LAPACK's xTGEX2, the rotations are formed directly from the entries rather than from a Sylvester solve, which
 * preserves the relative accuracy of small components: the right eigenvector of the lower eigenvalue
 * a22 / b22 is v = [g; -f] with f = a22 b11 - b22 a11 and g = a22 b12 - b22 a12, and the first column of Ql is
 * A v or B v, whichever is computed from the larger diagonal products.
 */
template <typename LocalMatrix>
bool generalized_schur_swap_1x1(const LocalMatrix& A, const LocalMatrix& B, LocalMatrix& Ql, LocalMatrix& Zl) {
  using Scalar = typename LocalMatrix::Scalar;
  using Vector2 = Matrix<Scalar, 2, 1>;
  const Scalar f = A(1, 1) * B(0, 0) - B(1, 1) * A(0, 0);
  const Scalar g = A(1, 1) * B(0, 1) - B(1, 1) * A(0, 1);
  if (!generalized_schur_rotation(Vector2(g, -f), Zl)) return false;
  const bool useA = numext::abs(A(1, 1)) * numext::abs(B(0, 0)) >= numext::abs(A(0, 0)) * numext::abs(B(1, 1));
  const Vector2 w = (useA ? A : B).template topLeftCorner<2, 2>() * Zl.col(0);
  return generalized_schur_rotation(w, Ql);
}

/** \internal
 * Transformations \p Ql, \p Zl that swap the diagonal blocks of sizes \p n1 and \p n2 of the block upper triangular
 * pencil (\p A, \p B), following B. Kagstrom and P. Poromaa, "Computing eigenspaces with specified eigenvalues of a
 * regular matrix pair (A, B) and condition estimation: theory, algorithms and software", Numer. Algorithms 12 (1996).
 *
 * The solution of the generalized Sylvester equation
 *   A11 X - Y A22 = -A12,  B11 X - Y B22 = -B12
 * gives the deflating subspaces of the lower block, spanned by [X; I] (right) and [Y; I] (left); the leading columns
 * of Zl and Ql are orthonormal bases of these subspaces.
 */
template <typename LocalMatrix>
bool generalized_schur_swap_sylvester(const LocalMatrix& A, const LocalMatrix& B, Index n1, Index n2, LocalMatrix& Ql,
                                      LocalMatrix& Zl) {
  using Scalar = typename LocalMatrix::Scalar;
  using SylvesterMatrix = Matrix<Scalar, Dynamic, Dynamic, 0, 8, 8>;
  using SylvesterVector = Matrix<Scalar, Dynamic, 1, 0, 8, 1>;
  const Index m = n1 + n2;
  const Index k = n1 * n2;

  // Column-major vectorization with unknowns [vec(X); vec(Y)].
  SylvesterMatrix K = SylvesterMatrix::Zero(2 * k, 2 * k);
  SylvesterVector rhs(2 * k);
  for (Index c = 0; c < n2; ++c) {
    for (Index r = 0; r < n1; ++r) {
      const Index row = c * n1 + r;
      for (Index i = 0; i < n1; ++i) {
        K(row, c * n1 + i) = A(r, i);
        K(k + row, c * n1 + i) = B(r, i);
      }
      for (Index i = 0; i < n2; ++i) {
        K(row, k + i * n1 + r) = -A(n1 + i, n1 + c);
        K(k + row, k + i * n1 + r) = -B(n1 + i, n1 + c);
      }
      rhs(row) = -A(r, n1 + c);
      rhs(k + row) = -B(r, n1 + c);
    }
  }
  const FullPivLU<SylvesterMatrix> sylvester(K);
  if (!sylvester.isInvertible()) return false;
  const SylvesterVector sol = sylvester.solve(rhs);

  LocalMatrix basis(m, n2);
  basis.topRows(n1) = Map<const LocalMatrix>(sol.data(), n1, n2);
  basis.bottomRows(n2).setIdentity();
  Zl = HouseholderQR<LocalMatrix>(basis).householderQ();
  basis.topRows(n1) = Map<const LocalMatrix>(sol.data() + k, n1, n2);
  Ql = HouseholderQR<LocalMatrix>(basis).householderQ();
  return true;
}

/** \internal
 * Swaps the adjacent diagonal blocks of sizes \p n1 (upper) and \p n2 (lower) starting at row/column \p j of the
 * generalized Schur form (\p S, \p T), where \p S is upper (quasi-)triangular and \p T is upper triangular.
 *
 * \p Q and \p Z are updated such that A = Q S Z^* and B = Q T Z^* keep holding (as in LAPACK, \p Z is the adjoint of
 * the matrix returned by RealQZ::matrixZ()).
 *
 * \returns false, leaving all matrices untouched, if the swap fails the weak or strong stability test of Kagstrom and
 * Poromaa (1996), typically because the eigenvalues of both blocks are too close.
 */
template <typename MatrixType>
bool swap_generalized_schur_blocks(MatrixType& S, MatrixType& T, MatrixType& Q, MatrixType& Z, Index j, Index n1,
                                   Index n2) {
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename NumTraits<Scalar>::Real;
  using LocalMatrix = Matrix<Scalar, Dynamic, Dynamic, 0, 4, 4>;

  const Index n = S.rows();
  const Index m = n1 + n2;
  eigen_assert(n1 >= 1 && n1 <= 2 && n2 >= 1 && n2 <= 2 && j >= 0 && j + m <= n);

  const LocalMatrix A = S.block(j, j, m, m);
  const LocalMatrix B = T.block(j, j, m, m);
  LocalMatrix Ql, Zl;
  const bool computed =
      (m == 2) ? generalized_schur_swap_1x1(A, B, Ql, Zl) : generalized_schur_swap_sylvester(A, B, n1, n2, Ql, Zl);
  if (!computed) return false;

  LocalMatrix An = Ql.adjoint() * A * Zl;
  LocalMatrix Bn = Ql.adjoint() * B * Zl;

  // Weak stability test: the new blocks decouple up to thresh = 20 eps ||(A, B)||_F.
  const RealScalar thresh =
      numext::maxi(RealScalar(20) * NumTraits<RealScalar>::epsilon() * numext::sqrt(A.squaredNorm() + B.squaredNorm()),
                   (std::numeric_limits<RealScalar>::min)());
  if (!(numext::sqrt(An.bottomLeftCorner(n1, n2).squaredNorm() + Bn.bottomLeftCorner(n1, n2).squaredNorm()) <= thresh))
    return false;
  An.bottomLeftCorner(n1, n2).setZero();
  Bn.bottomLeftCorner(n1, n2).setZero();

  // Restore the triangularity of T within the new 2x2 diagonal blocks.
  for (Index p = 0; p < m;) {
    const Index bs = (p == 0) ? n2 : n1;
    if (bs == 2) {
      const LocalMatrix H = HouseholderQR<LocalMatrix>(Bn.block(p, p, 2, 2)).householderQ();
      An.middleRows(p, 2) = H.adjoint() * An.middleRows(p, 2);
      Bn.middleRows(p, 2) = H.adjoint() * Bn.middleRows(p, 2);
      Bn(p + 1, p) = Scalar(0);
      Ql.middleCols(p, 2) = Ql.middleCols(p, 2) * H;
    }
    p += bs;
  }

  // Strong stability test: ||Ql (An, Bn) Zl^* - (A, B)||_F <= thresh.
  if (!(numext::sqrt((Ql * An * Zl.adjoint() - A).squaredNorm() + (Ql * Bn * Zl.adjoint() - B).squaredNorm()) <=
        thresh))
    return false;

  S.block(j, j, m, m) = An;
  T.block(j, j, m, m) = Bn;
  const Index right = n - j - m;
  if (right > 0) {
    S.block(j, j + m, m, right) = Ql.adjoint() * S.block(j, j + m, m, right);
    T.block(j, j + m, m, right) = Ql.adjoint() * T.block(j, j + m, m, right);
  }
  if (j > 0) {
    S.block(0, j, j, m) = S.block(0, j, j, m) * Zl;
    T.block(0, j, j, m) = T.block(0, j, j, m) * Zl;
  }
  Q.middleCols(j, m) = Q.middleCols(j, m) * Ql;
  Z.middleCols(j, m) = Z.middleCols(j, m) * Zl;
  return true;
}

/** \internal
 * Sets the entries below the structure of the generalized Schur form (\p S, \p T) to zero: below the diagonal of \p T,
 * and below the diagonal (complex) or the first subdiagonal (real) of \p S. These are rounding errors left by the QZ
 * iteration.
 */
template <typename MatrixType>
void clean_generalized_schur_form(MatrixType& S, MatrixType& T) {
  const Index n = S.rows();
  T.template triangularView<StrictlyLower>().setZero();
  if (NumTraits<typename MatrixType::Scalar>::IsComplex)
    S.template triangularView<StrictlyLower>().setZero();
  else if (n > 2)
    S.bottomLeftCorner(n - 2, n - 2).template triangularView<Lower>().setZero();
}

/** \internal
 * Reorders the generalized Schur form (\p S, \p T) with transformations \p Q, \p Z (see
 * swap_generalized_schur_blocks()) such that all eigenvalues strictly inside the unit circle lead the diagonal, keeping
 * their relative order. This corresponds to SciPy's `ordqz(..., sort='iuc')`; infinite eigenvalues are outside the unit
 * circle.
 *
 * \returns the number of eigenvalues inside the unit circle, or -1 if a swap was rejected.
 */
template <typename MatrixType>
Index reorder_generalized_schur_inside_unit_circle(MatrixType& S, MatrixType& T, MatrixType& Q, MatrixType& Z) {
  using Scalar = typename MatrixType::Scalar;
  const Index n = S.rows();
  clean_generalized_schur_form(S, T);

  Index selected = 0;
  for (Index k = 0; k < n;) {
    const Index bs = generalized_schur_block_size(S, k);
    bool inside;
    if (bs == 1) {
      inside = numext::abs(S(k, k)) < numext::abs(T(k, k));
    } else {
      // Complex conjugate pair of equal modulus: |lambda|^2 = det(S_kk) / det(T_kk), with T_kk triangular.
      inside = numext::abs(S.template block<2, 2>(k, k).determinant()) < numext::abs(T(k, k) * T(k + 1, k + 1));
    }
    if (inside) {
      // Bubble the block up past the unselected ones.
      for (Index pos = k; pos > selected;) {
        const Index pbs =
            (!NumTraits<Scalar>::IsComplex && pos - 2 >= selected && !numext::is_exactly_zero(S(pos - 1, pos - 2))) ? 2
                                                                                                                    : 1;
        if (!swap_generalized_schur_blocks(S, T, Q, Z, pos - pbs, pbs, bs)) return -1;
        pos -= pbs;
      }
      selected += bs;
    }
    k += bs;
  }
  return selected;
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_GENERALIZED_SCHUR_REORDERING_H

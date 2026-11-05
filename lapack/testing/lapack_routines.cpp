// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// Checks of Eigen's LAPACK routines against the reference LAPACK contract that need no Fortran compiler.

#include "main.h"
#include "../lapack.h"

namespace {

using std::complex;

void larfb(const char* side, const char* trans, const char* direct, const char* storev, int* m, int* n, int* k,
           double* v, int* ldv, double* t, int* ldt, double* c, int* ldc, double* work, int* ldwork) {
  BLASFUNC(dlarfb)(side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c, ldc, work, ldwork);
}
void larfb(const char* side, const char* trans, const char* direct, const char* storev, int* m, int* n, int* k,
           complex<double>* v, int* ldv, complex<double>* t, int* ldt, complex<double>* c, int* ldc,
           complex<double>* work, int* ldwork) {
  BLASFUNC(zlarfb)
  (side, trans, direct, storev, m, n, k, reinterpret_cast<double*>(v), ldv, reinterpret_cast<double*>(t), ldt,
   reinterpret_cast<double*>(c), ldc, reinterpret_cast<double*>(work), ldwork);
}

// xLARFB applies H = I - V T V^H, or H^H, where V has an implicit unit triangle: in its first k rows (DIRECT = 'F') or
// its last k rows (DIRECT = 'B'). The diagonal of that triangle and the cells beyond it are not referenced. This stores
// `unreferenced` there and compares with H formed explicitly. Zeros there, as QL or RQ leaves them for a trivial
// reflector, once made xLARFB misplace the triangle; a NaN there shows any read of those cells.
template <typename Scalar>
void check_larfb(char side, char trans, char direct, char storev, const Scalar& unreferenced) {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  int k = internal::random<int>(1, 4);
  int len = k + internal::random<int>(0, 5);
  int other = internal::random<int>(1, 5);
  int m = side == 'L' ? len : other, n = side == 'L' ? other : len;
  const bool forward = direct == 'F', colwise = storev == 'C';

  Mat V_eff = Mat::Random(len, k);  // V as a len x k matrix of columns
  for (int j = 0; j < k; ++j) {
    const int pivot = forward ? j : len - k + j;
    V_eff(pivot, j) = Scalar(1);
    for (int i = forward ? 0 : pivot + 1; i < (forward ? pivot : len); ++i) V_eff(i, j) = Scalar(0);
  }
  Mat V = colwise ? Mat(V_eff) : Mat(V_eff.adjoint());
  for (int j = 0; j < k; ++j) {
    const int pivot = forward ? j : len - k + j;
    for (int i = forward ? 0 : pivot; i <= (forward ? pivot : len - 1); ++i)
      (colwise ? V(i, j) : V(j, i)) = unreferenced;
  }
  Mat T = Mat::Random(k, k), C = Mat::Random(m, n);
  const Mat T_eff = forward ? Mat(T.template triangularView<Upper>()) : Mat(T.template triangularView<Lower>());
  const Mat H = Mat::Identity(len, len) - V_eff * T_eff * V_eff.adjoint();
  const Mat op = trans == 'N' ? H : Mat(H.adjoint());
  const Mat expected = side == 'L' ? Mat(op * C) : Mat(C * op);

  int ldv = int(V.rows()), ldt = k, ldc = m, ldwork = (std::max)(m, n);
  Mat work(ldwork, k);
  larfb(&side, &trans, &direct, &storev, &m, &n, &k, V.data(), &ldv, T.data(), &ldt, C.data(), &ldc, work.data(),
        &ldwork);
  VERIFY_IS_APPROX(C, expected);
}

template <typename Scalar>
void test_larfb() {
  const char trans_adj = NumTraits<Scalar>::IsComplex ? 'C' : 'T';
  for (const Scalar unreferenced : {Scalar(0), Scalar(std::numeric_limits<double>::quiet_NaN())})
    for (char side : {'L', 'R'})
      for (char trans : {'N', trans_adj})
        for (char direct : {'F', 'B'})
          for (char storev : {'C', 'R'}) check_larfb<Scalar>(side, trans, direct, storev, unreferenced);
}

// Like the reference, xLARFB applies H to all of C: a NaN in a row of C that V does not reach still spreads through
// W = C^H V to the whole column.
void test_larfb_nan_propagation() {
  int m = 4, n = 2, k = 1, ldv = 4, ldt = 1, ldc = 4, ldwork = 2;
  MatrixXd V(4, 1), T(1, 1), C = MatrixXd::Ones(4, 2), work(2, 1);
  V << 1, 0.5, 0.5, 0;
  T << 1;
  C(3, 0) = std::numeric_limits<double>::quiet_NaN();
  BLASFUNC(dlarfb)
  ("L", "N", "F", "C", &m, &n, &k, V.data(), &ldv, T.data(), &ldt, C.data(), &ldc, work.data(), &ldwork);
  VERIFY(C.col(0).array().isNaN().all());
  VERIFY(C.col(1).allFinite());
}

void test_ladiv() {
  const double eps = NumTraits<double>::epsilon();
  // Smith's algorithm without scaling overflows to (NaN, 0) here, since c + d (d / c) = 2e308.
  double big = 1e308, p, q;
  BLASFUNC(dladiv)(&big, &big, &big, &big, &p, &q);
  VERIFY_IS_APPROX(p, 1.0);
  VERIFY(q == 0 && !(numext::signbit)(q));
  double zx[2] = {big, big}, zy[2] = {big, big};
  const eigen_lapack_complex_double z = BLASFUNC(zladiv)(zx, zy);
  VERIFY_IS_APPROX(z.r, 1.0);
  VERIFY(z.i == 0 && !(numext::signbit)(z.i));
  float bigf = 3e38f, pf, qf;
  BLASFUNC(sladiv)(&bigf, &bigf, &bigf, &bigf, &pf, &qf);
  VERIFY_IS_APPROX(pf, 1.0f);
  VERIFY(qf == 0 && !(numext::signbit)(qf));

  // ... and underflows to (0, 0) here: (1 + i) / (c + ic) = 1 / c, which is subnormal.
  double one = 1, c = 1.5e308;
  BLASFUNC(dladiv)(&one, &one, &c, &c, &p, &q);
  VERIFY(numext::abs(p - 1 / c) <= 4 * eps * (1 / c) + 4 * eps * (std::numeric_limits<double>::min)());
  VERIFY(q == 0 && !(numext::signbit)(q));

  // The hard cases of M. Baudin and R. L. Smith, "A Robust Complex Division in Scilab", arXiv:1210.4539 (2012), with
  // their exact quotients: (a + ib) / (c + id) = p + iq. They reach the scaling, the swap for |d| > |c|, and the
  // branches for an underflowing d / c or b d / c.
  const double hard[][6] = {
      {1, 1, 1, std::ldexp(1.0, 1023), std::ldexp(1.0, -1023), -std::ldexp(1.0, -1023)},
      {1, 1, std::ldexp(1.0, -1023), std::ldexp(1.0, -1023), std::ldexp(1.0, 1023), 0},
      {std::ldexp(1.0, 1023), std::ldexp(1.0, -1023), std::ldexp(1.0, 677), std::ldexp(1.0, -677), std::ldexp(1.0, 346),
       -std::ldexp(1.0, -1008)},
      {std::ldexp(1.0, 1023), std::ldexp(1.0, 1023), 1, 1, std::ldexp(1.0, 1023), 0},
      {std::ldexp(1.0, 1020), std::ldexp(1.0, -844), std::ldexp(1.0, 656), std::ldexp(1.0, -780), std::ldexp(1.0, 364),
       -std::ldexp(1.0, -1072)},
      {std::ldexp(1.0, -71), std::ldexp(1.0, 1021), std::ldexp(1.0, 1001), std::ldexp(1.0, -323),
       std::ldexp(1.0, -1072), std::ldexp(1.0, 20)},
      {std::ldexp(1.0, -347), std::ldexp(1.0, -54), std::ldexp(1.0, -1037), std::ldexp(1.0, -1058),
       3.898125604559113300e289, 8.174961907852353577e295},
      {std::ldexp(1.0, -1074), std::ldexp(1.0, -1074), std::ldexp(1.0, -1073), std::ldexp(1.0, -1074), 0.6, 0.2},
      {std::ldexp(1.0, 1015), std::ldexp(1.0, -989), std::ldexp(1.0, 1023), std::ldexp(1.0, 1023), 0.001953125,
       -0.001953125},
      {std::ldexp(1.0, -622), std::ldexp(1.0, -1071), std::ldexp(1.0, -343), std::ldexp(1.0, -798),
       1.02951151789360578e-84, 6.97145987515076231e-220},
  };
  for (const auto& h : hard) {
    BLASFUNC(dladiv)(&h[0], &h[1], &h[2], &h[3], &p, &q);
    VERIFY(numext::hypot(p - h[4], q - h[5]) <= 4 * eps * numext::hypot(h[4], h[5]));
  }

  // For operands of moderate size, both this and std::complex division err by a few eps normwise.
  for (int i = 0; i < 1000; ++i) {
    const complex<double> x = internal::random<complex<double> >(), y = internal::random<complex<double> >();
    const double xr = x.real(), xi = x.imag(), yr = y.real(), yi = y.imag();
    BLASFUNC(dladiv)(&xr, &xi, &yr, &yi, &p, &q);
    VERIFY(numext::abs(complex<double>(p, q) - x / y) <= 8 * eps * numext::abs(x / y));
  }
}

void test_lapy2() {
  double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
  double three = 3, four = 4;
  VERIFY_IS_EQUAL(BLASFUNC(dlapy2)(&three, &four), 5.0);
  VERIFY((numext::isinf)(BLASFUNC(dlapy2)(&inf, &three)));
  // Unlike hypot, xLAPY2 propagates a NaN even when the other argument is infinite.
  VERIFY((numext::isnan)(BLASFUNC(dlapy2)(&inf, &nan)));
  VERIFY((numext::isnan)(BLASFUNC(dlapy2)(&nan, &inf)));
}

void potrf(const char* uplo, int* n, double* a, int* lda, int* info) { BLASFUNC(dpotrf)(uplo, n, a, lda, info); }
void potrf(const char* uplo, int* n, complex<double>* a, int* lda, int* info) {
  BLASFUNC(zpotrf)(uplo, n, reinterpret_cast<double*>(a), lda, info);
}

// xPOTRF returns INFO = j for the first pivot j that is <= 0 or NaN.
template <typename Scalar>
void test_potrf_info() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const Scalar nan(std::numeric_limits<double>::quiet_NaN());
  Mat spd(3, 3);
  spd << Scalar(4), Scalar(1), Scalar(1), Scalar(1), Scalar(5), Scalar(1), Scalar(1), Scalar(1), Scalar(6);
  // spd with entries (i, j) and (j, i) set to value.
  const auto modified = [&](int i, int j, const Scalar& value) {
    Mat a = spd;
    a(i, j) = a(j, i) = value;
    return a;
  };
  struct Case {
    Mat a;
    int info;
  };
  const Case cases[] = {
      {spd, 0},
      {modified(0, 0, nan), 1},
      {modified(1, 1, nan), 2},
      {modified(2, 0, nan), 3},           // a NaN off the diagonal first makes pivot 3 NaN
      {modified(1, 1, Scalar(0.25)), 2},  // 0.25 - 1/4 = 0: not positive definite
  };
  for (const Case& cs : cases) {
    for (char uplo : {'L', 'U'}) {
      Mat factor = cs.a;
      int n = 3, lda = 3, info = -1;
      potrf(&uplo, &n, factor.data(), &lda, &info);
      VERIFY_IS_EQUAL(info, cs.info);
    }
  }
}

// The same in a matrix large enough for the blocked kernel, with the NaN pivot in the first or a later block. The NaN
// must spread through the block solve and the trailing update for the scan of the diagonal to find it.
template <typename Scalar>
void test_potrf_info_blocked() {
  using Mat = Matrix<Scalar, Dynamic, Dynamic>;
  const int size = 100;
  const Scalar nan(std::numeric_limits<double>::quiet_NaN());
  const Mat spd = Mat::Constant(size, size, Scalar(1)) + Scalar(size) * Mat::Identity(size, size);
  struct Case {
    int i;
    Scalar value;
    int info;
  };
  const Case cases[] = {{-1, Scalar(0), 0}, {5, nan, 6}, {64, nan, 65}, {80, Scalar(-1000), 81}};
  for (const Case& cs : cases) {
    for (char uplo : {'L', 'U'}) {
      Mat a = spd;
      if (cs.i >= 0) a(cs.i, cs.i) = cs.value;
      if (cs.info == 6) a(80, 80) = Scalar(-1000);  // a later non-positive pivot must not mask the NaN
      int n = size, lda = size, info = -1;
      potrf(&uplo, &n, a.data(), &lda, &info);
      VERIFY_IS_EQUAL(info, cs.info);
    }
  }
}

// ZPOTRF uses only the real part of a diagonal entry, so a NaN imaginary part there is not a NaN pivot.
void test_potrf_imaginary_nan() {
  const int size = 100;
  MatrixXcd a = MatrixXcd::Constant(size, size, 1) + double(size) * MatrixXcd::Identity(size, size);
  a(2, 2) = complex<double>(size + 1, std::numeric_limits<double>::quiet_NaN());
  a(70, 70) = complex<double>(size + 1, std::numeric_limits<double>::quiet_NaN());
  for (char uplo : {'L', 'U'}) {
    MatrixXcd factor = a;
    int n = size, lda = size, info = -1;
    potrf(&uplo, &n, factor.data(), &lda, &info);
    VERIFY_IS_EQUAL(info, 0);
  }
}

// Reference xGESDD documents INFO = -4 for a NaN in A.
void test_gesdd_nan() {
  const char jobz = 'A';
  int m = 3, n = 2, lda = 3, ldu = 3, ldvt = 2, lwork = 64, info = 0;
  std::vector<double> s(2), work(128), rwork(128);
  std::vector<int> iwork(16);

  MatrixXd A = MatrixXd::Ones(3, 2), U(3, 3), VT(2, 2);
  A(1, 1) = std::numeric_limits<double>::quiet_NaN();
  BLASFUNC(dgesdd)
  (&jobz, &m, &n, A.data(), &lda, s.data(), U.data(), &ldu, VT.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
  VERIFY_IS_EQUAL(info, -4);

  MatrixXcd Z = MatrixXcd::Ones(3, 2), ZU(3, 3), ZVT(2, 2);
  Z(2, 0) = complex<double>(1, std::numeric_limits<double>::quiet_NaN());
  info = 0;
  BLASFUNC(zgesdd)
  (&jobz, &m, &n, reinterpret_cast<double*>(Z.data()), &lda, s.data(), reinterpret_cast<double*>(ZU.data()), &ldu,
   reinterpret_cast<double*>(ZVT.data()), &ldvt, work.data(), &lwork, rwork.data(), iwork.data(), &info);
  VERIFY_IS_EQUAL(info, -4);
}

void test_lacgv() {
  complex<double> x[2] = {complex<double>(1, 2), complex<double>(3, 4)};
  int n = 2, inc = 1;
  BLASFUNC(zlacgv)(&n, reinterpret_cast<double*>(x), &inc);
  VERIFY_IS_EQUAL(x[1], complex<double>(3, -4));
  n = -1;  // the reference loop runs zero times
  BLASFUNC(zlacgv)(&n, reinterpret_cast<double*>(x), &inc);
  VERIFY_IS_EQUAL(x[0], complex<double>(1, -2));
}

}  // namespace

EIGEN_DECLARE_TEST(lapack_routines) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST(test_larfb<double>());
    CALL_SUBTEST(test_larfb<complex<double> >());
  }
  CALL_SUBTEST(test_larfb_nan_propagation());
  CALL_SUBTEST(test_ladiv());
  CALL_SUBTEST(test_lapy2());
  CALL_SUBTEST(test_potrf_info<double>());
  CALL_SUBTEST(test_potrf_info<complex<double> >());
  CALL_SUBTEST(test_potrf_info_blocked<double>());
  CALL_SUBTEST(test_potrf_info_blocked<complex<double> >());
  CALL_SUBTEST(test_potrf_imaginary_nan());
  CALL_SUBTEST(test_gesdd_nan());
  CALL_SUBTEST(test_lacgv());
}

// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// SpGEMM hash accumulator with 64-bit indices: the hash h = (i - lo) * m / span by plain division, IntDivider, or a
// 64.64 fixed-point scale (monotone, not the exact floor).
#include <Eigen/SparseCore>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <random>
#include <vector>

using namespace Eigen;
using I = std::int64_t;
using U = std::uint64_t;
using SpMat = SparseMatrix<double, ColMajor, I>;

static SpMat fromTriplets(Index r, Index c, std::vector<Triplet<double, I>>& t) {
  SpMat A(r, c);
  A.setFromTriplets(t.begin(), t.end());
  return A;
}
static SpMat lap2d(I k) {
  std::vector<Triplet<double, I>> t;
  for (I i = 0; i < k; ++i)
    for (I j = 0; j < k; ++j) {
      I v = i * k + j;
      t.emplace_back(v, v, 4);
      if (i > 0) t.emplace_back(v, v - k, -1), t.emplace_back(v - k, v, -1);
      if (j > 0) t.emplace_back(v, v - 1, -1), t.emplace_back(v - 1, v, -1);
    }
  return fromTriplets(k * k, k * k, t);
}
static SpMat randmat(I rows, I cols, int perCol, unsigned seed) {
  std::mt19937_64 g(seed);
  std::uniform_int_distribution<I> d(0, rows - 1);
  std::vector<Triplet<double, I>> t;
  for (I j = 0; j < cols; ++j)
    for (int k = 0; k < perCol; ++k) t.emplace_back(d(g), j, 1.0 + k);
  return fromTriplets(rows, cols, t);
}
static SpMat banded(I n, I bw) {
  std::vector<Triplet<double, I>> t;
  for (I j = 0; j < n; ++j)
    for (I i = std::max<I>(0, j - bw); i <= std::min(n - 1, j + bw); ++i) t.emplace_back(i, j, 1.0 + double(i - j));
  return fromTriplets(n, n, t);
}

// Hash policies: set(m, span) once per column, then h(x) for x = i - lo in [0, span).
struct DivSigned {
  I m = 1, span = 1;
  void set(I m_, I span_) { m = m_, span = span_; }
  I operator()(I x) const { return x * m / span; }
};
struct DivUnsigned {
  U m = 1, span = 1;
  void set(I m_, I span_) { m = U(m_), span = U(span_); }
  I operator()(I x) const { return I(U(x) * m / span); }
};
// The IntDivider policies need MR !3509; build with -DEIGEN_BENCH_INT_DIVIDER against that branch.
#ifdef EIGEN_BENCH_INT_DIVIDER
struct IntDivSigned {
  I m = 1;
  IntDivider<I> d;
  void set(I m_, I span_) { m = m_, d.compute(span_); }
  I operator()(I x) const { return d.divide(x * m); }
};
struct IntDivUnsigned {
  U m = 1;
  IntDivider<U> d;
  void set(I m_, I span_) { m = U(m_), d.compute(U(span_)); }
  I operator()(I x) const { return I(d.divide(U(x) * m)); }
};
#endif
struct FixedPoint {
  U scale = 0;  // floor(m 2^64 / span) < 2^64 since m < span
  void set(I m_, I span_) { scale = U((static_cast<unsigned __int128>(U(m_)) << 64) / U(span_)); }
  I operator()(I x) const { return I((static_cast<unsigned __int128>(U(x)) * scale) >> 64); }
};

template <typename Hash>
static void hash_product(const SpMat& A, const SpMat& B, SpMat& C) {
  const Index rows = A.rows(), cols = B.cols();
  const I* Ap = A.outerIndexPtr();
  const I* Ai = A.innerIndexPtr();
  const double* Ax = A.valuePtr();
  constexpr I kEmpty = -1;
  static thread_local std::vector<I> key;
  static thread_local std::vector<double> val;
  Hash h;
  C.resize(rows, cols);
  C.reserve(A.nonZeros() + B.nonZeros());
  for (Index j = 0; j < cols; ++j) {
    C.startVec(j);
    I lo = rows, hi = -1, bound = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const I k = it.index(), b = Ap[k], e = Ap[k + 1];
      if (b == e) continue;
      lo = std::min(lo, Ai[b]);
      hi = std::max(hi, Ai[e - 1]);
      bound += e - b;
    }
    if (bound == 0) continue;
    const I span = hi - lo + 1, cap = std::min(bound, span);
    I m = std::max<I>(1, cap * 3 / 2);
    const bool direct = m >= span;
    if (direct)
      m = span;
    else
      h.set(m, span);
    const I size = direct ? span : m + cap;
    if (I(key.size()) < size) key.resize(size, kEmpty), val.resize(size);
    I* K = key.data();
    double* W = val.data();
    I top = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const double y = it.value();
      const I k = it.index();
      for (I p = Ap[k]; p < Ap[k + 1]; ++p) {
        const I i = Ai[p];
        const double v = Ax[p] * y;
        I s = direct ? i - lo : h(i - lo);
        while (K[s] != kEmpty && K[s] < i) ++s;
        if (K[s] == i) {
          W[s] += v;
        } else {
          if (K[s] != kEmpty) {
            I q = s;
            while (K[q] != kEmpty) ++q;
            top = std::max(top, q);
            for (; q > s; --q) K[q] = K[q - 1], W[q] = W[q - 1];
          }
          K[s] = i;
          W[s] = v;
          top = std::max(top, s);
        }
      }
    }
    for (I s = 0; s <= top; ++s)
      if (K[s] != kEmpty) C.insertBackByOuterInner(j, K[s]) = W[s], K[s] = kEmpty;
  }
  C.finalize();
}

// Same table, borrowed from C: column j's table occupies C's free storage starting at nnz(C[:, :j]), and the
// sorted scan compacts it left in place (the write slot never passes the read slot).
template <typename Hash>
static void borrowed_product(const SpMat& A, const SpMat& B, SpMat& C) {
  const Index rows = A.rows(), cols = B.cols();
  const I* Ap = A.outerIndexPtr();
  const I* Ai = A.innerIndexPtr();
  const double* Ax = A.valuePtr();
  constexpr I kEmpty = -1;
  Hash h;
  C.resize(rows, cols);
  C.reserve(A.nonZeros() + B.nonZeros());
  I* Cp = C.outerIndexPtr();
  I nz = 0;
  for (Index j = 0; j < cols; ++j) {
    Cp[j] = nz;
    I lo = rows, hi = -1, bound = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const I k = it.index(), b = Ap[k], e = Ap[k + 1];
      if (b == e) continue;
      lo = std::min(lo, Ai[b]);
      hi = std::max(hi, Ai[e - 1]);
      bound += e - b;
    }
    if (bound == 0) continue;
    const I span = hi - lo + 1, cap = std::min(bound, span);
    I m = std::max<I>(1, cap * 3 / 2);
    const bool direct = m >= span;
    if (direct)
      m = span;
    else
      h.set(m, span);
    const I size = direct ? span : m + cap;
    if (C.data().allocatedSize() < nz + size) {
      C.resizeNonZeros(nz);
      C.data().reserve(std::max<I>(size, nz));
    }
    I* K = C.innerIndexPtr() + nz;
    double* W = C.valuePtr() + nz;
    std::fill(K, K + size, kEmpty);
    I top = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const double y = it.value();
      const I k = it.index();
      for (I p = Ap[k]; p < Ap[k + 1]; ++p) {
        const I i = Ai[p];
        const double v = Ax[p] * y;
        I s = direct ? i - lo : h(i - lo);
        while (K[s] != kEmpty && K[s] < i) ++s;
        if (K[s] == i) {
          W[s] += v;
        } else {
          if (K[s] != kEmpty) {
            I q = s;
            while (K[q] != kEmpty) ++q;
            top = std::max(top, q);
            for (; q > s; --q) K[q] = K[q - 1], W[q] = W[q - 1];
          }
          K[s] = i;
          W[s] = v;
          top = std::max(top, s);
        }
      }
    }
    I w = 0;
    for (I s = 0; s <= top; ++s)
      if (K[s] != kEmpty) K[w] = K[s], W[w] = W[s], ++w;
    nz += w;
  }
  Cp[cols] = nz;
  C.resizeNonZeros(nz);
}

static void dense_product(const SpMat& A, const SpMat& B, SpMat& C) {
  const Index rows = A.rows(), cols = B.cols();
  const I* Ap = A.outerIndexPtr();
  const I* Ai = A.innerIndexPtr();
  const double* Ax = A.valuePtr();
  std::vector<char> mask(rows, 0);
  std::vector<double> val(rows);
  std::vector<I> idx(rows);
  C.resize(rows, cols);
  C.reserve(A.nonZeros() + B.nonZeros());
  for (Index j = 0; j < cols; ++j) {
    C.startVec(j);
    Index nnz = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const double y = it.value();
      const I k = it.index();
      for (I p = Ap[k]; p < Ap[k + 1]; ++p) {
        const I i = Ai[p];
        if (!mask[i])
          mask[i] = 1, val[i] = Ax[p] * y, idx[nnz++] = i;
        else
          val[i] += Ax[p] * y;
      }
    }
    std::sort(idx.begin(), idx.begin() + nnz);
    for (Index q = 0; q < nnz; ++q) C.insertBackByOuterInner(j, idx[q]) = val[idx[q]], mask[idx[q]] = 0;
  }
  C.finalize();
}

struct Case {
  const char* name;
  SpMat A, B;
};
static std::vector<Case>& cases() {
  static std::vector<Case> c;
  if (c.empty()) {
    SpMat L = lap2d(500);
    c.push_back({"lap2d500^2", L, L});
    SpMat R = randmat(200000, 200000, 8, 1);
    c.push_back({"rand200k_8^2", R, R});
    SpMat Bd = banded(200000, 4);
    c.push_back({"band200k_9^2", Bd, Bd});
    c.push_back({"tall2M_x_thin", randmat(2000000, 2000, 50, 2), randmat(2000, 2000, 4, 3)});
    c.push_back({"wide_rand", randmat(50000, 50000, 4, 4), randmat(50000, 500000, 2, 5)});
  }
  return c;
}

using Fn = void (*)(const SpMat&, const SpMat&, SpMat&);
template <Fn F>
static void BM(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  SpMat C;
  F(k.A, k.B, C);
  SpMat ref;
  dense_product(k.A, k.B, ref);
  if (ref.nonZeros() != C.nonZeros() || (ref - C).norm() > 1e-12 * ref.norm()) {
    fprintf(stderr, "MISMATCH %s\n", k.name);
    std::abort();
  }
  for (auto _ : s) {
    F(k.A, k.B, C);
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
BENCHMARK_TEMPLATE(BM, dense_product)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_product<DivSigned>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_product<DivUnsigned>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
#ifdef EIGEN_BENCH_INT_DIVIDER
BENCHMARK_TEMPLATE(BM, hash_product<IntDivSigned>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_product<IntDivUnsigned>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
#endif
BENCHMARK_TEMPLATE(BM, hash_product<FixedPoint>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, borrowed_product<FixedPoint>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();

// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Prototype: order-preserving open-addressing accumulator for C = A * B (col-major).
// h(i) = floor((i - lo) * m / span), span = hi - lo + 1. h is monotone, and an insertion that meets a larger key
// shifts the rest of the cluster right (ordered linear probing), so the occupied slots stay sorted. Probing runs
// into a tail of `cap` slots instead of wrapping, so one scan of the table yields a sorted column.
#include <Eigen/SparseCore>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

using namespace Eigen;
using SpMat = SparseMatrix<double, ColMajor, int>;

static SpMat lap2d(int k) {
  std::vector<Triplet<double>> t;
  for (int i = 0; i < k; ++i)
    for (int j = 0; j < k; ++j) {
      int v = i * k + j;
      t.emplace_back(v, v, 4);
      if (i > 0) t.emplace_back(v, v - k, -1), t.emplace_back(v - k, v, -1);
      if (j > 0) t.emplace_back(v, v - 1, -1), t.emplace_back(v - 1, v, -1);
    }
  SpMat A(k * k, k * k);
  A.setFromTriplets(t.begin(), t.end());
  return A;
}
static SpMat randmat(int rows, int cols, int perCol, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<int> d(0, rows - 1);
  std::vector<Triplet<double>> t;
  for (int j = 0; j < cols; ++j)
    for (int k = 0; k < perCol; ++k) t.emplace_back(d(g), j, 1.0 + k);
  SpMat A(rows, cols);
  A.setFromTriplets(t.begin(), t.end());
  return A;
}
static SpMat banded(int n, int bw) {
  std::vector<Triplet<double>> t;
  for (int j = 0; j < n; ++j)
    for (int i = std::max(0, j - bw); i <= std::min(n - 1, j + bw); ++i) t.emplace_back(i, j, 1.0 + i - j);
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  return A;
}

// Hash accumulator product. Lhs columns must be sorted (compressed SparseMatrix).
template <bool FixedPoint>
static void hash_product_t(const SpMat& A, const SpMat& B, SpMat& C) {
  const Index rows = A.rows(), cols = B.cols();
  const int* Ap = A.outerIndexPtr();
  const int* Ai = A.innerIndexPtr();
  const double* Ax = A.valuePtr();
  // Worst-case column bound, for the one-time table allocation.
  Index maxCap = 0, total = 0;
  for (Index j = 0; j < cols; ++j) {
    Index bound = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) bound += Ap[it.index() + 1] - Ap[it.index()];
    maxCap = std::max(maxCap, std::min<Index>(bound, rows));
    total += bound;
  }
  constexpr int kEmpty = -1;
  std::vector<int> key(3 * maxCap + 1, kEmpty);
  std::vector<double> val(3 * maxCap + 1);
  C.resize(rows, cols);
  C.reserve(std::min<Index>(total, rows * cols));
  for (Index j = 0; j < cols; ++j) {
    C.startVec(j);
    Index lo = rows, hi = -1, bound = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const Index k = it.index(), b = Ap[k], e = Ap[k + 1];
      if (b == e) continue;
      lo = std::min<Index>(lo, Ai[b]);
      hi = std::max<Index>(hi, Ai[e - 1]);
      bound += e - b;
    }
    if (bound == 0) continue;
    const Index span = hi - lo + 1;
    const Index cap = std::min(bound, span);
    Index m = 2 * cap;
    const bool direct = m >= span;
    if (direct) m = span;
    const Index size = direct ? span : m + cap;
    // floor(m 2^32 / span) keeps h monotone and below m; it need not equal the exact floor((i - lo) m / span).
    const std::uint64_t scale = direct ? 0 : (std::uint64_t(m) << 32) / std::uint64_t(span);
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const double y = it.value();
      const Index k = it.index();
      for (int p = Ap[k]; p < Ap[k + 1]; ++p) {
        const int i = Ai[p];
        const double v = Ax[p] * y;
        Index s = direct ? i - lo : FixedPoint ? Index((std::uint64_t(i - lo) * scale) >> 32) : (i - lo) * m / span;
        while (key[s] != kEmpty && key[s] < i) ++s;
        if (key[s] == i) {
          val[s] += v;
        } else if (key[s] == kEmpty) {
          key[s] = i;
          val[s] = v;
        } else {
          // Shift the cluster right by one to keep it sorted.
          Index t = s;
          while (key[t] != kEmpty) ++t;
          for (; t > s; --t) key[t] = key[t - 1], val[t] = val[t - 1];
          key[s] = i;
          val[s] = v;
        }
      }
    }
    for (Index s = 0; s < size; ++s) {
      if (key[s] != kEmpty) {
        C.insertBackByOuterInner(j, key[s]) = val[s];
        key[s] = kEmpty;
      }
    }
  }
  C.finalize();
}

static double g_ins, g_probe, g_shift, g_scan, g_out, g_lohi;
static void hash_stats(const SpMat& A, const SpMat& B, SpMat& C) {
  const Index rows = A.rows(), cols = B.cols();
  const int* Ap = A.outerIndexPtr();
  const int* Ai = A.innerIndexPtr();
  const double* Ax = A.valuePtr();
  // Worst-case column bound, for the one-time table allocation.
  Index maxCap = 0, total = 0;
  for (Index j = 0; j < cols; ++j) {
    Index bound = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) bound += Ap[it.index() + 1] - Ap[it.index()];
    maxCap = std::max(maxCap, std::min<Index>(bound, rows));
    total += bound;
  }
  constexpr int kEmpty = -1;
  std::vector<int> key(3 * maxCap + 1, kEmpty);
  std::vector<double> val(3 * maxCap + 1);
  C.resize(rows, cols);
  C.reserve(std::min<Index>(total, rows * cols));
  for (Index j = 0; j < cols; ++j) {
    C.startVec(j);
    Index lo = rows, hi = -1, bound = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const Index k = it.index(), b = Ap[k], e = Ap[k + 1];
      if (b == e) continue;
      lo = std::min<Index>(lo, Ai[b]);
      hi = std::max<Index>(hi, Ai[e - 1]);
      bound += e - b;
    }
    if (bound == 0) continue;
    const Index span = hi - lo + 1;
    const Index cap = std::min(bound, span);
    Index m = 2 * cap;
    const bool direct = m >= span;
    if (direct) m = span;
    const Index size = direct ? span : m + cap;
    // floor(m 2^32 / span) keeps h monotone and below m; it need not equal the exact floor((i - lo) m / span).
    const std::uint64_t scale = direct ? 0 : (std::uint64_t(m) << 32) / std::uint64_t(span);
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const double y = it.value();
      const Index k = it.index();
      for (int p = Ap[k]; p < Ap[k + 1]; ++p) {
        const int i = Ai[p];
        const double v = Ax[p] * y;
        Index s = direct ? i - lo : (i - lo) * m / span;
        g_ins++;
        while (key[s] != kEmpty && key[s] < i) ++s, g_probe++;
        if (key[s] == i) {
          val[s] += v;
        } else if (key[s] == kEmpty) {
          key[s] = i;
          val[s] = v;
        } else {
          // Shift the cluster right by one to keep it sorted.
          Index t = s;
          while (key[t] != kEmpty) ++t;
          g_shift += t - s;
          for (; t > s; --t) key[t] = key[t - 1], val[t] = val[t - 1];
          key[s] = i;
          val[s] = v;
        }
      }
    }
    g_scan += size;
    for (Index s = 0; s < size; ++s) {
      if (key[s] != kEmpty) {
        g_out++;
        C.insertBackByOuterInner(j, key[s]) = val[s];
        key[s] = kEmpty;
      }
    }
  }
  C.finalize();
}

static void hash_product(const SpMat& A, const SpMat& B, SpMat& C) { hash_product_t<false>(A, B, C); }
static void hashfp_product(const SpMat& A, const SpMat& B, SpMat& C) { hash_product_t<true>(A, B, C); }

// Baseline with the same raw access: dense mask + values + index list, sort per column.
static void dense_product(const SpMat& A, const SpMat& B, SpMat& C) {
  const Index rows = A.rows(), cols = B.cols();
  const int* Ap = A.outerIndexPtr();
  const int* Ai = A.innerIndexPtr();
  const double* Ax = A.valuePtr();
  std::vector<char> mask(rows, 0);
  std::vector<double> val(rows);
  std::vector<int> idx(rows);
  C.resize(rows, cols);
  C.reserve(A.nonZeros() + B.nonZeros());
  for (Index j = 0; j < cols; ++j) {
    C.startVec(j);
    Index nnz = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const double y = it.value();
      const Index k = it.index();
      for (int p = Ap[k]; p < Ap[k + 1]; ++p) {
        const int i = Ai[p];
        if (!mask[i])
          mask[i] = 1, val[i] = Ax[p] * y, idx[nnz++] = i;
        else
          val[i] += Ax[p] * y;
      }
    }
    std::sort(idx.begin(), idx.begin() + nnz);
    for (Index q = 0; q < nnz; ++q) {
      const int i = idx[q];
      C.insertBackByOuterInner(j, i) = val[i];
      mask[i] = 0;
    }
  }
  C.finalize();
}

// v2: tunable. LoadX2: m = bound * LoadX2 / 2. Scan stops at the highest occupied slot; the table grows lazily.
template <int LoadX2, bool AoS>
static void hash2_product(const SpMat& A, const SpMat& B, SpMat& C) {
  const Index rows = A.rows(), cols = B.cols();
  const int* Ap = A.outerIndexPtr();
  const int* Ai = A.innerIndexPtr();
  const double* Ax = A.valuePtr();
  constexpr int kEmpty = -1;
  struct Slot {
    int key;
    double val;
  };
  static thread_local std::vector<int> key;
  static thread_local std::vector<double> val;
  static thread_local std::vector<Slot> slot;
  C.resize(rows, cols);
  C.reserve(A.nonZeros() + B.nonZeros());
  for (Index j = 0; j < cols; ++j) {
    C.startVec(j);
    Index lo = rows, hi = -1, bound = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const Index k = it.index(), b = Ap[k], e = Ap[k + 1];
      if (b == e) continue;
      lo = std::min<Index>(lo, Ai[b]);
      hi = std::max<Index>(hi, Ai[e - 1]);
      bound += e - b;
    }
    if (bound == 0) continue;
    const Index span = hi - lo + 1;
    const Index cap = std::min(bound, span);
    Index m = std::max<Index>(1, cap * LoadX2 / 2);
    const bool direct = m >= span;
    if (direct) m = span;
    const Index size = direct ? span : m + cap;
    if (AoS) {
      if (Index(slot.size()) < size) slot.resize(size, Slot{kEmpty, 0});
    } else if (Index(key.size()) < size)
      key.resize(size, kEmpty), val.resize(size);
    int* K = AoS ? nullptr : key.data();
    double* V = AoS ? nullptr : val.data();
    Slot* S = AoS ? slot.data() : nullptr;
    auto keyAt = [&](Index p) -> int& { return AoS ? S[p].key : K[p]; };
    auto valAt = [&](Index p) -> double& { return AoS ? S[p].val : V[p]; };
    const std::uint64_t scale = direct ? 0 : (std::uint64_t(m) << 32) / std::uint64_t(span);
    Index top = 0;
    for (SpMat::InnerIterator it(B, j); it; ++it) {
      const double y = it.value();
      const Index k = it.index();
      for (int p = Ap[k]; p < Ap[k + 1]; ++p) {
        const int i = Ai[p];
        const double v = Ax[p] * y;
        Index s = direct ? i - lo : Index((std::uint64_t(i - lo) * scale) >> 32);
        while (keyAt(s) != kEmpty && keyAt(s) < i) ++s;
        if (keyAt(s) == i) {
          valAt(s) += v;
        } else {
          if (keyAt(s) != kEmpty) {
            Index t = s;
            while (keyAt(t) != kEmpty) ++t;
            top = std::max(top, t);
            for (; t > s; --t) keyAt(t) = keyAt(t - 1), valAt(t) = valAt(t - 1);
          }
          keyAt(s) = i;
          valAt(s) = v;
          top = std::max(top, s);
        }
      }
    }
    for (Index s = 0; s <= top; ++s) {
      if (keyAt(s) != kEmpty) {
        C.insertBackByOuterInner(j, keyAt(s)) = valAt(s);
        keyAt(s) = kEmpty;
      }
    }
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

static void check(int id) {
  static std::vector<bool> done(16);
  if (done[id]) return;
  done[id] = true;
  const Case& k = cases()[id];
  SpMat ref = k.A * k.B, h;
  hashfp_product(k.A, k.B, h);
  if (ref.nonZeros() != h.nonZeros() || (ref - h).norm() > 1e-12 * ref.norm()) {
    fprintf(stderr, "FP MISMATCH\n");
    std::abort();
  }
  hash_product(k.A, k.B, h);
  auto chk = [&](const char* n, void (*f)(const SpMat&, const SpMat&, SpMat&)) {
    SpMat x;
    f(k.A, k.B, x);
    if (ref.nonZeros() != x.nonZeros() || (ref - x).norm() > 1e-12 * ref.norm()) {
      fprintf(stderr, "%s MISMATCH\n", n);
      std::abort();
    }
    for (Index c = 0; c < x.cols(); ++c)
      for (int p = x.outerIndexPtr()[c] + 1; p < x.outerIndexPtr()[c + 1]; ++p)
        if (x.innerIndexPtr()[p - 1] >= x.innerIndexPtr()[p]) {
          fprintf(stderr, "%s UNSORTED\n", n);
          std::abort();
        }
  };
  chk("v2_2", hash2_product<2, false>);
  chk("v2_3", hash2_product<3, false>);
  chk("v2_4", hash2_product<4, false>);
  chk("v2_3aos", hash2_product<3, true>);
  if (ref.nonZeros() != h.nonZeros() || (ref - h).norm() > 1e-12 * ref.norm()) {
    fprintf(stderr, "MISMATCH %s\n", k.name);
    std::abort();
  }
  for (Index j = 0; j < h.cols(); ++j)
    for (int p = h.outerIndexPtr()[j] + 1; p < h.outerIndexPtr()[j + 1]; ++p)
      if (h.innerIndexPtr()[p - 1] >= h.innerIndexPtr()[p]) {
        fprintf(stderr, "UNSORTED %s\n", k.name);
        std::abort();
      }
}

static void BM_eigen(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  check(int(s.range(0)));
  SpMat C;
  for (auto _ : s) {
    C = k.A * k.B;
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
  s.counters["nnzC"] = double(C.nonZeros());
}
static void BM_hash(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  check(int(s.range(0)));
  SpMat C;
  for (auto _ : s) {
    hash_product(k.A, k.B, C);
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
static void BM_dense(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  SpMat C;
  for (auto _ : s) {
    dense_product(k.A, k.B, C);
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
static void BM_stats(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  const int* Ap = k.A.outerIndexPtr();
  const int* Ai = k.A.innerIndexPtr();
  double direct = 0, cols = 0;
  for (Index j = 0; j < k.B.cols(); ++j) {
    Index lo = k.A.rows(), hi = -1, bound = 0;
    for (SpMat::InnerIterator it(k.B, j); it; ++it) {
      Index c = it.index();
      if (Ap[c] == Ap[c + 1]) continue;
      lo = std::min<Index>(lo, Ai[Ap[c]]), hi = std::max<Index>(hi, Ai[Ap[c + 1] - 1]), bound += Ap[c + 1] - Ap[c];
    }
    if (!bound) continue;
    cols++;
    if (2 * std::min(bound, hi - lo + 1) >= hi - lo + 1) direct++;
  }
  for (auto _ : s) {
  }
  s.SetLabel(k.name);
  s.counters["directFrac"] = direct / cols;
}
static void BM_hashfp(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  check(int(s.range(0)));
  SpMat C;
  for (auto _ : s) {
    hashfp_product(k.A, k.B, C);
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
BENCHMARK(BM_hashfp)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
static void BM_probe(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  g_ins = g_probe = g_shift = g_scan = g_out = 0;
  SpMat C;
  hash_stats(k.A, k.B, C);
  for (auto _ : s) {
  }
  s.SetLabel(k.name);
  s.counters["probe/ins"] = g_probe / g_ins;
  s.counters["shift/ins"] = g_shift / g_ins;
  s.counters["scan/out"] = g_scan / g_out;
  s.counters["ins/out"] = g_ins / g_out;
  s.counters["out/col"] = g_out / k.B.cols();
}
template <void (*F)(const SpMat&, const SpMat&, SpMat&)>
static void BM_v(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  check(int(s.range(0)));
  SpMat C;
  for (auto _ : s) {
    F(k.A, k.B, C);
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
BENCHMARK_TEMPLATE(BM_v, hash2_product<2, false>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_v, hash2_product<3, false>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_v, hash2_product<4, false>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_v, hash2_product<3, true>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_probe)->DenseRange(0, 4)->Iterations(1);
BENCHMARK(BM_stats)->DenseRange(0, 4)->Iterations(1);
BENCHMARK(BM_dense)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_eigen)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_hash)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();

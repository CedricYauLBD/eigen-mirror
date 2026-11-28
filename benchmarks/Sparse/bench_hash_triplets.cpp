// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/SparseCore>
#include <benchmark/benchmark.h>
#include <algorithm>
#include <random>
#include <vector>

using namespace Eigen;
using SpMat = SparseMatrix<double, ColMajor, int>;
using Trip = Triplet<double, int>;

// Scatter triplets into mat's own storage order (unsorted, with duplicates). Returns per-outer segment starts in Op.
static void scatter(const std::vector<Trip>& t, Index rows, Index cols, SpMat& mat) {
  mat.resize(rows, cols);
  int* Op = mat.outerIndexPtr();
  for (const Trip& x : t) Op[x.col() + 1]++;
  std::partial_sum(Op, Op + cols + 1, Op);
  mat.resizeNonZeros(Op[cols]);
  std::vector<int> next(Op, Op + cols);
  int* Ii = mat.innerIndexPtr();
  double* V = mat.valuePtr();
  for (const Trip& x : t) {
    int k = next[x.col()]++;
    Ii[k] = x.row();
    V[k] = x.value();
  }
}

// No transpose: scatter, collapse duplicates with the wi marking, sort each outer vector in place.
static void mark_sort(const std::vector<Trip>& t, Index rows, Index cols, SpMat& mat) {
  scatter(t, rows, cols, mat);
  int* Op = mat.outerIndexPtr();
  int* Ii = mat.innerIndexPtr();
  double* V = mat.valuePtr();
  std::vector<int> wi(rows, -1);
  std::vector<std::pair<int, double>> buf;
  int count = 0;
  for (Index j = 0; j < cols; ++j) {
    const int begin = count, end = Op[j + 1];
    for (int k = Op[j]; k < end; ++k) {
      const int i = Ii[k];
      if (wi[i] >= begin) {
        V[wi[i]] += V[k];
      } else {
        Ii[count] = i;
        V[count] = V[k];
        wi[i] = count++;
      }
    }
    Op[j] = begin;
    const int n = count - begin;
    if (n > 1) {
      buf.resize(n);
      for (int q = 0; q < n; ++q) buf[q] = {Ii[begin + q], V[begin + q]};
      std::sort(buf.begin(), buf.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      for (int q = 0; q < n; ++q) Ii[begin + q] = buf[q].first, V[begin + q] = buf[q].second;
    }
  }
  Op[cols] = count;
  mat.data().resize(count);
}

// Hash: scatter, then per outer vector accumulate into the order-preserving table and write back sorted.
template <int LoadX2, int SmallT = 0>
static void hash_set(const std::vector<Trip>& t, Index rows, Index cols, SpMat& mat) {
  scatter(t, rows, cols, mat);
  int* Op = mat.outerIndexPtr();
  int* Ii = mat.innerIndexPtr();
  double* V = mat.valuePtr();
  constexpr int kEmpty = -1;
  static thread_local std::vector<int> key;
  static thread_local std::vector<double> val;
  int count = 0;
  for (Index j = 0; j < cols; ++j) {
    const int b = Op[j], e = Op[j + 1];
    Op[j] = count;
    if (b == e) continue;
    int lo = Ii[b], hi = Ii[b];
    for (int k = b + 1; k < e; ++k) lo = std::min(lo, Ii[k]), hi = std::max(hi, Ii[k]);
    const Index bound = e - b, span = Index(hi) - lo + 1, cap = std::min(bound, span);
    // A short vector degenerates to ordered insertion into one run (m = 1).
    Index m = bound <= SmallT ? 1 : std::max<Index>(1, cap * LoadX2 / 2);
    const bool direct = m >= span;
    if (direct) m = span;
    const Index size = direct ? span : m + cap;
    if (Index(key.size()) < size) key.resize(size, kEmpty), val.resize(size);
    int* K = key.data();
    double* W = val.data();
    const std::uint64_t scale = direct ? 0 : (std::uint64_t(m) << 32) / std::uint64_t(span);
    Index top = 0;
    for (int k = b; k < e; ++k) {
      const int i = Ii[k];
      const double v = V[k];
      Index s = direct ? i - lo : Index((std::uint64_t(i - lo) * scale) >> 32);
      while (K[s] != kEmpty && K[s] < i) ++s;
      if (K[s] == i) {
        W[s] += v;
      } else {
        if (K[s] != kEmpty) {
          Index q = s;
          while (K[q] != kEmpty) ++q;
          top = std::max(top, q);
          for (; q > s; --q) K[q] = K[q - 1], W[q] = W[q - 1];
        }
        K[s] = i;
        W[s] = v;
        top = std::max(top, s);
      }
    }
    for (Index s = 0; s <= top; ++s)
      if (K[s] != kEmpty) Ii[count] = K[s], V[count] = W[s], ++count, K[s] = kEmpty;
  }
  Op[cols] = count;
  mat.data().resize(count);
}

static void eigen_set(const std::vector<Trip>& t, Index rows, Index cols, SpMat& mat) {
  mat.resize(rows, cols);
  mat.setFromTriplets(t.begin(), t.end());
}

struct Case {
  const char* name;
  Index rows, cols;
  std::vector<Trip> t;
};
static std::vector<Case>& cases() {
  static std::vector<Case> c;
  if (!c.empty()) return c;
  std::mt19937 g(7);
  {  // bilinear quads on a 1000x1000 node grid, element order: 16 triplets per element, ~4x duplicates
    const int k = 1000;
    Case x{"fe_quad1000", k * k, k * k, {}};
    for (int a = 0; a + 1 < k; ++a)
      for (int b = 0; b + 1 < k; ++b) {
        int n[4] = {a * k + b, a * k + b + 1, (a + 1) * k + b, (a + 1) * k + b + 1};
        for (int r : n)
          for (int s : n) x.t.emplace_back(r, s, 1.0);
      }
    c.push_back(std::move(x));
  }
  {
    Case x{"rand1M_8M", 1000000, 1000000, {}};
    std::uniform_int_distribution<int> d(0, 999999);
    for (int q = 0; q < 8000000; ++q) x.t.emplace_back(d(g), d(g), 1.0);
    c.push_back(std::move(x));
  }
  {
    Case x{"dup2k_4M", 2000, 2000, {}};
    std::uniform_int_distribution<int> d(0, 1999);
    for (int q = 0; q < 4000000; ++q) x.t.emplace_back(d(g), d(g), 1.0);
    c.push_back(std::move(x));
  }
  {
    Case x{"tall10M_x100", 10000000, 100, {}};
    std::uniform_int_distribution<int> d(0, 9999999), e(0, 99);
    for (int q = 0; q < 5000000; ++q) x.t.emplace_back(d(g), e(g), 1.0);
    c.push_back(std::move(x));
  }
  {  // banded, shuffled
    Case x{"band1M_9_shuf", 1000000, 1000000, {}};
    for (int j = 0; j < 1000000; ++j)
      for (int i = std::max(0, j - 4); i <= std::min(999999, j + 4); ++i) x.t.emplace_back(i, j, 1.0);
    std::shuffle(x.t.begin(), x.t.end(), g);
    c.push_back(std::move(x));
  }
  return c;
}

using Fn = void (*)(const std::vector<Trip>&, Index, Index, SpMat&);
static void check(int id, Fn f, const char* n) {
  const Case& k = cases()[id];
  SpMat ref, x;
  eigen_set(k.t, k.rows, k.cols, ref);
  f(k.t, k.rows, k.cols, x);
  bool ok = ref.nonZeros() == x.nonZeros();
  for (Index q = 0; ok && q <= k.cols; ++q) ok = ref.outerIndexPtr()[q] == x.outerIndexPtr()[q];
  for (Index q = 0; ok && q < ref.nonZeros(); ++q)
    ok = ref.innerIndexPtr()[q] == x.innerIndexPtr()[q] && ref.valuePtr()[q] == x.valuePtr()[q];
  if (!ok) {
    fprintf(stderr, "%s MISMATCH on %s\n", n, k.name);
    std::abort();
  }
}

template <Fn F>
static void BM(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  static bool checked[8] = {};
  if (!checked[s.range(0)]) check(int(s.range(0)), F, "variant"), checked[s.range(0)] = true;
  SpMat m;
  for (auto _ : s) {
    F(k.t, k.rows, k.cols, m);
    benchmark::DoNotOptimize(m.valuePtr());
  }
  s.SetLabel(k.name);
  s.counters["nnz"] = double(m.nonZeros());
}
BENCHMARK_TEMPLATE(BM, eigen_set)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, mark_sort)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_set<2>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_set<3>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_set<4>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_set<6>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_set<3, 16>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_set<3, 32>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, hash_set<3, 64>)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
static void BM_scatter(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  SpMat m;
  for (auto _ : s) {
    scatter(k.t, k.rows, k.cols, m);
    benchmark::DoNotOptimize(m.valuePtr());
  }
  s.SetLabel(k.name);
}
BENCHMARK(BM_scatter)->DenseRange(0, 4)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();

// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Sorting the inner vectors of a compressed matrix: std::sort (sortInnerIndices), double transpose, and
// internal::sort_inner_vector.
#include <Eigen/SparseCore>
#include <benchmark/benchmark.h>
#include <random>
#include <numeric>
using namespace Eigen;
using I = int;
using SpMat = SparseMatrix<double, ColMajor, I>;
using SpMatR = SparseMatrix<double, RowMajor, I>;
using Trip = Triplet<double, I>;
static SpMat randmat(I rows, I cols, int perCol, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d(0, rows - 1);
  std::vector<Trip> t;
  for (I j = 0; j < cols; ++j)
    for (int q = 0; q < perCol; ++q) t.emplace_back(d(g), j, 1.0);
  SpMat m(rows, cols);
  m.setFromTriplets(t.begin(), t.end());
  return m;
}
static SpMat lap2d(I k) {
  std::vector<Trip> t;
  for (I a = 0; a < k; ++a)
    for (I b = 0; b < k; ++b) {
      I n = a * k + b;
      t.emplace_back(n, n, 4.0);
      if (a) t.emplace_back(n, n - k, -1.0);
      if (a + 1 < k) t.emplace_back(n, n + k, -1.0);
      if (b) t.emplace_back(n, n - 1, -1.0);
      if (b + 1 < k) t.emplace_back(n, n + 1, -1.0);
    }
  SpMat m(k * k, k * k);
  m.setFromTriplets(t.begin(), t.end());
  return m;
}
// Relabel the rows by a random permutation: columns become unsorted.
static SpMat scramble(SpMat m, unsigned seed) {
  std::vector<I> p(m.rows());
  std::iota(p.begin(), p.end(), 0);
  std::mt19937 g(seed);
  std::shuffle(p.begin(), p.end(), g);
  for (Index k = 0; k < m.nonZeros(); ++k) m.innerIndexPtr()[k] = p[m.innerIndexPtr()[k]];
  return m;
}
struct Case {
  const char* name;
  SpMat m;
};
static std::vector<Case>& cases() {
  static std::vector<Case> c;
  if (c.empty()) {
    c.push_back({"lap2d1000_perm", scramble(lap2d(1000), 1)});
    c.push_back({"rand1M_8", scramble(randmat(1000000, 1000000, 8, 2), 3)});
    c.push_back({"rand100k_60", scramble(randmat(100000, 100000, 60, 4), 5)});
    c.push_back({"rand20k_1000", scramble(randmat(20000, 20000, 1000, 6), 7)});
    c.push_back({"tall2M_2k_500", scramble(randmat(2000000, 2000, 500, 8), 9)});
    c.push_back({"wide2k_2M_2", scramble(randmat(2000, 2000000, 2, 10), 11)});
  }
  return c;
}
static void std_sort(SpMat& m) { m.sortInnerIndices(); }
static void transpose2(SpMat& m) {
  SpMatR t(m);
  m = t;
}
// internal::sort_inner_vector, the sort of the sorted sparse product.
static void inner_sort(SpMat& m) {
  I* outer = m.outerIndexPtr();
  I* idx = m.innerIndexPtr();
  double* val = m.valuePtr();
  internal::inner_sort_workspace<double, I> workspace;
  for (Index j = 0; j < m.outerSize(); ++j) {
    Index b = outer[j], e = outer[j + 1];
    if (e - b < 2) continue;
    I lo = idx[b], hi = lo;
    Index descents = 0;
    for (Index k = b + 1; k < e; ++k) {
      descents += idx[k] < idx[k - 1];
      lo = std::min(lo, idx[k]);
      hi = std::max(hi, idx[k]);
    }
    internal::sort_inner_vector(idx + b, val + b, e - b, lo, hi, descents, workspace);
  }
}
using Fn = void (*)(SpMat&);
template <Fn F>
static void BM(benchmark::State& s) {
  const Case& k = cases()[s.range(0)];
  {
    SpMat a = k.m, b = k.m;
    F(a);
    std_sort(b);
    if ((a - b).norm() != 0 || a.innerIndicesAreSorted() != a.outerSize()) std::abort();
  }
  for (auto _ : s) {
    s.PauseTiming();
    SpMat m = k.m;
    s.ResumeTiming();
    F(m);
    benchmark::DoNotOptimize(m.valuePtr());
  }
  s.SetLabel(k.name);
}
BENCHMARK_TEMPLATE(BM, std_sort)->DenseRange(0, 5)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, transpose2)->DenseRange(0, 5)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM, inner_sort)->DenseRange(0, 5)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();

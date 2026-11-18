// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Public-API benchmark: sparse products and setFromTriplets, for comparing two Eigen trees.
#include <Eigen/SparseCore>
#include <benchmark/benchmark.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
using namespace Eigen;
#ifndef IDX
#define IDX int
#endif
using I = IDX;
using SpMat = SparseMatrix<double, ColMajor, I>;
using SpMatR = SparseMatrix<double, RowMajor, I>;
using Trip = Triplet<double, I>;

static SpMat fromTriplets(Index r, Index c, std::vector<Trip>& t) {
  SpMat m(r, c);
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
  return fromTriplets(k * k, k * k, t);
}
static SpMat randmat(I rows, I cols, int perCol, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d(0, rows - 1);
  std::vector<Trip> t;
  for (I j = 0; j < cols; ++j)
    for (int q = 0; q < perCol; ++q) t.emplace_back(d(g), j, 1.0);
  return fromTriplets(rows, cols, t);
}
static SpMat banded(I n, I bw) {
  std::vector<Trip> t;
  for (I j = 0; j < n; ++j)
    for (I i = std::max<I>(0, j - bw); i <= std::min<I>(n - 1, j + bw); ++i) t.emplace_back(i, j, 1.0);
  return fromTriplets(n, n, t);
}
// Trilinear hexahedra on a k^3 node grid, 3 dof per node: 81 entries per interior column, in 9 clusters.
static SpMat fehex(I k) {
  std::vector<Trip> t;
  for (I a = 0; a + 1 < k; ++a)
    for (I b = 0; b + 1 < k; ++b)
      for (I e = 0; e + 1 < k; ++e) {
        I dof[24];
        int q = 0;
        for (int da = 0; da < 2; ++da)
          for (int db = 0; db < 2; ++db)
            for (int de = 0; de < 2; ++de)
              for (int f = 0; f < 3; ++f) dof[q++] = 3 * (((a + da) * k + b + db) * k + e + de) + f;
        for (I r : dof)
          for (I w : dof) t.emplace_back(r, w, 1.0);
      }
  return fromTriplets(3 * k * k * k, 3 * k * k * k, t);
}
// Each column is one contiguous run of len entries starting at a random row.
static SpMat runs(I rows, I cols, I len, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d(0, rows - len);
  std::vector<Trip> t;
  for (I j = 0; j < cols; ++j) {
    I first = d(g);
    for (I q = 0; q < len; ++q) t.emplace_back(first + q, j, 1.0);
  }
  return fromTriplets(rows, cols, t);
}
// Columns are contiguous runs of len entries; the columns j, j + groups, j + 2 groups share one run.
static SpMat grouped_runs(I rows, I groups, I len, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d(0, rows - len);
  std::vector<I> first(groups);
  for (I& f : first) f = d(g);
  std::vector<Trip> t;
  for (I j = 0; j < 3 * groups; ++j)
    for (I q = 0; q < len; ++q) t.emplace_back(first[j % groups] + q, j, 1.0);
  return fromTriplets(rows, 3 * groups, t);
}
// Each column takes picks random groups and has an entry for each of the three columns of a group.
static SpMat group_picks(I groups, I cols, int picks, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_int_distribution<I> d(0, groups - 1);
  std::vector<Trip> t;
  for (I j = 0; j < cols; ++j)
    for (int q = 0; q < picks; ++q) {
      I k = d(g);
      for (I r = 0; r < 3; ++r) t.emplace_back(k + r * groups, j, 1.0);
    }
  return fromTriplets(3 * groups, cols, t);
}
// Renumbers rows and columns by one random permutation, as an unstructured mesh would be numbered.
static SpMat renumber(const SpMat& m, unsigned seed) {
  std::vector<I> p(m.rows());
  std::iota(p.begin(), p.end(), I(0));
  std::mt19937 g(seed);
  std::shuffle(p.begin(), p.end(), g);
  std::vector<Trip> t;
  for (Index j = 0; j < m.outerSize(); ++j)
    for (SpMat::InnerIterator it(m, j); it; ++it) t.emplace_back(p[it.index()], p[j], it.value());
  return fromTriplets(m.rows(), m.cols(), t);
}
struct PCase {
  const char* name;
  SpMat A, B;
};
static std::vector<PCase>& pcases() {
  static std::vector<PCase> c;
  if (c.empty()) {
    SpMat L = lap2d(500);
    c.push_back({"lap2d500^2", L, L});
    SpMat R = randmat(200000, 200000, 8, 1);
    c.push_back({"rand200k_8^2", R, R});
    SpMat Bd = banded(200000, 4);
    c.push_back({"band200k_9^2", Bd, Bd});
    c.push_back({"tall2M_x_thin", randmat(2000000, 2000, 50, 2), randmat(2000, 2000, 4, 3)});
    c.push_back({"wide_rand", randmat(50000, 50000, 4, 4), randmat(50000, 500000, 2, 5)});
    // Galerkin-like triple product pieces: restriction (wide) times fine operator
    SpMat P = randmat(250000, 62500, 4, 6);
    SpMat Pt = P.transpose();
    c.push_back({"RtA_lap", Pt, L});
    c.push_back({"AP_lap", L, P});
    SpMat S = randmat(20000, 20000, 60, 7);
    c.push_back({"rand20k_60^2", S, S});
    SpMat H = fehex(16);
    c.push_back({"fehex16^2", H, H});
    SpMat Q = lap2d(300);
    SpMat Q2 = Q * Q;
    c.push_back({"lap2d300^2_x_lap", Q2, Q});
    SpMat D = randmat(3000, 3000, 300, 8);
    c.push_back({"rand3k_300^2", D, D});
    SpMat Lp = renumber(L, 9);
    c.push_back({"lap2d500_renum^2", Lp, Lp});
    SpMat Hp = renumber(H, 10);
    c.push_back({"fehex16_renum^2", Hp, Hp});
    SpMat G = lap2d(1000);
    c.push_back({"lap2d1000^2", G, G});
    SpMat W = randmat(2000000, 2000000, 8, 11);
    c.push_back({"rand2M_8^2", W, W});
    // Large vectors with few repeated indices over a wide span: scattered, then in a few dense clusters.
    c.push_back({"tall2M_60x60", randmat(2000000, 20000, 60, 12), randmat(20000, 2000, 60, 13)});
    c.push_back({"runs100_x_rand4", runs(2000000, 20000, 100, 14), randmat(20000, 20000, 4, 15)});
    c.push_back({"runs1000_x_rand4", runs(2000000, 2000, 1000, 16), randmat(2000, 2000, 4, 17)});
    c.push_back({"runs100_x_rand16", runs(2000000, 20000, 100, 18), randmat(20000, 5000, 16, 19)});
    // Each result vector is a few dense clusters whose indices each sum three products.
    c.push_back({"runs100_tripled4", grouped_runs(2000000, 2500, 100, 20), group_picks(2500, 20000, 4, 21)});
    c.push_back({"runs1000_tripled4", grouped_runs(2000000, 2500, 1000, 22), group_picks(2500, 2000, 4, 23)});
    // A sparse column with one entry: the product must not cost a pass over the whole left operand.
    {
      std::vector<Trip> one{Trip(7, 0, 2.0)};
      c.push_back({"rand20k_200_x_col1", randmat(20000, 20000, 200, 24), fromTriplets(20000, 1, one)});
    }
    // Right operands with one entry per vector, so each result vector is one scaled vector of the left operand: a
    // sparse diagonal scaling, and a selection of every second column of a matrix too tall for a cached marker.
    {
      std::vector<Trip> d, e;
      for (I k = 0; k < L.cols(); ++k) d.emplace_back(k, k, 1.0 + (k % 3));
      c.push_back({"lap2d500_x_diag", L, fromTriplets(L.cols(), L.cols(), d)});
      for (I k = 0; k < W.cols() / 2; ++k) e.emplace_back(2 * k, k, 2.0);
      c.push_back({"rand2M_8_x_select", W, fromTriplets(W.cols(), W.cols() / 2, e)});
    }
  }
  return c;
}
// ccc: col*col->col; rrr: row*row->row; rcc: row*col->col; ccr: col*col->row
static void BM_ccc(benchmark::State& s) {
  const PCase& k = pcases()[s.range(0)];
  SpMat C;
  for (auto _ : s) {
    C = k.A * k.B;
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
static void BM_rrr(benchmark::State& s) {
  const PCase& k = pcases()[s.range(0)];
  SpMatR A = k.A, B = k.B, C;
  for (auto _ : s) {
    C = A * B;
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
static void BM_rcc(benchmark::State& s) {
  const PCase& k = pcases()[s.range(0)];
  SpMatR A = k.A;
  SpMat C;
  for (auto _ : s) {
    C = A * k.B;
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
static void BM_ccr(benchmark::State& s) {
  const PCase& k = pcases()[s.range(0)];
  SpMatR C;
  for (auto _ : s) {
    C = k.A * k.B;
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
static void BM_rrc(benchmark::State& s) {
  const PCase& k = pcases()[s.range(0)];
  SpMatR A = k.A, B = k.B;
  SpMat C;
  for (auto _ : s) {
    C = A * B;
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
// (A * B).pruned(1, tolerance), col*col->col. Arg 1 selects the tolerance: 0 drops only exact zeros, 1 is the median
// magnitude of the entries of A * B and drops about half of them.
static void BM_ccc_pruned(benchmark::State& s) {
  const PCase& k = pcases()[s.range(0)];
  SpMat C = k.A * k.B;
  double tolerance = 0.0;
  if (s.range(1) == 1 && C.nonZeros() > 0) {
    std::vector<double> magnitudes(C.valuePtr(), C.valuePtr() + C.nonZeros());
    for (double& v : magnitudes) v = std::abs(v);
    std::nth_element(magnitudes.begin(), magnitudes.begin() + magnitudes.size() / 2, magnitudes.end());
    tolerance = magnitudes[magnitudes.size() / 2];
  }
  Index full = C.nonZeros();
  for (auto _ : s) {
    C = (k.A * k.B).pruned(1.0, tolerance);
    benchmark::DoNotOptimize(C.valuePtr());
  }
  Index expected = 0;
  SpMat ref = k.A * k.B;
  for (Index p = 0; p < ref.nonZeros(); ++p) expected += !(std::abs(ref.valuePtr()[p]) <= tolerance);
  if (C.nonZeros() != expected) s.SkipWithError("pruned product keeps the wrong number of entries");
  s.counters["kept"] = full == 0 ? 1.0 : double(C.nonZeros()) / double(full);
  s.SetLabel(k.name);
}
static void BM_rrr_pruned(benchmark::State& s) {
  const PCase& k = pcases()[s.range(0)];
  SpMatR A = k.A, B = k.B, C;
  for (auto _ : s) {
    C = (A * B).pruned();
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
static void BM_ccr_pruned(benchmark::State& s) {
  const PCase& k = pcases()[s.range(0)];
  SpMatR C;
  for (auto _ : s) {
    C = (k.A * k.B).pruned();
    benchmark::DoNotOptimize(C.valuePtr());
  }
  s.SetLabel(k.name);
}
BENCHMARK(BM_ccc)->DenseRange(0, 23)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_rrr)->DenseRange(0, 7)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_rcc)->DenseRange(0, 7)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ccr)->DenseRange(0, 7)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_rrc)->DenseRange(0, 7)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ccc_pruned)->ArgsProduct({benchmark::CreateDenseRange(0, 23, 1), {0, 1}})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_rrr_pruned)->DenseRange(0, 7)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ccr_pruned)->DenseRange(0, 7)->Unit(benchmark::kMillisecond);

struct TCase {
  const char* name;
  Index rows, cols;
  std::vector<Trip> t;
};
static std::vector<TCase>& tcases() {
  static std::vector<TCase> c;
  if (!c.empty()) return c;
  std::mt19937 g(7);
  {
    const I k = 1000;
    TCase x{"fe_quad1000", k * k, k * k, {}};
    for (I a = 0; a + 1 < k; ++a)
      for (I b = 0; b + 1 < k; ++b) {
        I n[4] = {a * k + b, a * k + b + 1, (a + 1) * k + b, (a + 1) * k + b + 1};
        for (I r : n)
          for (I q : n) x.t.emplace_back(r, q, 1.0);
      }
    c.push_back(std::move(x));
  }
  {
    TCase x{"rand1M_8M", 1000000, 1000000, {}};
    std::uniform_int_distribution<I> d(0, 999999);
    for (int q = 0; q < 8000000; ++q) x.t.emplace_back(d(g), d(g), 1.0);
    c.push_back(std::move(x));
  }
  {
    TCase x{"dup2k_4M", 2000, 2000, {}};
    std::uniform_int_distribution<I> d(0, 1999);
    for (int q = 0; q < 4000000; ++q) x.t.emplace_back(d(g), d(g), 1.0);
    c.push_back(std::move(x));
  }
  {
    TCase x{"tall10M_x100", 10000000, 100, {}};
    std::uniform_int_distribution<I> d(0, 9999999), e(0, 99);
    for (int q = 0; q < 5000000; ++q) x.t.emplace_back(d(g), e(g), 1.0);
    c.push_back(std::move(x));
  }
  {
    TCase x{"wide100_x10M", 100, 10000000, {}};
    std::uniform_int_distribution<I> d(0, 9999999), e(0, 99);
    for (int q = 0; q < 5000000; ++q) x.t.emplace_back(e(g), d(g), 1.0);
    c.push_back(std::move(x));
  }
  {
    TCase x{"band1M_9_shuf", 1000000, 1000000, {}};
    for (I j = 0; j < 1000000; ++j)
      for (I i = std::max<I>(0, j - 4); i <= std::min<I>(999999, j + 4); ++i) x.t.emplace_back(i, j, 1.0);
    std::shuffle(x.t.begin(), x.t.end(), g);
    c.push_back(std::move(x));
  }
  {
    TCase x{"band1M_9_sorted", 1000000, 1000000, {}};
    for (I j = 0; j < 1000000; ++j)
      for (I i = std::max<I>(0, j - 4); i <= std::min<I>(999999, j + 4); ++i) x.t.emplace_back(i, j, 1.0);
    c.push_back(std::move(x));
  }
  {
    const I k = 60;
    TCase x{"fe_hex60_3dof", 3 * k * k * k, 3 * k * k * k, {}};  // 24x24 element matrices
    for (I a = 0; a + 1 < k; ++a)
      for (I b = 0; b + 1 < k; ++b)
        for (I e = 0; e + 1 < k; ++e) {
          I dof[24];
          int q = 0;
          for (int da = 0; da < 2; ++da)
            for (int db = 0; db < 2; ++db)
              for (int de = 0; de < 2; ++de)
                for (int f = 0; f < 3; ++f) dof[q++] = 3 * (((a + da) * k + b + db) * k + e + de) + f;
          for (I r : dof)
            for (I w : dof) x.t.emplace_back(r, w, 1.0);
        }
    c.push_back(std::move(x));
  }
  return c;
}
template <typename M>
static void BM_set(benchmark::State& s) {
  const TCase& k = tcases()[s.range(0)];
  M m(k.rows, k.cols);
  for (auto _ : s) {
    m.setFromTriplets(k.t.begin(), k.t.end());
    benchmark::DoNotOptimize(m.valuePtr());
  }
  s.SetLabel(k.name);
}
BENCHMARK_TEMPLATE(BM_set, SpMat)->DenseRange(0, 7)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_set, SpMatR)->DenseRange(0, 7)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();

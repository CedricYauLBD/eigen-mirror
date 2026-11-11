// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Operations that shift stored entries inside a SparseMatrix or a SparseVector: sorted insertion, reserve(),
// makeCompressed(), adding a diagonal, and assigning an inner vector. Each runs for double, where a shift is a
// memmove, and for a scalar that owns heap memory, where it is one assignment per entry.

#include <benchmark/benchmark.h>
#include <Eigen/SparseCore>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

using namespace Eigen;

namespace {

// Owns heap memory, as multi-precision and automatic-differentiation scalars do: a copy duplicates kLimbs doubles,
// a move exchanges a pointer.
class HeapScalar {
 public:
  static constexpr int kLimbs = 8;
  HeapScalar() = default;
  HeapScalar(double value) : limbs_(kLimbs, value) {}
  HeapScalar& operator+=(const HeapScalar& other) {
    if (other.limbs_.empty()) return *this;
    if (limbs_.empty()) limbs_.assign(kLimbs, 0.0);
    for (int i = 0; i < kLimbs; ++i) limbs_[i] += other.limbs_[i];
    return *this;
  }
  friend HeapScalar operator+(HeapScalar a, const HeapScalar& b) { return a += b; }
  double value() const { return limbs_.empty() ? 0.0 : limbs_[0]; }

 private:
  std::vector<double> limbs_;
};

double toDouble(double x) { return x; }
double toDouble(const HeapScalar& x) { return x.value(); }

template <typename SparseType>
double sumOfEntries(const SparseType& m) {
  double sum = 0;
  for (Index j = 0; j < m.outerSize(); ++j)
    for (typename SparseType::InnerIterator it(m, j); it; ++it) sum += toDouble(it.value());
  return sum;
}

// A compressed size x size matrix with nnzPerVector entries in each column, none on the diagonal.
template <typename Scalar>
SparseMatrix<Scalar> makeOffDiagonal(Index size, Index nnzPerVector) {
  SparseMatrix<Scalar> m(size, size);
  m.reserve(VectorXi::Constant(size, int(nnzPerVector)));
  for (Index j = 0; j < size; ++j)
    for (Index k = 0; k < nnzPerVector; ++k) m.insert((j + 1 + 7 * k) % size, j) = Scalar(1.0);
  m.makeCompressed();
  return m;
}

}  // namespace

namespace Eigen {
template <>
struct NumTraits<HeapScalar> : NumTraits<double> {
  static constexpr int RequireInitialization = 1;
};
}  // namespace Eigen

// insert() into reserved room in descending inner order: every insertion shifts the entries already in its vector.
template <typename Scalar>
static void BM_InsertDescending(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index outerSize = 256;
  SparseMatrix<Scalar> m(4 * nnzPerVector, outerSize);
  const VectorXi sizes = VectorXi::Constant(outerSize, int(nnzPerVector));
  for (auto _ : state) {
    m.setZero();
    m.reserve(sizes);
    for (Index j = 0; j < outerSize; ++j)
      for (Index k = nnzPerVector - 1; k >= 0; --k) m.insert(3 * k, j) = Scalar(1.0);
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != outerSize * nnzPerVector || sumOfEntries(m) != double(outerSize * nnzPerVector))
    state.SkipWithError("insertion lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * outerSize * nnzPerVector);
}

// insert() in random order without reserve(): a vector that is full takes room from a neighbor or makes the matrix
// reserve one more slot per vector, and both shift entries across vectors.
template <typename Scalar>
static void BM_InsertRandomNoReserve(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index outerSize = 256;
  const Index innerSize = 4 * nnzPerVector;
  std::vector<std::pair<Index, Index>> coords;
  for (Index j = 0; j < outerSize; ++j)
    for (Index k = 0; k < nnzPerVector; ++k) coords.emplace_back(3 * k, j);
  std::shuffle(coords.begin(), coords.end(), std::mt19937(42));
  SparseMatrix<Scalar> m(innerSize, outerSize);
  for (auto _ : state) {
    m.setZero();
    for (const std::pair<Index, Index>& c : coords) m.insert(c.first, c.second) = Scalar(1.0);
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != outerSize * nnzPerVector || sumOfEntries(m) != double(outerSize * nnzPerVector))
    state.SkipWithError("insertion lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * outerSize * nnzPerVector);
}

// reserve() spreads the vectors of a compressed matrix apart, makeCompressed() packs them again.
template <typename Scalar>
static void BM_ReserveMakeCompressed(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index size = 1024;
  SparseMatrix<Scalar> m = makeOffDiagonal<Scalar>(size, nnzPerVector);
  const VectorXi sizes = VectorXi::Constant(size, 2);
  for (auto _ : state) {
    m.reserve(sizes);
    m.makeCompressed();
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != size * nnzPerVector || sumOfEntries(m) != double(size * nnzPerVector))
    state.SkipWithError("reserve() and makeCompressed() lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * size * nnzPerVector);
}

// m += diagonal where m stores no diagonal entry: one sorted insertion per vector. range(1) selects a compressed
// matrix, which shifts every entry behind an insertion, or one free slot per vector, which shifts within the vector.
template <typename Scalar>
static void BM_AddDiagonal(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const bool withCapacity = state.range(1) != 0;
  const Index size = 4096;
  const SparseMatrix<Scalar> base = makeOffDiagonal<Scalar>(size, nnzPerVector);
  const Matrix<Scalar, Dynamic, 1> diagonal = Matrix<Scalar, Dynamic, 1>::Constant(size, Scalar(1.0));
  const VectorXi sizes = VectorXi::Constant(size, 1);
  SparseMatrix<Scalar> m;
  for (auto _ : state) {
    state.PauseTiming();
    m = base;
    if (withCapacity) m.reserve(sizes);
    state.ResumeTiming();
    m += diagonal.asDiagonal();
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != size * (nnzPerVector + 1) || sumOfEntries(m) != double(size * (nnzPerVector + 1)))
    state.SkipWithError("adding the diagonal lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * size * nnzPerVector);
}

// Assigning the first inner vector of a compressed matrix shifts every entry behind it: right when the vector
// grows, left when it shrinks.
template <typename Scalar>
static void BM_AssignInnerVector(benchmark::State& state) {
  const Index nnzPerVector = state.range(0);
  const Index size = 1024;
  SparseMatrix<Scalar> m = makeOffDiagonal<Scalar>(size, nnzPerVector);
  SparseMatrix<Scalar> longer(size, 1), shorter(size, 1);
  for (Index k = 0; k < 2 * nnzPerVector; ++k) longer.insert(k, 0) = Scalar(1.0);
  shorter.insert(0, 0) = Scalar(1.0);
  longer.makeCompressed();
  shorter.makeCompressed();
  for (auto _ : state) {
    m.col(0) = longer;
    m.col(0) = shorter;
    benchmark::DoNotOptimize(m.valuePtr());
    benchmark::ClobberMemory();
  }
  if (m.nonZeros() != (size - 1) * nnzPerVector + 1 || sumOfEntries(m) != double((size - 1) * nnzPerVector + 1))
    state.SkipWithError("assigning an inner vector lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * 2 * (size - 1) * nnzPerVector);
}

// SparseVector::coeffRef() in descending index order: every insertion shifts all the stored entries.
template <typename Scalar>
static void BM_VectorCoeffRefDescending(benchmark::State& state) {
  const Index nnz = state.range(0);
  SparseVector<Scalar> v(4 * nnz);
  for (auto _ : state) {
    v.setZero();
    for (Index k = nnz - 1; k >= 0; --k) v.coeffRef(3 * k) = Scalar(1.0);
    benchmark::DoNotOptimize(v.valuePtr());
    benchmark::ClobberMemory();
  }
  if (v.nonZeros() != nnz || sumOfEntries(v) != double(nnz))
    state.SkipWithError("insertion lost or changed coefficients");
  state.SetItemsProcessed(state.iterations() * nnz);
}

BENCHMARK_TEMPLATE(BM_InsertDescending, double)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_InsertDescending, HeapScalar)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_InsertRandomNoReserve, double)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_InsertRandomNoReserve, HeapScalar)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_ReserveMakeCompressed, double)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_ReserveMakeCompressed, HeapScalar)->Arg(4)->Arg(16)->Arg(64);
BENCHMARK_TEMPLATE(BM_AddDiagonal, double)->ArgsProduct({{4, 16, 64}, {0, 1}});
BENCHMARK_TEMPLATE(BM_AddDiagonal, HeapScalar)->ArgsProduct({{4, 16, 64}, {0, 1}});
BENCHMARK_TEMPLATE(BM_AssignInnerVector, double)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_AssignInnerVector, HeapScalar)->Arg(4)->Arg(16);
BENCHMARK_TEMPLATE(BM_VectorCoeffRefDescending, double)->Arg(16)->Arg(128)->Arg(1024);
BENCHMARK_TEMPLATE(BM_VectorCoeffRefDescending, HeapScalar)->Arg(16)->Arg(128)->Arg(1024);

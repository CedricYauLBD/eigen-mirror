// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Constant evaluation of tensors on the default device. main.h adds instrumentation that breaks constant
// evaluation, so this test only compiles.

#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

// Tensor evaluation in a constant expression needs constexpr destructors.
#if EIGEN_COMP_CXXVER >= 20

constexpr bool fixed_size() {
  TensorFixedSize<int, Sizes<2, 3>> a;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = 3 * i + j;
  TensorFixedSize<int, Sizes<2, 3>> b;
  b.setConstant(2);
  TensorFixedSize<int, Sizes<2, 3>> c;
  c = a * b + a;
  bool ok = c(1, 2) == 15 && c(0, 0) == 0;
  TensorFixedSize<int, Sizes<>> total;
  total = a.sum();
  ok = ok && total() == 15;
  TensorFixedSize<int, Sizes<>> largest;
  largest = a.maximum();
  return ok && largest() == 5;
}

static_assert(fixed_size());

#if defined(__cpp_lib_constexpr_dynamic_alloc)
constexpr bool dynamic_size() {
  Tensor<double, 2> a(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = 3 * i + j;
  Tensor<double, 2> b = a * 2.0 - a.constant(1.0);
  bool ok = b(1, 2) == 9 && b.dimension(0) == 2 && b.dimension(1) == 3;

  // Reductions over one dimension and over all of them.
  const array<Index, 1> rows{0};
  Tensor<double, 1> column_sums = a.sum(rows);
  ok = ok && column_sums.size() == 3 && column_sums(0) == 3 && column_sums(2) == 7;
  Tensor<double, 0> total = a.sum();
  Tensor<double, 0> largest = a.maximum();
  Tensor<double, 0> smallest = a.minimum();
  ok = ok && total() == 15 && largest() == 5 && smallest() == 0;

  // a a^T = [[5, 14], [14, 50]].
  const array<IndexPair<Index>, 1> dims{IndexPair<Index>(1, 1)};
  Tensor<double, 2> gram = a.contract(a, dims);
  ok = ok && gram(0, 0) == 5 && gram(0, 1) == 14 && gram(1, 1) == 50;

  const array<Index, 2> shape{3, 2};
  Tensor<double, 2> reshaped = a.reshape(shape);
  ok = ok && reshaped(1, 0) == 3 && reshaped(0, 1) == 4;
  const array<Index, 2> offsets{0, 1};
  const array<Index, 2> extents{2, 2};
  Tensor<double, 2> sliced = a.slice(offsets, extents);
  ok = ok && sliced(1, 1) == 5 && sliced(0, 0) == 1;
  const array<int, 2> swap{1, 0};
  Tensor<double, 2> shuffled = a.shuffle(swap);
  ok = ok && shuffled(2, 1) == 5 && shuffled.dimension(0) == 3;
  const array<Index, 2> shifts{1, 1};
  Tensor<double, 2> rolled = a.roll(shifts);
  ok = ok && rolled(0, 0) == a(1, 1) && rolled(1, 2) == a(0, 0);
  return ok;
}

static_assert(dynamic_size());

constexpr bool more_operations() {
  Tensor<int, 2> a(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j) a(i, j) = 3 * i + j;

  const double sum = Tensor<double, 2>(a.cast<double>()).sum();
  bool ok = sum == 15;
  Tensor<int, 2> joined = a.concatenate(a, 0);
  ok = ok && joined.dimension(0) == 4 && joined(3, 2) == 5;
  const array<bool, 2> flip{true, false};
  Tensor<int, 2> reversed = a.reverse(flip);
  ok = ok && reversed(0, 0) == 3 && reversed(1, 2) == 2;
  const array<std::pair<Index, Index>, 2> padding{std::make_pair(1, 0), std::make_pair(0, 1)};
  Tensor<int, 2> padded = a.pad(padding);
  ok = ok && padded.dimension(0) == 3 && padded.dimension(1) == 4 && padded(0, 0) == 0 && padded(2, 1) == 4;
  const array<Index, 2> strides{1, 2};
  Tensor<int, 2> strided = a.stride(strides);
  ok = ok && strided.dimension(1) == 2 && strided(1, 1) == 5;
  Tensor<int, 1> row = a.chip(1, 0);
  ok = ok && row.size() == 3 && row(2) == 5;
  const array<Index, 2> copies{2, 1};
  Tensor<int, 2> broadcast = a.broadcast(copies);
  ok = ok && broadcast.dimension(0) == 4 && broadcast(2, 1) == 1;
  Tensor<int, 2, RowMajor> swapped = a.swap_layout();
  ok = ok && swapped.dimension(0) == 3 && swapped(2, 1) == 5;
  Tensor<int, 2> selected = (a > a.constant(2)).select(a, a.constant(-1));
  ok = ok && selected(0, 2) == -1 && selected(1, 0) == 3;
  Tensor<Index, 0> largest = a.argmax();
  ok = ok && largest() == 5;

  // A seeded generator is a pure function of (seed, index).
  Tensor<double, 1> u(16);
  u = u.random(internal::UniformRandomGenerator<double>(42));
  Tensor<double, 1> v(16);
  v = v.random(internal::UniformRandomGenerator<double>(42));
  for (int i = 0; i < 16; ++i) ok = ok && u(i) >= 0 && u(i) < 1 && u(i) == v(i);
  return ok && u(0) != u(1);
}

static_assert(more_operations());
#endif

#endif

int main() { return 0; }

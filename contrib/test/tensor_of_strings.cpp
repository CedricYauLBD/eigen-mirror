// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <Eigen/Tensor>

using Eigen::Tensor;
using Eigen::TensorMap;

static void test_assign() {
  std::string data1[6];
  TensorMap<Tensor<std::string, 2>> mat1(data1, 2, 3);
  std::string data2[6];
  const TensorMap<Tensor<const std::string, 2>> mat2(data2, 2, 3);

  for (int i = 0; i < 6; ++i) {
    std::ostringstream s1;
    s1 << "abc" << i * 3;
    data1[i] = s1.str();
    std::ostringstream s2;
    s2 << "def" << i * 5;
    data2[i] = s2.str();
  }

  Tensor<std::string, 2> rslt1;
  rslt1 = mat1;
  Tensor<std::string, 2> rslt2;
  rslt2 = mat2;

  Tensor<std::string, 2> rslt3 = mat1;
  Tensor<std::string, 2> rslt4 = mat2;

  Tensor<std::string, 2> rslt5(mat1);
  Tensor<std::string, 2> rslt6(mat2);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(rslt1(i, j), data1[i + 2 * j]);
      VERIFY_IS_EQUAL(rslt2(i, j), data2[i + 2 * j]);
      VERIFY_IS_EQUAL(rslt3(i, j), data1[i + 2 * j]);
      VERIFY_IS_EQUAL(rslt4(i, j), data2[i + 2 * j]);
      VERIFY_IS_EQUAL(rslt5(i, j), data1[i + 2 * j]);
      VERIFY_IS_EQUAL(rslt6(i, j), data2[i + 2 * j]);
    }
  }
}

static void test_concat() {
  Tensor<std::string, 2> t1(2, 3);
  Tensor<std::string, 2> t2(2, 3);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::ostringstream s1;
      s1 << "abc" << i + j * 2;
      t1(i, j) = s1.str();
      std::ostringstream s2;
      s2 << "def" << i * 5 + j * 32;
      t2(i, j) = s2.str();
    }
  }

  Tensor<std::string, 2> result = t1.concatenate(t2, 1);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 6);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(result(i, j), t1(i, j));
      VERIFY_IS_EQUAL(result(i, j + 3), t2(i, j));
    }
  }
}

static void test_slices() {
  Tensor<std::string, 2> data(2, 6);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::ostringstream s1;
      s1 << "abc" << i + j * 2;
      data(i, j) = s1.str();
    }
  }

  const Eigen::DSizes<ptrdiff_t, 2> half_size(2, 3);
  const Eigen::DSizes<ptrdiff_t, 2> first_half(0, 0);
  const Eigen::DSizes<ptrdiff_t, 2> second_half(0, 3);

  Tensor<std::string, 2> t1 = data.slice(first_half, half_size);
  Tensor<std::string, 2> t2 = data.slice(second_half, half_size);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      VERIFY_IS_EQUAL(data(i, j), t1(i, j));
      VERIFY_IS_EQUAL(data(i, j + 3), t2(i, j));
    }
  }
}

static void test_additions() {
  Tensor<std::string, 1> data1(3);
  Tensor<std::string, 1> data2(3);
  for (int i = 0; i < 3; ++i) {
    data1(i) = "abc";
    std::ostringstream s1;
    s1 << i;
    data2(i) = s1.str();
  }

  Tensor<std::string, 1> sum = data1 + data2;
  for (int i = 0; i < 3; ++i) {
    std::ostringstream concat;
    concat << "abc" << i;
    std::string expected = concat.str();
    VERIFY_IS_EQUAL(sum(i), expected);
  }
}

static void test_initialization() {
  Tensor<std::string, 2> a(2, 3);
  a.setConstant(std::string("foo"));
  for (int i = 0; i < 2 * 3; ++i) {
    VERIFY_IS_EQUAL(a(i), std::string("foo"));
  }
}

struct StringSize {
  int operator()(const std::string& s) const { return static_cast<int>(s.size()); }
};

// Longer than the small-string buffer, so every string owns heap memory.
struct NumberToString {
  std::string operator()(int number) const {
    std::ostringstream s;
    s << "a string too long for the small buffer " << number;
    return s.str();
  }
};

template <typename Scalar>
struct DefaultGenerator {
  Scalar operator()(const Eigen::array<Eigen::Index, 2>&) const { return Scalar(); }
};

template <typename Expr>
constexpr bool block_access() {
  return Eigen::TensorEvaluator<Expr, Eigen::DefaultDevice>::BlockAccess;
}

// Blocks are materialized in scratch memory that holds no constructed coefficients, so the operations that
// materialize them serve blocks only of a scalar that needs no construction.
template <typename Scalar, bool Expected>
static void check_block_access_of_materializing_operations() {
  Tensor<Scalar, 2> lvalue;
  const Tensor<Scalar, 2>& t = lvalue;
  const Tensor<Scalar, 4> t4;
  const Tensor<Scalar, 5> t5;
  const Eigen::array<Eigen::Index, 2> indices{{1, 1}};
  const Eigen::array<bool, 2> reversed{{false, true}};
  const Eigen::array<std::pair<Eigen::Index, Eigen::Index>, 2> paddings;
  const DefaultGenerator<Scalar> generator;

  STATIC_CHECK((block_access<decltype(t.shuffle(indices))>() == Expected));
  STATIC_CHECK((block_access<decltype(lvalue.shuffle(indices))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.roll(indices))>() == Expected));
  STATIC_CHECK((block_access<decltype(lvalue.roll(indices))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.stridedSlice(indices, indices, indices))>() == Expected));
  STATIC_CHECK((block_access<decltype(lvalue.stridedSlice(indices, indices, indices))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.concatenate(t, 0))>() == Expected));
  STATIC_CHECK((block_access<decltype(lvalue.concatenate(lvalue, 0))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.reverse(reversed))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.pad(paddings))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.generate(generator))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.inflate(indices))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.extract_patches(indices))>() == Expected));
  STATIC_CHECK((block_access<decltype(t4.extract_image_patches(1, 1))>() == Expected));
  STATIC_CHECK((block_access<decltype(t5.extract_volume_patches(1, 1, 1))>() == Expected));
  // These forward the blocks of their argument and materialize the lazy ones. A reshape serves views of any scalar,
  // and the slice of it has no raw buffer for the layout swap to serve instead.
  STATIC_CHECK((block_access<decltype(t.reshape(indices).broadcast(indices))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.reshape(indices).chip(0, 0))>() == Expected));
  STATIC_CHECK((block_access<decltype(t.reshape(indices).slice(indices, indices).swap_layout())>() == Expected));
}

static void test_block_evaluation() {
  check_block_access_of_materializing_operations<std::string, false>();
  check_block_access_of_materializing_operations<std::complex<float>, true>();
  check_block_access_of_materializing_operations<float, true>();

  Tensor<int, 2> numbers(4, 6);
  Tensor<std::string, 2> t(4, 6);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 6; ++j) {
      numbers(i, j) = i + 4 * j;
      t(i, j) = NumberToString()(numbers(i, j));
    }
  }
  const Eigen::array<bool, 2> reverse_first{{true, false}};
  const Eigen::array<bool, 2> reverse_second{{false, true}};
  const Eigen::array<int, 2> transpose{{1, 0}};

  // An lvalue and an rvalue operation that both prefer block evaluation.
  Tensor<std::string, 2> reordered(6, 4);
  reordered.shuffle(transpose) = t.reverse(reverse_second);
  // An arithmetic root, which is tiled, over operands that are not arithmetic.
  Tensor<int, 2> sizes = t.concatenate(t, 0).unaryExpr(StringSize());
  Tensor<int, 2> equal = (t.shuffle(transpose) == reordered.reverse(reverse_first)).cast<int>();
  // A lazy block of strings under an operation that materializes it.
  Tensor<int, 1> row_sizes = numbers.unaryExpr(NumberToString()).chip(1, 0).unaryExpr(StringSize());

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 6; ++j) {
      VERIFY_IS_EQUAL(reordered(j, i), t(i, 5 - j));
      VERIFY_IS_EQUAL(sizes(i, j), StringSize()(t(i, j)));
      VERIFY_IS_EQUAL(sizes(i + 4, j), StringSize()(t(i, j)));
      VERIFY_IS_EQUAL(equal(j, i), 1);
    }
  }
  for (int j = 0; j < 6; ++j) {
    VERIFY_IS_EQUAL(row_sizes(j), StringSize()(t(1, j)));
  }
}

EIGEN_DECLARE_TEST(tensor_of_strings) {
  // Beware: none of this is likely to ever work on a GPU.
  CALL_SUBTEST(test_assign());
  CALL_SUBTEST(test_concat());
  CALL_SUBTEST(test_slices());
  CALL_SUBTEST(test_additions());
  CALL_SUBTEST(test_initialization());
  CALL_SUBTEST(test_block_evaluation());
}

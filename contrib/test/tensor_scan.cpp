// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Igor Babuschkin <igor@babuschk.in>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_TEST_ANNOYING_SCALAR_DONT_THROW

#include "main.h"
#include <limits>
#include <numeric>
#include <Eigen/Tensor>
#include <contrib/Eigen/AutoDiff>

#include "AnnoyingScalar.h"

using Eigen::Tensor;

template <int DataLayout, typename Type = float, bool Exclusive = false>
static void test_1d_scan() {
  int size = 50;
  Tensor<Type, 1, DataLayout> tensor(size);
  tensor.setRandom();
  Tensor<Type, 1, DataLayout> result = tensor.cumsum(0, Exclusive);

  VERIFY_IS_EQUAL(tensor.dimension(0), result.dimension(0));

  float accum = 0;
  for (int i = 0; i < size; i++) {
    if (Exclusive) {
      VERIFY_IS_EQUAL(result(i), accum);
      accum += tensor(i);
    } else {
      accum += tensor(i);
      VERIFY_IS_EQUAL(result(i), accum);
    }
  }

  accum = 1;
  result = tensor.cumprod(0, Exclusive);
  for (int i = 0; i < size; i++) {
    if (Exclusive) {
      VERIFY_IS_EQUAL(result(i), accum);
      accum *= tensor(i);
    } else {
      accum *= tensor(i);
      VERIFY_IS_EQUAL(result(i), accum);
    }
  }
}

template <int DataLayout, typename Type = float>
static void test_4d_scan() {
  int size = 5;
  Tensor<Type, 4, DataLayout> tensor(size, size, size, size);
  tensor.setRandom();

  Tensor<Type, 4, DataLayout> result(size, size, size, size);

  result = tensor.cumsum(0);
  float accum = 0;
  for (int i = 0; i < size; i++) {
    accum += tensor(i, 1, 2, 3);
    VERIFY_IS_EQUAL(result(i, 1, 2, 3), accum);
  }
  result = tensor.cumsum(1);
  accum = 0;
  for (int i = 0; i < size; i++) {
    accum += tensor(1, i, 2, 3);
    VERIFY_IS_EQUAL(result(1, i, 2, 3), accum);
  }
  result = tensor.cumsum(2);
  accum = 0;
  for (int i = 0; i < size; i++) {
    accum += tensor(1, 2, i, 3);
    VERIFY_IS_EQUAL(result(1, 2, i, 3), accum);
  }
  result = tensor.cumsum(3);
  accum = 0;
  for (int i = 0; i < size; i++) {
    accum += tensor(1, 2, 3, i);
    VERIFY_IS_EQUAL(result(1, 2, 3, i), accum);
  }
}

template <int DataLayout>
static void test_tensor_maps() {
  int inputs[20];
  TensorMap<Tensor<int, 1, DataLayout> > tensor_map(inputs, 20);
  setRandomDataInRange(tensor_map, -1000, 1000);

  Tensor<int, 1, DataLayout> result = tensor_map.cumsum(0);

  int accum = 0;
  for (int i = 0; i < 20; ++i) {
    accum += tensor_map(i);
    VERIFY_IS_EQUAL(result(i), accum);
  }
}

// A scan nested in another expression evaluates into a temporary result buffer. For scalars that require
// initialization, that buffer must be constructed before the scan assigns into it, and destructed afterwards.
template <int DataLayout>
static void test_scalar_require_initialization() {
  Tensor<float, 2, DataLayout> tensor(5, 7);
  tensor.setRandom();

  AnnoyingScalar::instances = 0;
  {
    const Tensor<AnnoyingScalar, 2, DataLayout> a_tensor = tensor.template cast<AnnoyingScalar>();
    for (int axis = 0; axis < 2; ++axis) {
      for (bool exclusive : {false, true}) {
        Tensor<float, 2, DataLayout> expected = tensor.cumsum(axis, exclusive) + tensor;
        Tensor<AnnoyingScalar, 2, DataLayout> result = a_tensor.cumsum(axis, exclusive) + a_tensor;
        Tensor<float, 2, DataLayout> actual = result.template cast<float>();
        for (Index i = 0; i < actual.size(); ++i) VERIFY_IS_APPROX(actual.data()[i], expected.data()[i]);

        expected = tensor.cumprod(axis, exclusive) + tensor;
        result = a_tensor.cumprod(axis, exclusive) + a_tensor;
        actual = result.template cast<float>();
        for (Index i = 0; i < actual.size(); ++i) VERIFY_IS_APPROX(actual.data()[i], expected.data()[i]);
      }
    }
  }
  VERIFY_IS_EQUAL(AnnoyingScalar::instances, 0);
}

template <int DataLayout>
static void test_autodiff_scan() {
  using AD = AutoDiffScalar<VectorXd>;
  const int size = 6;
  Tensor<AD, 1, DataLayout> x(size);
  for (int i = 0; i < size; ++i) x(i) = AD(i + 1.0, size, i);

  // sum_{j < i} x(j), with a derivative of 1 with respect to each x(j), j < i.
  const Tensor<AD, 1, DataLayout> before = x.cumsum(0) - x;
  double value = 0;
  for (int i = 0; i < size; ++i) {
    VectorXd derivatives = VectorXd::Zero(size);
    derivatives.head(i).setOnes();
    VERIFY_IS_EQUAL(before(i).value(), value);
    VERIFY_IS_EQUAL(before(i).derivatives(), derivatives);
    value += x(i).value();
  }
}

// A sum that throws from reduce() while armed.
template <typename T>
struct ThrowingSumReducer {
  static bool armed;
  void reduce(const T t, T* accum) const {
    if (armed) EIGEN_THROW_X(std::runtime_error("ThrowingSumReducer"));
    *accum += t;
  }
  T initialize() const { return T(0); }
  T finalize(const T accum) const { return accum; }
};
template <typename T>
bool ThrowingSumReducer<T>::armed = false;

// The input of a scan can own a temporary (here, the buffer of an inner scan). When the scan throws while it is set up,
// the executor has no evaluator to clean up yet, so the scan must release the temporary itself.
template <int DataLayout>
static void test_throwing_scan_releases_input() {
#ifdef EIGEN_EXCEPTIONS
  Tensor<float, 2, DataLayout> tensor(5, 7);
  tensor.setRandom();
  const Tensor<AnnoyingScalar, 2, DataLayout> a_tensor = tensor.template cast<AnnoyingScalar>();
  AnnoyingScalar::instances = 0;
  const int base = AnnoyingScalar::instances;
  {
    Tensor<AnnoyingScalar, 2, DataLayout> result(5, 7);
    result.setZero();
    const auto scan = [&](const Tensor<AnnoyingScalar, 2, DataLayout>& in) {
      return in.cumsum(0).scan(1, ThrowingSumReducer<AnnoyingScalar>());
    };
    ThrowingSumReducer<AnnoyingScalar>::armed = true;
    bool threw = false;
    try {
      result = scan(a_tensor);  // same shape: the scan writes into the destination directly
    } catch (const std::runtime_error&) {
      threw = true;
    }
    VERIFY(threw);
    VERIFY_IS_EQUAL(AnnoyingScalar::instances, base + 35);  // result only

    threw = false;
    try {
      result = scan(a_tensor) + a_tensor;  // the scan is a subexpression, so it allocates a temporary output
    } catch (const std::runtime_error&) {
      threw = true;
    }
    ThrowingSumReducer<AnnoyingScalar>::armed = false;
    VERIFY(threw);
    VERIFY_IS_EQUAL(AnnoyingScalar::instances, base + 35);
  }
  VERIFY_IS_EQUAL(AnnoyingScalar::instances, base);
#endif
}

// Scalars that require initialization are constructed in temporaries from the host only for devices whose temporaries
// are host memory. A GPU or SYCL device allocates them in device memory, which host code must not construct in.
static void test_temporaries_are_host_memory() {
  STATIC_CHECK((internal::temporaries_are_host_memory<DefaultDevice>::value));
  STATIC_CHECK((internal::temporaries_are_host_memory<const DefaultDevice>::value));
  STATIC_CHECK((!internal::temporaries_are_host_memory<GpuDevice>::value));
  STATIC_CHECK((!internal::temporaries_are_host_memory<const GpuDevice>::value));
  STATIC_CHECK((!internal::temporaries_are_host_memory<SyclDevice>::value));
}

EIGEN_DECLARE_TEST(tensor_scan) {
  CALL_SUBTEST((test_1d_scan<ColMajor, float, true>()));
  CALL_SUBTEST((test_1d_scan<ColMajor, float, false>()));
  CALL_SUBTEST((test_1d_scan<RowMajor, float, true>()));
  CALL_SUBTEST((test_1d_scan<RowMajor, float, false>()));
  CALL_SUBTEST(test_4d_scan<ColMajor>());
  CALL_SUBTEST(test_4d_scan<RowMajor>());
  CALL_SUBTEST(test_tensor_maps<ColMajor>());
  CALL_SUBTEST(test_tensor_maps<RowMajor>());
  CALL_SUBTEST(test_scalar_require_initialization<ColMajor>());
  CALL_SUBTEST(test_scalar_require_initialization<RowMajor>());
  CALL_SUBTEST(test_autodiff_scan<ColMajor>());
  CALL_SUBTEST(test_autodiff_scan<RowMajor>());
  CALL_SUBTEST(test_throwing_scan_releases_input<ColMajor>());
  CALL_SUBTEST(test_throwing_scan_releases_input<RowMajor>());
  CALL_SUBTEST(test_temporaries_are_host_memory());
}

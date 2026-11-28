// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_USE_THREADS 1

#include "main.h"
#include <Eigen/ThreadPool>

namespace Eigen {
namespace internal {
// conveniently control vectorization logic
template <typename Scalar, bool Vectorize>
struct scalar_dummy_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar operator()(const Scalar& a) const { return a; }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return a;
  }
};
template <typename Scalar, bool Vectorize>
struct functor_traits<scalar_dummy_op<Scalar, Vectorize> > {
  enum { Cost = 1'000'000, PacketAccess = Vectorize && packet_traits<Scalar>::Vectorizable };
};
}  // namespace internal
}  // namespace Eigen

template <typename PlainObject>
void test_threaded_assignment(const PlainObject&, Index rows = PlainObject::RowsAtCompileTime,
                              Index cols = PlainObject::ColsAtCompileTime) {
  using Scalar = typename PlainObject::Scalar;
  using VectorizationOff = internal::scalar_dummy_op<Scalar, false>;
  using VectorizationOn = internal::scalar_dummy_op<Scalar, true>;

  int threads = 4;
  ThreadPool pool(threads);
  CoreThreadPoolDevice threadPoolDevice(pool);

  PlainObject dst(rows, cols), ref(rows, cols), rhs(rows, cols);
  rhs.setRandom();
  const auto rhs_xpr = rhs + rhs;

  // linear access
  dst.setRandom();
  ref.setRandom();
  ref = rhs_xpr.unaryExpr(VectorizationOff());
  dst.device(threadPoolDevice) = rhs_xpr.unaryExpr(VectorizationOff());
  VERIFY_IS_CWISE_EQUAL(ref, dst);

  ref = rhs_xpr.unaryExpr(VectorizationOn());
  dst.device(threadPoolDevice) = rhs_xpr.unaryExpr(VectorizationOn());
  VERIFY_IS_CWISE_EQUAL(ref, dst);

  EIGEN_IF_CONSTEXPR (PlainObject::SizeAtCompileTime == Dynamic) {
    using RealScalar = typename NumTraits<Scalar>::Real;
    Matrix<RealScalar, Dynamic, 1> raw_buf(2 * rows * cols + 2);
    Map<PlainObject, Unaligned> unaligned_dst(reinterpret_cast<Scalar*>(raw_buf.data() + 1), rows, cols);
    unaligned_dst.setZero();
    unaligned_dst.device(threadPoolDevice) = rhs_xpr.unaryExpr(VectorizationOn());
    VERIFY_IS_CWISE_EQUAL(ref, unaligned_dst);
  }

  // outer-inner access
  Index blockRows = numext::maxi(Index(1), rows - 1);
  Index blockCols = numext::maxi(Index(1), cols - 1);
  dst.setRandom();
  ref.setRandom();
  ref.bottomRightCorner(blockRows, blockCols) =
      rhs_xpr.bottomRightCorner(blockRows, blockCols).unaryExpr(VectorizationOff());
  dst.bottomRightCorner(blockRows, blockCols).device(threadPoolDevice) =
      rhs_xpr.bottomRightCorner(blockRows, blockCols).unaryExpr(VectorizationOff());
  VERIFY_IS_CWISE_EQUAL(ref.bottomRightCorner(blockRows, blockCols), dst.bottomRightCorner(blockRows, blockCols));

  ref.setZero();
  dst.setZero();
  ref.bottomRightCorner(blockRows, blockCols) =
      rhs_xpr.bottomRightCorner(blockRows, blockCols).unaryExpr(VectorizationOn());
  dst.bottomRightCorner(blockRows, blockCols).device(threadPoolDevice) =
      rhs_xpr.bottomRightCorner(blockRows, blockCols).unaryExpr(VectorizationOn());
  VERIFY_IS_CWISE_EQUAL(ref.bottomRightCorner(blockRows, blockCols), dst.bottomRightCorner(blockRows, blockCols));
}

// A destination that is less aligned than its packets: the packet loop starts at its first aligned coefficient, where
// the narrower source of a cast() is aligned on fewer bytes. square() reads the source packet twice, which keeps
// optimizers from folding its load.
void test_threaded_cast_underaligned_dst() {
  ThreadPool pool(4);
  CoreThreadPoolDevice threadPoolDevice(pool);

  constexpr Index Size = 1003;
  constexpr Index MaxOffset = 64 / Index(sizeof(double));
  EIGEN_ALIGN_TO_BOUNDARY(64) float srcData[Size];
  EIGEN_ALIGN_TO_BOUNDARY(64) double dstData[MaxOffset + Size];
  Map<ArrayXf, Aligned16> src(srcData, Size);
  src.setRandom();
  const ArrayXd ref = src.square().cast<double>();
  for (Index offset = 0; offset <= MaxOffset; offset += Aligned16 / Index(sizeof(double))) {
    Map<ArrayXd, Aligned16> dst(dstData + offset, Size);
    dst.setZero();
    dst.device(threadPoolDevice) = src.square().cast<double>();
    VERIFY_IS_CWISE_EQUAL(ref, dst);
  }
}

EIGEN_DECLARE_TEST(assignment_threaded) {
  CALL_SUBTEST(test_threaded_cast_underaligned_dst());
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST(test_threaded_assignment(MatrixXd(), 123, 123));
    CALL_SUBTEST(test_threaded_assignment(MatrixXcf(), 123, 123));
    CALL_SUBTEST(test_threaded_assignment(MatrixXcd(), 123, 123));
    CALL_SUBTEST(test_threaded_assignment(Matrix<float, 16, 16>()));
  }
}

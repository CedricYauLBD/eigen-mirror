// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"
#include <Eigen/LU>
#include <memory>
#include <utility>

namespace solver_move_test {

struct CopyError {};

class HeapScalar {
 public:
  static int copies;
  static bool forbid_copy;

  HeapScalar(double value = 0) : value_(new double(value)) {}
  HeapScalar(const HeapScalar& other) : value_(new double(*other.value_)) { record_copy(); }
  HeapScalar& operator=(const HeapScalar& other) {
    record_copy();
    value_.reset(new double(*other.value_));
    return *this;
  }
  HeapScalar(HeapScalar&&) noexcept = default;
  HeapScalar& operator=(HeapScalar&&) noexcept = default;

  const double* data() const { return value_.get(); }

 private:
  static void record_copy() {
    ++copies;
#ifdef EIGEN_EXCEPTIONS
    if (forbid_copy) throw CopyError();
#endif
  }

  std::unique_ptr<double> value_;
};

int HeapScalar::copies = 0;
bool HeapScalar::forbid_copy = false;

}  // namespace solver_move_test

namespace Eigen {
template <>
struct NumTraits<solver_move_test::HeapScalar> : NumTraits<double> {
  using Real = solver_move_test::HeapScalar;
  using NonInteger = Real;
  using Nested = Real;
  static constexpr bool RequireInitialization = true;
};
}  // namespace Eigen

namespace solver_move_test {

using MatrixType = Matrix<HeapScalar, Dynamic, Dynamic>;
using Solver = FullPivLU<MatrixType>;

struct InspectedSolver : Solver {
  InspectedSolver(Index rows, Index cols) : Solver(rows, cols) { m_det_pq = 1; }
  const HeapScalar& threshold_storage() const { return m_prescribedThreshold; }
  const HeapScalar& pivot_storage() const { return m_maxpivot; }
  const HeapScalar* matrix_storage() const { return m_lu.data(); }
};

void check_rank_metadata_moves() {
  STATIC_CHECK((std::is_nothrow_move_constructible<Solver>::value));

  InspectedSolver source(3, 3);
  source.setThreshold(HeapScalar(7));
  const double* threshold = source.threshold_storage().data();
  const double* pivot = source.pivot_storage().data();
  const HeapScalar* coefficients = source.matrix_storage();
  InspectedSolver target(2, 2);

  HeapScalar::copies = 0;
  HeapScalar::forbid_copy = true;
  InspectedSolver moved(std::move(source));
  VERIFY_IS_EQUAL(HeapScalar::copies, 0);
  VERIFY(moved.threshold_storage().data() == threshold);
  VERIFY(moved.pivot_storage().data() == pivot);
  VERIFY(moved.matrix_storage() == coefficients);

  target = std::move(moved);
  VERIFY_IS_EQUAL(HeapScalar::copies, 0);
  VERIFY(target.threshold_storage().data() == threshold);
  VERIFY(target.pivot_storage().data() == pivot);
  VERIFY(target.matrix_storage() == coefficients);
  VERIFY_IS_EQUAL(*target.threshold_storage().data(), 7.0);
  HeapScalar::forbid_copy = false;
}

}  // namespace solver_move_test

EIGEN_DECLARE_TEST(solver_move) { solver_move_test::check_rank_metadata_moves(); }

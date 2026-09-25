// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include <Eigen/SVD>
#include <utility>

template <typename Svd>
void check_svd_copy_and_move() {
  using MatrixType = typename Svd::MatrixType;
  STATIC_CHECK((std::is_copy_assignable<SVDBase<Svd>>::value));
  STATIC_CHECK((std::is_move_assignable<SVDBase<Svd>>::value));
  const MatrixType matrix = MatrixType::Random(5, 3);
  Svd source(matrix);
  VERIFY_IS_EQUAL(source.info(), Success);

  const auto* singularValues = source.singularValues().data();
  const auto* u = source.matrixU().data();
  const auto* v = source.matrixV().data();
  Svd copy(source);
  Svd assigned;
  assigned = source;
  VERIFY(copy.singularValues().data() != singularValues);
  VERIFY(assigned.singularValues().data() != singularValues);
  VERIFY_IS_EQUAL(copy.singularValues(), source.singularValues());
  VERIFY_IS_EQUAL(copy.matrixU(), source.matrixU());
  VERIFY_IS_EQUAL(copy.matrixV(), source.matrixV());
  VERIFY_IS_EQUAL(assigned.singularValues(), source.singularValues());
  VERIFY_IS_EQUAL(assigned.matrixU(), source.matrixU());
  VERIFY_IS_EQUAL(assigned.matrixV(), source.matrixV());

  // Moving a computed SVD must transfer every dynamic workspace without allocating.
  internal::set_is_malloc_allowed(false);
  Svd moved(std::move(source));
  internal::set_is_malloc_allowed(true);
  VERIFY(moved.singularValues().data() == singularValues);
  VERIFY(moved.matrixU().data() == u);
  VERIFY(moved.matrixV().data() == v);
  VERIFY_IS_EQUAL(moved.info(), Success);
  VERIFY_IS_EQUAL(moved.singularValues(), copy.singularValues());
  VERIFY_IS_EQUAL(moved.matrixU(), copy.matrixU());
  VERIFY_IS_EQUAL(moved.matrixV(), copy.matrixV());

  Svd target(MatrixType::Random(4, 2));
  internal::set_is_malloc_allowed(false);
  target = std::move(moved);
  internal::set_is_malloc_allowed(true);
  VERIFY(target.singularValues().data() == singularValues);
  VERIFY(target.matrixU().data() == u);
  VERIFY(target.matrixV().data() == v);
  VERIFY_IS_EQUAL(target.info(), Success);
  VERIFY_IS_EQUAL(target.singularValues(), copy.singularValues());
  VERIFY_IS_EQUAL(target.matrixU(), copy.matrixU());
  VERIFY_IS_EQUAL(target.matrixV(), copy.matrixV());
}

EIGEN_DECLARE_TEST(svd_move) {
  constexpr int options = ComputeThinU | ComputeThinV;
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1((check_svd_copy_and_move<JacobiSVD<MatrixXd, options>>()));
    CALL_SUBTEST_2((check_svd_copy_and_move<JacobiSVD<MatrixXcd, options>>()));
    CALL_SUBTEST_3((check_svd_copy_and_move<BDCSVD<MatrixXd, options>>()));
    CALL_SUBTEST_4((check_svd_copy_and_move<BDCSVD<MatrixXcd, options>>()));
  }
}

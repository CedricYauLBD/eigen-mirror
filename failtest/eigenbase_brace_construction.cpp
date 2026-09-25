// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <Eigen/Core>

int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Eigen::EigenBase<Eigen::MatrixXd> expression{};
#else
  Eigen::MatrixXd expression{};
#endif
  (void)expression;
}

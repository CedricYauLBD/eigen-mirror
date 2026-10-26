// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 Joel Holdsworth <joel.holdsworth@vcatechnology.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include <sstream>

#include "main.h"

template <typename Scalar>
struct check_ostream_impl {
  static void run() {
    const Array<Scalar, 1, 1> array(123);
    std::ostringstream ss;
    ss << array;
    VERIFY(ss.str() == "123");

    check_ostream_impl<std::complex<Scalar> >::run();
  };
};

template <>
struct check_ostream_impl<bool> {
  static void run() {
    const Array<bool, 1, 2> array(1, 0);
    std::ostringstream ss;
    ss << array;
    VERIFY(ss.str() == "1  0");
  };
};

template <typename Scalar>
struct check_ostream_impl<std::complex<Scalar> > {
  static void run() {
    const Array<std::complex<Scalar>, 1, 1> array(std::complex<Scalar>(12, 34));
    std::ostringstream ss;
    ss << array;
    VERIFY(ss.str() == "(12,34)");
  };
};

template <typename Scalar>
static void check_ostream() {
  check_ostream_impl<Scalar>::run();
}

static std::string format_2x2(const std::string& rowSeparator = "\n", const std::string& rowPrefix = "[",
                              const std::string& rowSuffix = "]", const std::string& matPrefix = "",
                              const std::string& matSuffix = "", int flags = 0) {
  const auto matrix = (Matrix2i() << 6473, 27, 1, 22334).finished();
  std::ostringstream ss;
  const IOFormat fmt(StreamPrecision, flags, ", ", rowSeparator, rowPrefix, rowSuffix, matPrefix, matSuffix);
  ss << matrix.format(fmt);
  return ss.str();
}

static void check_ioformat_standard() {
  const std::string found = format_2x2();
  const std::string expected =
      "[ 6473,    27]\n"
      "[    1, 22334]";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_separator_newline() {
  const std::string found = format_2x2("\nNext\n");
  const std::string expected =
      "[ 6473,    27]\n"
      "Next\n"
      "[    1, 22334]";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_separator_multiple_newlines() {
  const std::string found = format_2x2("\nNext\n  ", "[", "]", "Matrix = [", "]");
  const std::string expected =
      "Matrix = [[ 6473,    27]\n"
      "Next\n"
      "          [    1, 22334]]";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_prefix_newline() {
  const std::string found = format_2x2("\n", "[\n", "]");
  const std::string expected =
      "[\n"
      " 6473,    27]\n"
      "[\n"
      "    1, 22334]";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_suffix_newline() {
  const std::string found = format_2x2("\n", "[", "\n]");
  const std::string expected =
      "[ 6473,    27\n"
      "]\n"
      "[    1, 22334\n"
      "]";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_rows_same_line() {
  const std::string found = format_2x2(", ", "[", "]", "[", "]");
  const std::string expected = "[[ 6473,    27], [    1, 22334]]";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_separator_indent() {
  const std::string found = format_2x2(",\n  ", "[", "]", "Matrix = [", "]");
  const std::string expected =
      "Matrix = [[ 6473,    27],\n"
      "          [    1, 22334]]";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_prefix_only_newline() {
  const std::string found = format_2x2("", "\n[", "]", "Matrix =");
  const std::string expected =
      "Matrix =\n"
      "[ 6473,    27]\n"
      "[    1, 22334]";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_mat_prefix_newline() {
  const std::string found = format_2x2("\n", "[", "]", "Matrix =\n [", "].");
  const std::string expected =
      "Matrix =\n"
      " [[ 6473,    27]\n"
      "  [    1, 22334]].";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_row_suffix_separator() {
  const std::string found = format_2x2(", ", "[", "\n]", "Matrix = [", "]");
  const std::string expected =
      "Matrix = [[ 6473,    27\n"
      "],        [    1, 22334\n"
      "]]";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_dont_align_cols() {
  const std::string found = format_2x2("\n", "[", "]", "Matrix = [", "].", DontAlignCols);
  const std::string expected =
      "Matrix = [[6473, 27]\n"
      "[1, 22334]].";
  VERIFY_IS_EQUAL(found, expected);
}

static void check_ioformat_carried_exceeds_indent() {
  const std::string found = format_2x2("\n ---carried--- ", "[", "]", "Matrix [", "]");
  const std::string expected =
      "Matrix [[ 6473,    27]\n"
      " ---carried--- [    1, 22334]]";
  VERIFY_IS_EQUAL(found, expected);
}

EIGEN_DECLARE_TEST(io) {
  CALL_SUBTEST(check_ostream<bool>());
  CALL_SUBTEST(check_ostream<float>());
  CALL_SUBTEST(check_ostream<double>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int8_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint8_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int16_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint16_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int32_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint32_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::int64_t>());
  CALL_SUBTEST(check_ostream<Eigen::numext::uint64_t>());
  CALL_SUBTEST(check_ioformat_standard());
  CALL_SUBTEST(check_ioformat_row_separator_newline());
  CALL_SUBTEST(check_ioformat_row_separator_multiple_newlines());
  CALL_SUBTEST(check_ioformat_row_prefix_newline());
  CALL_SUBTEST(check_ioformat_row_suffix_newline());
  CALL_SUBTEST(check_ioformat_rows_same_line());
  CALL_SUBTEST(check_ioformat_row_separator_indent());
  CALL_SUBTEST(check_ioformat_row_prefix_only_newline());
  CALL_SUBTEST(check_ioformat_mat_prefix_newline());
  CALL_SUBTEST(check_ioformat_row_suffix_separator());
  CALL_SUBTEST(check_ioformat_dont_align_cols());
  CALL_SUBTEST(check_ioformat_carried_exceeds_indent());
}

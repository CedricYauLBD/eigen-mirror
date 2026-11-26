// SPDX-FileCopyrightText: 2008-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSE_TEST_HELPERS_H
#define EIGEN_SPARSE_TEST_HELPERS_H

#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include <Eigen/SparseCore>

namespace Eigen {

template <typename T, std::enable_if_t<(T::Flags & RowMajorBit) != 0, int> = 0>
auto innervec(T& matrix, Index index) -> decltype(matrix.row(index)) {
  return matrix.row(index);
}

template <typename T, std::enable_if_t<(T::Flags & RowMajorBit) == 0, int> = 0>
auto innervec(T& matrix, Index index) -> decltype(matrix.col(index)) {
  return matrix.col(index);
}

template <typename T>
typename T::Map make_map(T& matrix) {
  return typename T::Map(matrix.rows(), matrix.cols(), matrix.nonZeros(), matrix.outerIndexPtr(),
                         matrix.innerIndexPtr(), matrix.valuePtr(), matrix.innerNonZeroPtr());
}

template <typename T, std::enable_if_t<(T::Flags & RowMajorBit) != 0, int> = 0>
auto innerpanel(T& matrix, Index start, Index size) -> decltype(matrix.middleRows(start, size)) {
  return matrix.middleRows(start, size);
}

template <typename T, std::enable_if_t<(T::Flags & RowMajorBit) == 0, int> = 0>
auto innerpanel(T& matrix, Index start, Index size) -> decltype(matrix.middleCols(start, size)) {
  return matrix.middleCols(start, size);
}

// Access by (outer, inner) index, whatever the storage order of the sparse matrix or expression.
template <typename T>
typename T::Scalar& insert_by_outer(T& matrix, Index outer, Index inner) {
  return T::IsRowMajor ? matrix.insert(outer, inner) : matrix.insert(inner, outer);
}

template <typename T>
typename T::Scalar coeff_by_outer(const T& matrix, Index outer, Index inner) {
  return T::IsRowMajor ? matrix.coeff(outer, inner) : matrix.coeff(inner, outer);
}

template <typename T>
typename T::Scalar& coeffRef_by_outer(T& matrix, Index outer, Index inner) {
  return T::IsRowMajor ? matrix.coeffRef(outer, inner) : matrix.coeffRef(inner, outer);
}

// Seven magnitudes from 1 up to the largest finite RealScalar and down to a subnormal, for checks that must not
// overflow or underflow in a norm or a difference. highest / 2 makes a - (-a) overflow although a and -a are finite.
template <typename RealScalar>
std::array<RealScalar, 7> extreme_scales() {
  return {{RealScalar(1), NumTraits<RealScalar>::highest() / RealScalar(2),
           NumTraits<RealScalar>::highest() / RealScalar(8), numext::sqrt(NumTraits<RealScalar>::highest()),
           RealScalar(1) / numext::sqrt(NumTraits<RealScalar>::highest()),
           (std::numeric_limits<RealScalar>::min)() * RealScalar(16),
           std::numeric_limits<RealScalar>::denorm_min() * RealScalar(64)}};
}

template <typename MatrixType>
void init_sparse_diagonal(MatrixType& matrix) {
  using Scalar = typename MatrixType::Scalar;
  matrix.resize(4, 4);
  matrix.insert(0, 0) = Scalar(10);
  matrix.insert(1, 1) = Scalar(20);
  matrix.insert(2, 2) = Scalar(30);
  matrix.insert(3, 3) = Scalar(40);
  matrix.makeCompressed();
}

// A single-pass iterator over a vector of triplets. All copies share one position, as the iterators of a stream do,
// so a range that has been counted or scanned cannot be read again.
template <typename Triplet>
struct single_pass_triplet_iterator {
  using iterator_category = std::input_iterator_tag;
  using value_type = Triplet;
  using difference_type = std::ptrdiff_t;
  using pointer = const Triplet*;
  using reference = const Triplet&;

  const std::vector<Triplet>* source;
  std::size_t* next;

  reference operator*() const { return (*source)[*next]; }
  pointer operator->() const { return &(*source)[*next]; }
  single_pass_triplet_iterator& operator++() {
    ++*next;
    return *this;
  }
  single_pass_triplet_iterator operator++(int) {
    single_pass_triplet_iterator old = *this;
    ++*next;
    return old;
  }
  bool operator==(const single_pass_triplet_iterator&) const { return *next == source->size(); }
  bool operator!=(const single_pass_triplet_iterator& other) const { return !(*this == other); }
};

}  // namespace Eigen

#endif  // EIGEN_SPARSE_TEST_HELPERS_H

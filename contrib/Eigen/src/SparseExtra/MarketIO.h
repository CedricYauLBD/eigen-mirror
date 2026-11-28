// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2012 Desire NUENTSA WAKAM <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPARSE_MARKET_IO_H
#define EIGEN_SPARSE_MARKET_IO_H

#include <iostream>
#include <vector>

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
template <typename Scalar, typename StorageIndex>
inline void GetMarketLine(const char* line, StorageIndex& i, StorageIndex& j, Scalar& value) {
  std::stringstream sline(line);
  sline >> i >> j >> value;
}

template <>
inline void GetMarketLine(const char* line, int& i, int& j, float& value) {
  std::sscanf(line, "%d %d %g", &i, &j, &value);
}

template <>
inline void GetMarketLine(const char* line, int& i, int& j, double& value) {
  std::sscanf(line, "%d %d %lg", &i, &j, &value);
}

template <>
inline void GetMarketLine(const char* line, int& i, int& j, std::complex<float>& value) {
  std::sscanf(line, "%d %d %g %g", &i, &j, &numext::real_ref(value), &numext::imag_ref(value));
}

template <>
inline void GetMarketLine(const char* line, int& i, int& j, std::complex<double>& value) {
  std::sscanf(line, "%d %d %lg %lg", &i, &j, &numext::real_ref(value), &numext::imag_ref(value));
}

template <typename Scalar, typename StorageIndex>
inline void GetMarketLine(const char* line, StorageIndex& i, StorageIndex& j, std::complex<Scalar>& value) {
  std::stringstream sline(line);
  Scalar valR, valI;
  sline >> i >> j >> valR >> valI;
  value = std::complex<Scalar>(valR, valI);
}

// Return whether the line held the coefficient; val is unspecified otherwise.
template <typename RealScalar>
inline bool GetDenseElt(const std::string& line, RealScalar& val) {
  std::istringstream newline(line);
  return bool(newline >> val);
}

template <typename RealScalar>
inline bool GetDenseElt(const std::string& line, std::complex<RealScalar>& val) {
  RealScalar valR, valI;
  std::istringstream newline(line);
  if (!(newline >> valR >> valI)) return false;
  val = std::complex<RealScalar>(valR, valI);
  return true;
}

// Whether resize() can give DenseType other dimensions. A plain Matrix or Array reallocates, and Transpose and the
// array/matrix wrappers forward to the expression they wrap. For anything else (Map, Ref, Block, ...) it is
// DenseBase::resize(), which only asserts that the dimensions are unchanged.
struct is_resizable_dense_impl {
  template <typename Derived>
  static std::true_type test(const PlainObjectBase<Derived>*);
  static std::false_type test(const void*);
};

template <typename DenseType>
struct is_resizable_dense : decltype(is_resizable_dense_impl::test(std::declval<DenseType*>())) {};

template <typename XprType>
struct is_resizable_dense<Transpose<XprType>> : is_resizable_dense<XprType> {};

template <typename XprType>
struct is_resizable_dense<ArrayWrapper<XprType>> : is_resizable_dense<XprType> {};

template <typename XprType>
struct is_resizable_dense<MatrixWrapper<XprType>> : is_resizable_dense<XprType> {};

template <typename Scalar>
inline void putMarketHeader(std::string& header, int sym) {
  header = "%%MatrixMarket matrix coordinate ";
  EIGEN_IF_CONSTEXPR ((std::is_same<Scalar, std::complex<float> >::value ||
                       std::is_same<Scalar, std::complex<double> >::value)) {
    header += " complex";
    if (sym == Symmetric)
      header += " symmetric";
    else if (sym == SelfAdjoint)
      header += " Hermitian";
    else
      header += " general";
  } else {
    header += " real";
    if (sym == Symmetric)
      header += " symmetric";
    else
      header += " general";
  }
}

template <typename Scalar, typename StorageIndex>
inline void PutMatrixElt(Scalar value, StorageIndex row, StorageIndex col, std::ofstream& out) {
  out << row << " " << col << " " << value << "\n";
}
template <typename Scalar, typename StorageIndex>
inline void PutMatrixElt(std::complex<Scalar> value, StorageIndex row, StorageIndex col, std::ofstream& out) {
  out << row << " " << col << " " << value.real() << " " << value.imag() << "\n";
}

template <typename Scalar>
inline void putDenseElt(Scalar value, std::ofstream& out) {
  out << value << "\n";
}
template <typename Scalar>
inline void putDenseElt(std::complex<Scalar> value, std::ofstream& out) {
  out << value.real() << " " << value.imag() << "\n";
}

}  // end namespace internal

/**
 * \ingroup SparseExtra_Module
 * @brief Reads the header of a matrixmarket file and determines the properties of a matrix
 *
 * @param filename of the file
 * @param sym if the matrix is hermitian, symmetric or none of the latter (sym=0)
 * @param iscomplex if the matrix has complex or real coefficients
 * @param isdense if the matrix is dense or sparse
 * @return true if the file was found
 */
inline bool getMarketHeader(const std::string& filename, int& sym, bool& iscomplex, bool& isdense) {
  sym = 0;
  iscomplex = false;
  isdense = false;
  std::ifstream in(filename.c_str(), std::ios::in);
  if (!in) return false;

  std::string line;
  // The matrix header is always the first line in the file
  std::getline(in, line);
  eigen_assert(in.good());

  std::stringstream fmtline(line);
  std::string substr[5];
  fmtline >> substr[0] >> substr[1] >> substr[2] >> substr[3] >> substr[4];
  if (substr[2].compare("array") == 0) isdense = true;
  if (substr[3].compare("complex") == 0) iscomplex = true;
  if (substr[4].compare("symmetric") == 0)
    sym = Symmetric;
  else if (substr[4].compare("Hermitian") == 0)
    sym = SelfAdjoint;

  return true;
}
/**
 * \ingroup SparseExtra_Module
 * @brief Loads a sparse matrix from a matrixmarket format file.
 *
 * @tparam SparseMatrixType to read into, symmetries are not supported
 * @param mat SparseMatrix to read into, current values are overwritten
 * @param filename to parse matrix from
 * @return returns true if file exists. Returns false if the parsing did not succeed.
 */
template <typename SparseMatrixType>
bool loadMarket(SparseMatrixType& mat, const std::string& filename) {
  using Scalar = typename SparseMatrixType::Scalar;
  using StorageIndex = typename SparseMatrixType::StorageIndex;
  std::ifstream input(filename.c_str(), std::ios::in);
  if (!input) return false;

  char rdbuffer[4096];
  input.rdbuf()->pubsetbuf(rdbuffer, 4096);

  const int maxBuffersize = 2048;
  char buffer[maxBuffersize];

  bool readsizes = false;

  using T = Triplet<Scalar, StorageIndex>;
  std::vector<T> elements;

  Index M(-1), N(-1), NNZ(-1);
  Index count = 0;
  while (input.getline(buffer, maxBuffersize)) {
    // skip comments
    // NOTE An appropriate test should be done on the header to get the symmetry
    if (buffer[0] == '%') continue;

    if (!readsizes) {
      std::stringstream line(buffer);
      line >> M >> N >> NNZ;
      if (M > 0 && N > 0) {
        readsizes = true;
        mat.resize(M, N);
        mat.reserve(NNZ);
        elements.reserve(NNZ);
      }
    } else {
      StorageIndex i(-1), j(-1);
      Scalar value;
      internal::GetMarketLine(buffer, i, j, value);

      i--;
      j--;
      if (i >= 0 && j >= 0 && i < M && j < N) {
        ++count;
        elements.push_back(T(i, j, value));
      } else {
        std::cerr << "Invalid read: " << i << "," << j << "\n";
        return false;
      }
    }
  }

  mat.setFromTriplets(elements.begin(), elements.end());
  if (count != NNZ) {
    std::cerr << count << "!=" << NNZ << "\n";
    return false;
  }
  input.close();
  return true;
}

/**
 * \ingroup SparseExtra_Module
 * @brief Loads a dense Matrix or Vector from a matrixmarket file. \a mat is resized to the dimensions in the file. A
 * fixed-size matrix, or an expression that cannot be resized such as a Map, a Ref or a Block, must already have them.
 *
 * @tparam DenseMatrixType to read into
 * @param mat DenseMatrix to read into, current values are overwritten, symmetries are not supported. It is left
 * untouched if the parsing does not succeed.
 * @param filename to parse matrix from
 * @return true if parsing was successful. Returns false if the parsing did not succeed, or if \a mat cannot take the
 * dimensions in the file.
 */
template <typename DenseType>
bool loadMarketDense(DenseType& mat, const std::string& filename) {
  using Scalar = typename DenseType::Scalar;
  std::ifstream in(filename.c_str(), std::ios::in);
  if (!in) return false;

  std::string line;
  Index rows(0), cols(0);
  do {  // Skip comments
    std::getline(in, line);
    eigen_assert(in.good());
  } while (line[0] == '%');
  std::istringstream newline(line);
  newline >> rows >> cols;

  bool sizes_not_positive = (rows < 1 || cols < 1);
  bool wrong_input_rows = (DenseType::MaxRowsAtCompileTime != Dynamic && rows > DenseType::MaxRowsAtCompileTime) ||
                          (DenseType::RowsAtCompileTime != Dynamic && rows != DenseType::RowsAtCompileTime);
  bool wrong_input_cols = (DenseType::MaxColsAtCompileTime != Dynamic && cols > DenseType::MaxColsAtCompileTime) ||
                          (DenseType::ColsAtCompileTime != Dynamic && cols != DenseType::ColsAtCompileTime);
  bool wrong_view_size = !internal::is_resizable_dense<DenseType>::value && (rows != mat.rows() || cols != mat.cols());

  if (sizes_not_positive || wrong_input_rows || wrong_input_cols || wrong_view_size) {
    if (sizes_not_positive) {
      std::cerr << "non-positive row or column size in file" << filename << "\n";
    } else {
      std::cerr << "Input matrix can not be resized to" << rows << " x " << cols << "as given in " << filename << "\n";
    }
    in.close();
    return false;
  }

  // The size line is untrusted: allocate rows * cols only once the file has supplied that many coefficients.
  std::vector<Scalar> values;
  Index row = 0;
  Index col = 0;
  Scalar value;
  while (std::getline(in, line) && (row < rows) && (col < cols)) {
    if (!internal::GetDenseElt(line, value)) {
      std::cerr << "Malformed element in file " << filename << "\n";
      in.close();
      return false;
    }
    values.push_back(value);
    row++;
    if (row == rows) {
      row = 0;
      col++;
    }
  }
  in.close();
  if (col != cols) {
    std::cerr << "Unable to read all elements from file " << filename << "\n";
    return false;
  }

  mat.resize(rows, cols);
  // matrixmarket format is column major
  auto next_value = values.cbegin();
  for (col = 0; col < cols; col++) {
    for (row = 0; row < rows; row++) {
      mat(row, col) = *next_value++;
    }
  }
  return true;
}
/**
 * \ingroup SparseExtra_Module
 * @brief Same functionality as loadMarketDense, deprecated
 */
template <typename VectorType>
bool loadMarketVector(VectorType& vec, const std::string& filename) {
  return loadMarketDense(vec, filename);
}

/**
 * \ingroup SparseExtra_Module
 * @brief writes a sparse Matrix to a matrixmarket format file
 *
 * @tparam SparseMatrixType to write to file
 * @param mat matrix to write to file
 * @param filename filename to write to
 * @param sym at the moment no symmetry operations are supported
 * @return true if writing succeeded
 */
template <typename SparseMatrixType>
bool saveMarket(const SparseMatrixType& mat, const std::string& filename, int sym = 0) {
  using Scalar = typename SparseMatrixType::Scalar;
  using RealScalar = typename SparseMatrixType::RealScalar;
  std::ofstream out(filename.c_str(), std::ios::out);
  if (!out) return false;

  out.flags(std::ios_base::scientific);
  out.precision(std::numeric_limits<RealScalar>::digits10 + 2);
  std::string header;
  internal::putMarketHeader<Scalar>(header, sym);
  out << header << std::endl;
  out << mat.rows() << " " << mat.cols() << " " << mat.nonZeros() << "\n";
  int count = 0;
  EIGEN_UNUSED_VARIABLE(count);
  for (int j = 0; j < mat.outerSize(); ++j)
    for (typename SparseMatrixType::InnerIterator it(mat, j); it; ++it) {
      ++count;
      internal::PutMatrixElt(it.value(), it.row() + 1, it.col() + 1, out);
    }
  out.close();
  return true;
}

/**
 * \ingroup SparseExtra_Module
 * @brief writes a dense Matrix or vector to a matrixmarket format file
 *
 * @tparam DenseMatrixType to write to file
 * @param mat matrix to write to file
 * @param filename filename to write to
 * @return true if writing succeeded
 */

template <typename DenseType>
bool saveMarketDense(const DenseType& mat, const std::string& filename) {
  using Scalar = typename DenseType::Scalar;
  using RealScalar = typename DenseType::RealScalar;
  std::ofstream out(filename.c_str(), std::ios::out);
  if (!out) return false;

  out.flags(std::ios_base::scientific);
  out.precision(std::numeric_limits<RealScalar>::digits10 + 2);
  EIGEN_IF_CONSTEXPR ((std::is_same<Scalar, std::complex<float> >::value ||
                       std::is_same<Scalar, std::complex<double> >::value)) {
    out << "%%MatrixMarket matrix array complex general\n";
  } else {
    out << "%%MatrixMarket matrix array real general\n";
  }
  out << mat.rows() << " " << mat.cols() << "\n";
  for (Index i = 0; i < mat.cols(); i++) {
    for (Index j = 0; j < mat.rows(); j++) {
      internal::putDenseElt(mat(j, i), out);
    }
  }
  out.close();
  return true;
}

/**
 * \ingroup SparseExtra_Module
 * @brief Same functionality as saveMarketDense, deprecated
 */
template <typename VectorType>
bool saveMarketVector(const VectorType& vec, const std::string& filename) {
  return saveMarketDense(vec, filename);
}

}  // end namespace Eigen

#endif  // EIGEN_SPARSE_MARKET_IO_H

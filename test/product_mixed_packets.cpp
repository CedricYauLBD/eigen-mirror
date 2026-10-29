// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

template <typename Real, int Order>
class MixedProxyMatrix;

namespace Eigen {
namespace internal {
template <typename Real, int Order>
struct traits<MixedProxyMatrix<Real, Order>> : traits<Matrix<Real, 2, 2, Order>> {
  static constexpr unsigned int Flags = traits<Matrix<Real, 2, 2, Order>>::Flags & ~LvalueBit;
};
}  // namespace internal
}  // namespace Eigen

template <typename Real, int Order>
class MixedProxyMatrix : public MatrixBase<MixedProxyMatrix<Real, Order>> {
 public:
  struct Proxy {
    const Real* value;
    operator Real() const { return *value; }
  };
  MixedProxyMatrix() {
    for (Index i = 0; i < 2; ++i)
      for (Index j = 0; j < 2; ++j) {
        const Index index = Order == ColMajor ? i + 2 * j : 2 * i + j;
        values[index] = Real(i + 2 * j + 1);
        proxies[index].value = values + index;
      }
  }
  Index rows() const { return 2; }
  Index cols() const { return 2; }
  const Real* data() const { return values; }
  Index innerStride() const { return 1; }
  Index outerStride() const { return 2; }
  Real values[4];
  Proxy proxies[4];
};

namespace Eigen {
namespace internal {
template <typename Real, int Order>
struct evaluator<MixedProxyMatrix<Real, Order>> : evaluator_base<MixedProxyMatrix<Real, Order>> {
  using Xpr = MixedProxyMatrix<Real, Order>;
  static constexpr int CoeffReadCost = 1;
  static constexpr unsigned int Flags = DirectAccessBit | PacketAccessBit | (Order == RowMajor ? RowMajorBit : 0);
  static constexpr int Alignment = 0;
  const Xpr& matrix;
  explicit evaluator(const Xpr& xpr) : matrix(xpr) {}
  const typename Xpr::Proxy& coeff(Index row, Index col) const {
    return matrix.proxies[Order == ColMajor ? row + 2 * col : 2 * row + col];
  }
  template <int LoadMode, typename Packet>
  Packet packet(Index row, Index col) const {
    return ploadu<Packet>(matrix.data() + (Order == ColMajor ? row + 2 * col : 2 * row + col));
  }
};
}  // namespace internal
}  // namespace Eigen

template <typename Real, int Order>
void mixed_proxy_packets() {
  using Complex = std::complex<Real>;
  const MixedProxyMatrix<Real, Order> matrix;
  STATIC_CHECK((!internal::product_real_packet_access<MixedProxyMatrix<Real, Order>>::value));
  Matrix<Complex, 2, 1> rhs;
  rhs << Complex(1, 2), Complex(3, 4);
  const Matrix<Complex, 2, 1> column = matrix.lazyProduct(rhs);
  const Matrix<Complex, 1, 2> row = rhs.transpose().lazyProduct(matrix);
  VERIFY_IS_EQUAL((matrix * rhs).eval(), column);
  VERIFY_IS_EQUAL((rhs.transpose() * matrix).eval(), row);
  VERIFY_IS_EQUAL(column(0), Complex(10, 14));
  VERIFY_IS_EQUAL(column(1), Complex(14, 20));
  VERIFY_IS_EQUAL(row(0), Complex(7, 10));
  VERIFY_IS_EQUAL(row(1), Complex(15, 22));
}

template <typename Product>
void mixed_infinite_segments(const Product&, internal::false_type) {}

template <typename Product>
void mixed_infinite_segments(const Product& product, internal::true_type) {
  using Scalar = typename Product::Scalar;
  using Packet = typename internal::packet_traits<Scalar>::type;
  constexpr Index packet_size = internal::unpacket_traits<Packet>::size;
  internal::evaluator<Product> evaluator(product);
  for (Index begin = 0; begin <= packet_size; ++begin) {
    for (Index count = 0; count <= packet_size - begin; ++count) {
      if (begin + count > product.size()) continue;
      Scalar values[packet_size];
      internal::pstoreu(values, evaluator.template packetSegment<Unaligned, Packet>(0, 0, begin, count));
      for (Index i = 0; i < packet_size; ++i) {
        if (i >= begin && i < begin + count) {
          VERIFY_IS_EQUAL(values[i].real(), NumTraits<typename NumTraits<Scalar>::Real>::infinity());
          VERIFY_IS_EQUAL(values[i].imag(), NumTraits<typename NumTraits<Scalar>::Real>::infinity());
        }
      }
    }
  }
}

template <typename Real>
EIGEN_DONT_INLINE void mixed_infinite_factor() {
#if !defined(__FAST_MATH__)
  using Complex = std::complex<Real>;
  for (Index size = 1; size <= 9; ++size) {
    for (Index depth : {Index(1), Index(3)}) {
      Matrix<Complex, Dynamic, Dynamic, ColMajor> a(size, depth);
      a.setConstant(Complex(1, 2));
      Matrix<Complex, Dynamic, Dynamic, RowMajor> transposed = a.transpose();
      Matrix<Real, Dynamic, 1> factors(depth);
      for (bool row_major : {false, true}) {
        for (bool expand_real : {false, true}) {
          const Real factor = NumTraits<Real>::infinity();
          factors.setConstant(factor);
          const Matrix<Real, Dynamic, Dynamic, ColMajor> real =
              Matrix<Real, Dynamic, Dynamic, ColMajor>::Constant(size, depth, Real(factor));
          const Matrix<Complex, Dynamic, 1> complex_factors =
              Matrix<Complex, Dynamic, 1>::Constant(depth, Complex(1, 2));
          Matrix<Complex, Dynamic, Dynamic> result;
          if (expand_real && row_major) {
            result = complex_factors.transpose().lazyProduct(real.transpose()).eval();
            mixed_infinite_segments(complex_factors.transpose().lazyProduct(real.transpose()),
                                    internal::bool_constant<internal::packet_traits<Complex>::Vectorizable>());
          } else if (expand_real) {
            result = real.lazyProduct(complex_factors).eval();
            mixed_infinite_segments(real.lazyProduct(complex_factors),
                                    internal::bool_constant<internal::packet_traits<Complex>::Vectorizable>());
          } else if (row_major) {
            result = factors.transpose().lazyProduct(transposed).eval();
            mixed_infinite_segments(factors.transpose().lazyProduct(transposed),
                                    internal::bool_constant<internal::packet_traits<Complex>::Vectorizable>());
          } else {
            result = a.lazyProduct(factors).eval();
            mixed_infinite_segments(a.lazyProduct(factors),
                                    internal::bool_constant<internal::packet_traits<Complex>::Vectorizable>());
          }
          if (size == 3 && depth == 3) {
            auto check_fixed = [](const auto& product) {
              using Product = internal::remove_all_t<decltype(product)>;
              const typename Product::PlainObject fixed = product;
              for (Index i = 0; i < fixed.size(); ++i) {
                VERIFY_IS_EQUAL(fixed(i).real(), NumTraits<Real>::infinity());
                VERIFY_IS_EQUAL(fixed(i).imag(), NumTraits<Real>::infinity());
              }
            };
            if (expand_real && row_major)
              check_fixed(complex_factors.template head<3>().transpose() *
                          real.template topLeftCorner<3, 3>().transpose());
            else if (expand_real)
              check_fixed(real.template topLeftCorner<3, 3>() * complex_factors.template head<3>());
            else if (row_major)
              check_fixed(factors.template head<3>().transpose() * transposed.template topLeftCorner<3, 3>());
            else
              check_fixed(a.template topLeftCorner<3, 3>() * factors.template head<3>());
          }
          for (Index i = 0; i < result.size(); ++i) {
            VERIFY_IS_EQUAL(result(i).real(), NumTraits<Real>::infinity());
            VERIFY_IS_EQUAL(result(i).imag(), NumTraits<Real>::infinity());
          }
        }
      }
    }
  }
#endif
}

template <typename Lhs, typename Rhs>
void check_mixed_packets(const Lhs& lhs, const Rhs& rhs) {
  using Real = typename NumTraits<typename Lhs::Scalar>::Real;
  using Complex = std::complex<Real>;
  using Product = decltype(lhs.lazyProduct(rhs));
  using Result = typename Product::PlainObject;
  Matrix<Complex, Dynamic, Dynamic> reference(lhs.rows(), rhs.cols());
  for (Index j = 0; j < rhs.cols(); ++j) {
    for (Index i = 0; i < lhs.rows(); ++i) {
      Real re = 0, im = 0;
      for (Index k = 0; k < lhs.cols(); ++k) {
        const Complex a(lhs.coeff(i, k)), b(rhs.coeff(k, j));
        re += a.real() * b.real();
        im += NumTraits<typename Lhs::Scalar>::IsComplex ? a.imag() * b.real() : a.real() * b.imag();
      }
      reference(i, j) = Complex(re, im);
    }
  }
  Result result = lhs.lazyProduct(rhs);
  VERIFY_IS_EQUAL(result, reference);
  result.setZero();
  result.noalias() += lhs * rhs;
  VERIFY_IS_EQUAL(result, reference);
  result.noalias() -= lhs * rhs;
  VERIFY(result.isZero(Real(0)));
  VERIFY_IS_EQUAL(lhs.lazyProduct(rhs).sum(), reference.sum());
  Result reversed = lhs.lazyProduct(rhs).reverse();
  VERIFY_IS_EQUAL(reversed, reference.reverse());
  Matrix<Complex, Dynamic, Dynamic, Result::IsRowMajor ? RowMajor : ColMajor> padded(lhs.rows() + 1, rhs.cols() + 1);
  padded.setConstant(Complex(19, 23));
  auto destination = padded.topLeftCorner(lhs.rows(), rhs.cols());
  destination.noalias() = lhs * rhs;
  VERIFY_IS_EQUAL(destination, reference);
  destination.noalias() += lhs * rhs;
  VERIFY_IS_EQUAL(destination, (Real(2) * reference).eval());
  destination.noalias() -= lhs * rhs;
  VERIFY_IS_EQUAL(destination, reference);
  VERIFY((padded.bottomRows(1).array() == Complex(19, 23)).all());
  VERIFY((padded.rightCols(1).array() == Complex(19, 23)).all());
}

template <typename Real, int Size>
void mixed_fixed_packets() {
  using Complex = std::complex<Real>;
  using ColMatrix = Matrix<Complex, Size, Size, ColMajor>;
  using RowMatrix = Matrix<Complex, Size, Size, RowMajor>;
  ColMatrix a;
  Matrix<Real, Size, 1> x;
  for (Index j = 0; j < Size; ++j) {
    x(j) = Real(j + 1) / Real(4);
    for (Index i = 0; i < Size; ++i) a(i, j) = Complex(Real(i + j + 1) / Real(2), Real(i - j) / Real(4));
  }
  using ColProduct = decltype(a.lazyProduct(x));
  STATIC_CHECK((bool(internal::evaluator<ColProduct>::Flags & PacketAccessBit) ==
                bool(internal::evaluator<ColMatrix>::Flags & PacketAccessBit)));
  check_mixed_packets(a, x);
  check_mixed_packets(a.conjugate(), x);
  const RowMatrix b = a;
  const Matrix<Real, 1, Size> row = x.transpose();
  using RowProduct = decltype(row.lazyProduct(b));
  STATIC_CHECK((bool(internal::evaluator<RowProduct>::Flags & PacketAccessBit) ==
                bool(internal::evaluator<RowMatrix>::Flags & PacketAccessBit)));
  check_mixed_packets(row, b);
  check_mixed_packets(row, b.conjugate());

  using RealColMatrix = Matrix<Real, Size, Size, ColMajor>;
  const RealColMatrix real = a.real();
  const Matrix<Complex, Size, 1> vector = a.col(0);
  using Expanded = decltype(real.lazyProduct(vector));
  STATIC_CHECK((bool(internal::evaluator<Expanded>::Flags & PacketAccessBit) ==
                (bool(internal::packet_traits<Complex>::Vectorizable) &&
                 bool(internal::evaluator<RealColMatrix>::Flags & PacketAccessBit))));
  check_mixed_packets(real, vector);
  check_mixed_packets(real, vector.conjugate());
  using RealRowMatrix = Matrix<Real, Size, Size, RowMajor>;
  const RealRowMatrix real_row = real;
  const Matrix<Complex, 1, Size> complex_row = vector.transpose();
  using ExpandedRow = decltype(complex_row.lazyProduct(real_row));
  STATIC_CHECK((bool(internal::evaluator<ExpandedRow>::Flags & PacketAccessBit) ==
                (bool(internal::packet_traits<Complex>::Vectorizable) &&
                 bool(internal::evaluator<RealRowMatrix>::Flags & PacketAccessBit))));
  check_mixed_packets(complex_row, real_row);
  check_mixed_packets(complex_row.conjugate(), real_row);
  check_mixed_packets(a.unaryExpr([](Complex value) { return value; }), x);
}

template <typename Real>
void mixed_dynamic_packets() {
  using Complex = std::complex<Real>;
  for (Index depth : {Index(0), Index(1), Index(3), Index(8)}) {
    Matrix<Complex, Dynamic, Dynamic, ColMajor> a(7, depth);
    Matrix<Real, Dynamic, Dynamic> b(depth, 3);
    for (Index j = 0; j < depth; ++j)
      for (Index i = 0; i < a.rows(); ++i) a(i, j) = Complex(Real(i + 1) / 2, Real(j + 1) / 4);
    for (Index j = 0; j < b.cols(); ++j)
      for (Index i = 0; i < depth; ++i) b(i, j) = Real(i - j + 1) / 4;
    check_mixed_packets(a, b);
    const Matrix<Real, Dynamic, Dynamic, ColMajor> real = a.real();
    const Matrix<Complex, Dynamic, Dynamic> complex = b.template cast<Complex>();
    check_mixed_packets(real, complex);
    check_mixed_packets(complex.transpose(), real.transpose());
    Matrix<Complex, Dynamic, Dynamic, RowMajor> transposed = a.transpose();
    Matrix<Real, 1, Dynamic> row(depth);
    for (Index i = 0; i < depth; ++i) row(i) = Real(i + 1) / 4;
    check_mixed_packets(row, transposed);
  }
}

template <typename Real>
void mixed_special_packets() {
  using Complex = std::complex<Real>;
  Matrix<Complex, 5, 1> a;
  const Real inf = NumTraits<Real>::infinity(), nan = NumTraits<Real>::quiet_NaN();
  a << Complex(inf, 2), Complex(2, inf), Complex(nan, 3), Complex(-Real(0), Real(0)), Complex(4, -Real(0));
  const Matrix<Real, 1, 1> factor = Matrix<Real, 1, 1>::Constant(2);
  const Matrix<Complex, 5, 1> result = a.lazyProduct(factor);
  for (Index i = 0; i < a.size(); ++i) {
    const Complex expected = a(i) * Real(2);
    VERIFY((numext::isnan)(expected.real()) ? (numext::isnan)(result(i).real()) : result(i).real() == expected.real());
    VERIFY((numext::isnan)(expected.imag()) ? (numext::isnan)(result(i).imag()) : result(i).imag() == expected.imag());
    if (expected.real() == 0) VERIFY_IS_EQUAL((std::signbit)(result(i).real()), (std::signbit)(expected.real()));
    if (expected.imag() == 0) VERIFY_IS_EQUAL((std::signbit)(result(i).imag()), (std::signbit)(expected.imag()));
  }
  Matrix<Complex, Dynamic, Dynamic, ColMajor> dynamic = a;
  Matrix<Real, Dynamic, Dynamic> dynamic_factor = factor;
  const Matrix<Complex, 5, 1> dynamic_result = dynamic.lazyProduct(dynamic_factor);
  VERIFY_IS_EQUAL((std::signbit)(dynamic_result(3).real()), (std::signbit)(result(3).real()));
  VERIFY_IS_EQUAL((std::signbit)(dynamic_result(4).imag()), (std::signbit)(result(4).imag()));

  const Matrix<Real, 5, 1> real = Matrix<Real, 5, 1>::Constant(2);
  const Matrix<Complex, 1, 1> complex_factor = Matrix<Complex, 1, 1>::Constant(Complex(-Real(0), -Real(0)));
  const Matrix<Complex, 5, 1> expanded = real.lazyProduct(complex_factor);
  const Matrix<Real, Dynamic, Dynamic, ColMajor> dynamic_real = real;
  const Matrix<Complex, Dynamic, Dynamic> dynamic_complex_factor = complex_factor;
  const Matrix<Complex, Dynamic, Dynamic> dynamic_expanded = dynamic_real.lazyProduct(dynamic_complex_factor);
  for (Index i = 0; i < real.size(); ++i) {
    VERIFY_IS_EQUAL(expanded(i).real(), Real(0));
    VERIFY_IS_EQUAL(expanded(i).imag(), Real(0));
    VERIFY((std::signbit)(expanded(i).real()));
    VERIFY((std::signbit)(expanded(i).imag()));
    VERIFY((std::signbit)(dynamic_expanded(i).real()));
    VERIFY((std::signbit)(dynamic_expanded(i).imag()));
  }
  Matrix<Real, 5, 1> real_special;
  real_special << inf, nan, Real(0), -Real(0), Real(2);
  const Matrix<Complex, 1, 1> factor_special = Matrix<Complex, 1, 1>::Constant(Complex(2, -3));
  const Matrix<Complex, 5, 1> special = real_special.lazyProduct(factor_special);
  for (Index i = 0; i < real_special.size(); ++i) {
    const Complex expected = real_special(i) * factor_special(0);
    VERIFY((numext::isnan)(expected.real()) ? (numext::isnan)(special(i).real())
                                            : special(i).real() == expected.real());
    VERIFY((numext::isnan)(expected.imag()) ? (numext::isnan)(special(i).imag())
                                            : special(i).imag() == expected.imag());
    if (expected.real() == 0) VERIFY_IS_EQUAL((std::signbit)(special(i).real()), (std::signbit)(expected.real()));
    if (expected.imag() == 0) VERIFY_IS_EQUAL((std::signbit)(special(i).imag()), (std::signbit)(expected.imag()));
  }
}

template <typename Real>
void mixed_strided_packets() {
  using Complex = std::complex<Real>;
  using ColMatrix = Matrix<Complex, Dynamic, Dynamic, ColMajor>;
  struct Storage {
    Real padding;
    Complex values[9];
  } storage;
  storage.padding = Real(41);
  Map<Matrix<Complex, 3, 3, ColMajor>, Unaligned> unaligned(storage.values);
  for (Index j = 0; j < 3; ++j)
    for (Index i = 0; i < 3; ++i) unaligned(i, j) = Complex(Real(i + j) / 2, Real(i - j) / 4);
  const Matrix<Real, 3, 1> ones = Matrix<Real, 3, 1>::Ones();
  check_mixed_packets(unaligned, ones);
  check_mixed_packets(ones.transpose(), unaligned.transpose());
  Real real_storage[9];
  Map<Matrix<Real, 3, 3, ColMajor>, Unaligned> real_map(real_storage);
  real_map = unaligned.real();
  const Matrix<Complex, 3, 1> complex_ones = Matrix<Complex, 3, 1>::Constant(Complex(1, 2));
  check_mixed_packets(real_map, complex_ones);
  check_mixed_packets(complex_ones.transpose(), real_map.transpose());
  VERIFY_IS_EQUAL(storage.padding, Real(41));
  for (Index inner : {Index(1), Index(2)}) {
    Matrix<Complex, 64, 1> strided_storage;
    strided_storage.setConstant(Complex(31, 37));
    Map<ColMatrix, Unaligned, Stride<Dynamic, Dynamic>> a(strided_storage.data() + 1, 7, 3,
                                                          Stride<Dynamic, Dynamic>(20, inner));
    for (Index j = 0; j < a.cols(); ++j)
      for (Index i = 0; i < a.rows(); ++i) a(i, j) = Complex(Real(i + j) / 2, Real(i - j) / 4);
    Matrix<Real, 3, 1> x;
    x << Real(1) / 4, Real(1) / 2, Real(3) / 4;
    check_mixed_packets(a, x);
    check_mixed_packets(x.transpose(), a.transpose());
    Matrix<Real, 64, 1> real_backing;
    real_backing.setConstant(Real(43));
    Map<Matrix<Real, Dynamic, Dynamic, ColMajor>, Unaligned, Stride<Dynamic, Dynamic>> real_map(
        real_backing.data() + 1, 7, 3, Stride<Dynamic, Dynamic>(20, inner));
    real_map = a.real();
    using Fallback = decltype(real_map.lazyProduct(complex_ones));
    STATIC_CHECK((!(internal::evaluator<Fallback>::Flags & PacketAccessBit)));
    check_mixed_packets(real_map, complex_ones);
    check_mixed_packets(complex_ones.transpose(), real_map.transpose());
    Map<Matrix<Real, Dynamic, Dynamic, ColMajor>, Unaligned, OuterStride<Dynamic>> unit_map(
        real_backing.data() + 1, 7, 3, OuterStride<Dynamic>(20));
    check_mixed_packets(unit_map, complex_ones);
    check_mixed_packets(complex_ones.transpose(), unit_map.transpose());
  }
}

template <typename Real>
void mixed_roundoff_packets() {
  using Complex = std::complex<Real>;
  const Matrix<Complex, 3, 3, ColMajor> a = Matrix<Complex, 3, 3, ColMajor>::Random();
  const Matrix<Real, 3, 1> x = Matrix<Real, 3, 1>::Random();
  const Matrix<Complex, 3, 1> result = a * x;
  const Matrix<Complex, 3, 3, RowMajor> transposed = a.transpose();
  const Matrix<Complex, 1, 3> row_result = x.transpose() * transposed;
  for (Index i = 0; i < 3; ++i) {
    long double real = 0, imag = 0, magnitude = 0;
    for (Index k = 0; k < 3; ++k) {
      const long double re = static_cast<long double>(a(i, k).real()) * static_cast<long double>(x(k));
      const long double im = static_cast<long double>(a(i, k).imag()) * static_cast<long double>(x(k));
      real += re;
      imag += im;
      magnitude += std::abs(re) + std::abs(im);
    }
    // Three products and two additions per component, allowing fused and unfused evaluation.
    const long double bound = 8 * static_cast<long double>(NumTraits<Real>::epsilon()) * magnitude;
    VERIFY(std::abs(static_cast<long double>(result(i).real()) - real) <= bound);
    VERIFY(std::abs(static_cast<long double>(result(i).imag()) - imag) <= bound);
    VERIFY(std::abs(static_cast<long double>(row_result(i).real()) - real) <= bound);
    VERIFY(std::abs(static_cast<long double>(row_result(i).imag()) - imag) <= bound);
  }
  const Matrix<Real, 3, 3, ColMajor> real_matrix = a.real();
  const Matrix<Complex, 3, 1> complex_vector = a.col(0);
  const Matrix<Complex, 3, 1> expanded = real_matrix * complex_vector;
  const Matrix<Complex, 1, 3> expanded_row = complex_vector.transpose() * real_matrix.transpose();
  for (Index i = 0; i < 3; ++i) {
    long double real = 0, imag = 0, magnitude = 0;
    for (Index k = 0; k < 3; ++k) {
      const long double re =
          static_cast<long double>(real_matrix(i, k)) * static_cast<long double>(complex_vector(k).real());
      const long double im =
          static_cast<long double>(real_matrix(i, k)) * static_cast<long double>(complex_vector(k).imag());
      real += re;
      imag += im;
      magnitude += std::abs(re) + std::abs(im);
    }
    const long double bound = 8 * static_cast<long double>(NumTraits<Real>::epsilon()) * magnitude;
    VERIFY(std::abs(static_cast<long double>(expanded(i).real()) - real) <= bound);
    VERIFY(std::abs(static_cast<long double>(expanded(i).imag()) - imag) <= bound);
    VERIFY(std::abs(static_cast<long double>(expanded_row(i).real()) - real) <= bound);
    VERIFY(std::abs(static_cast<long double>(expanded_row(i).imag()) - imag) <= bound);
  }
}

template <typename Real>
void mixed_packet_tests() {
  mixed_proxy_packets<Real, ColMajor>();
  mixed_proxy_packets<Real, RowMajor>();
  mixed_fixed_packets<Real, 2>();
  mixed_fixed_packets<Real, 3>();
  mixed_fixed_packets<Real, 4>();
  mixed_fixed_packets<Real, 5>();
  mixed_fixed_packets<Real, 7>();
  mixed_dynamic_packets<Real>();
  mixed_special_packets<Real>();
  mixed_strided_packets<Real>();
  mixed_roundoff_packets<Real>();
  mixed_infinite_factor<Real>();
}

EIGEN_DECLARE_TEST(product_mixed_packets) {
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_1(mixed_packet_tests<float>());
    CALL_SUBTEST_2(mixed_packet_tests<double>());
  }
}

// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// A half or bfloat16 GEMM accumulates in float over its whole depth and rounds each result coefficient once (#3158).
// The exact and the bounded checks hold for any kernel that does so, whatever its summation order, and whether it
// rounds the float sums to nearest or to odd.

#define EIGEN_RUNTIME_NO_MALLOC
#include "main.h"
#include "product_test_helpers.h"

namespace {

// Small enough cache sizes split the depth, rows and columns into many blocks, so that rounding the result between
// kc blocks would show.
class ScopedSmallCaches {
 public:
  ScopedSmallCaches() : m_l1(l1CacheSize()), m_l2(l2CacheSize()), m_l3(l3CacheSize()) {
    setCpuCacheSizes(4096, 16384, 65536);
  }
  ~ScopedSmallCaches() { setCpuCacheSizes(m_l1, m_l2, m_l3); }
  ScopedSmallCaches(const ScopedSmallCaches&) = delete;
  ScopedSmallCaches& operator=(const ScopedSmallCaches&) = delete;

 private:
  std::ptrdiff_t m_l1, m_l2, m_l3;
};

template <typename T>
std::uint16_t bits(const T& x) {
  return numext::bit_cast<std::uint16_t>(x);
}

// Fills with integers in [0, 3] from a hash of the position, so that the sums vary from one coefficient to the next.
template <typename MatrixType>
void fill_small_integers(MatrixType& m, int salt) {
  using T = typename MatrixType::Scalar;
  for (Index j = 0; j < m.cols(); ++j)
    for (Index i = 0; i < m.rows(); ++i) {
      // Unsigned, so that the products wrap where Index has 32 bits.
      const std::uint32_t h =
          std::uint32_t(i) * 73856093u ^ std::uint32_t(j) * 19349663u ^ std::uint32_t(salt) * 83492791u;
      m(i, j) = T(float((h >> 7) % 4));
    }
}

// sum_l a(i, l) * b(l, j), exactly: every term and partial sum is an integer below 2^53.
template <typename MatrixA, typename MatrixB>
MatrixXd exact_integer_product(const MatrixA& a, const MatrixB& b) {
  MatrixXd c = MatrixXd::Zero(a.rows(), b.cols());
  for (Index j = 0; j < b.cols(); ++j)
    for (Index l = 0; l < a.cols(); ++l) {
      const double blj = static_cast<double>(b(l, j));
      for (Index i = 0; i < a.rows(); ++i) c(i, j) += static_cast<double>(a(i, l)) * blj;
    }
  return c;
}

// With entries in [0, 3] and alpha in {1, 1/2, -1/2}, every partial sum of C0 + alpha * A * B is a multiple of 1/2
// below 2^23 in magnitude: exact in float under any rounding mode. The result must then be fl_T(exact) bit for bit.
// Accumulating in T instead goes wrong once the partial sums pass 2^11 (half) or 2^8 (bfloat16), and so does rounding
// to T between kc blocks. Odd sums between 2^8 and 2^9 are bfloat16 ties, between 2^11 and 2^12 half ties, which a
// final narrowing other than round-to-nearest-even gets wrong.
template <typename T, int LhsOrder, int RhsOrder, int ResOrder>
void bit_exact_case(Index rows, Index depth, Index cols) {
  using ResMatrix = Matrix<T, Dynamic, Dynamic, ResOrder>;
  Matrix<T, Dynamic, Dynamic, LhsOrder> a(rows, depth);
  Matrix<T, Dynamic, Dynamic, RhsOrder> b(depth, cols);
  ResMatrix c0(rows, cols);
  fill_small_integers(a, 1);
  fill_small_integers(b, 2);
  fill_small_integers(c0, 3);
  const MatrixXd ab = exact_integer_product(a, b);

  const auto verify = [&](const ResMatrix& c, double alpha, bool add) {
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i) {
        const double exact = (add ? static_cast<double>(c0(i, j)) : 0.0) + alpha * ab(i, j);
        const T expected = T(static_cast<float>(exact));
        if (bits(c(i, j)) != bits(expected)) {
          std::cerr << "rows=" << rows << " depth=" << depth << " cols=" << cols << " orders=" << LhsOrder << RhsOrder
                    << ResOrder << " alpha=" << alpha << " add=" << add << ": c(" << i << "," << j
                    << ") = " << static_cast<float>(c(i, j)) << ", expected " << static_cast<float>(expected)
                    << " from exact " << exact << std::endl;
          return false;
        }
      }
    return true;
  };

  // Assignment overwrites C without reading it.
  ResMatrix c = ResMatrix::Constant(rows, cols, NumTraits<T>::quiet_NaN());
  c.noalias() = a * b;
  VERIFY(verify(c, 1.0, false));
  c = a * b;
  VERIFY(verify(c, 1.0, false));
  c = c0;
  c.noalias() += (T(0.5f) * a) * b;
  VERIFY(verify(c, 0.5, true));
  c = c0;
  c.noalias() -= a * (b * T(0.5f));
  VERIFY(verify(c, -0.5, true));
}

template <typename T, int LhsOrder, int RhsOrder, int ResOrder>
void bit_exact_orders() {
  for (Index depth : {1, 2, 3, 15, 64, 129, 255, 256, 257, 511, 1000, 2049, 4097}) {
    bit_exact_case<T, LhsOrder, RhsOrder, ResOrder>(37, depth, 29);
    bit_exact_case<T, LhsOrder, RhsOrder, ResOrder>(3, depth, 70);
  }
  ScopedSmallCaches small_caches;
  for (Index depth : {255, 1000, 4097}) bit_exact_case<T, LhsOrder, RhsOrder, ResOrder>(53, depth, 41);
}

template <typename T>
void bit_exact() {
  bit_exact_orders<T, ColMajor, ColMajor, ColMajor>();
  bit_exact_orders<T, RowMajor, ColMajor, ColMajor>();
  bit_exact_orders<T, ColMajor, RowMajor, ColMajor>();
  bit_exact_orders<T, RowMajor, RowMajor, ColMajor>();
  bit_exact_orders<T, ColMajor, ColMajor, RowMajor>();
  bit_exact_orders<T, RowMajor, ColMajor, RowMajor>();
  bit_exact_orders<T, ColMajor, RowMajor, RowMajor>();
  bit_exact_orders<T, RowMajor, RowMajor, RowMajor>();
}

// Operands that are blocks of larger matrices: the conversion reads them through their outer strides.
template <typename T, int LhsOrder, int RhsOrder>
void strided_operands() {
  const Index rows = 41, depth = internal::random<Index>(1200, 1600), cols = 35;
  Matrix<T, Dynamic, Dynamic, LhsOrder> a_storage(rows + 5, depth + 3);
  Matrix<T, Dynamic, Dynamic, RhsOrder> b_storage(depth + 7, cols + 2);
  fill_small_integers(a_storage, 10);
  fill_small_integers(b_storage, 11);
  const auto a = a_storage.block(3, 1, rows, depth);
  const auto b = b_storage.block(5, 2, depth, cols);
  const MatrixXd ab = exact_integer_product(a, b);
  Matrix<T, Dynamic, Dynamic> c(rows, cols);
  c.noalias() = a * b;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(bits(c(i, j)), bits(T(static_cast<float>(ab(i, j)))));
}

// A destination with inner stride 2: C0 is read and the result written at every other coefficient, and the ones
// between stay unchanged.
template <typename T, int ResOrder>
void strided_destination() {
  const Index rows = 45, depth = 1200, cols = 23;
  Matrix<T, Dynamic, Dynamic> a(rows, depth), b(depth, cols), c0(rows, cols);
  fill_small_integers(a, 4);
  fill_small_integers(b, 5);
  fill_small_integers(c0, 13);
  const MatrixXd ab = exact_integer_product(a, b);
  const Index inner_size = ResOrder == ColMajor ? rows : cols;
  const Index outer_size = ResOrder == ColMajor ? cols : rows;
  const Index outer_stride = 2 * inner_size + 1;
  const T sentinel = T(-7.0f);
  Matrix<T, Dynamic, 1> storage = Matrix<T, Dynamic, 1>::Constant(outer_stride * outer_size, sentinel);
  using Strided = Map<Matrix<T, Dynamic, Dynamic, ResOrder>, 0, Stride<Dynamic, Dynamic>>;
  Strided c(storage.data(), rows, cols, Stride<Dynamic, Dynamic>(outer_stride, 2));
  c = c0;
  c.noalias() += a * b;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i)
      VERIFY_IS_EQUAL(bits(c(i, j)), bits(T(static_cast<float>(static_cast<double>(c0(i, j)) + ab(i, j)))));
  Index untouched = 0;
  for (Index p = 0; p < storage.size(); ++p) {
    const Index inner = p % outer_stride;
    if (inner % 2 == 1 || inner / 2 >= inner_size) {
      VERIFY_IS_EQUAL(bits(storage(p)), bits(sentinel));
      ++untouched;
    }
  }
  VERIFY_IS_EQUAL(untouched, storage.size() - rows * cols);
}

// C = C0 + alpha * A * B against a double reference from the same inputs: one rounding to T, plus at most 3 k + 2 float
// roundings (k multiply-adds, a scale and an add per kc block; Higham & Mary 2019, as in product_test_helpers.h):
//   ||C - C_ref||_F <= eps_T / 2 ||C_ref||_F + denorm_min_T / 2 sqrt(m n)
//                      + lambda sqrt(3 k + 2) eps_f32 || |alpha| |A| |B| + |C0| ||_F.
// eps_f32, not the unit roundoff, admits round-to-odd accumulation. In T, inputs in [0, 1] would stagnate.
template <typename T>
void random_values_case(Index rows, Index depth, Index cols, bool nonnegative, bool update) {
  using MatrixT = Matrix<T, Dynamic, Dynamic>;
  const auto draw = [nonnegative](Index r, Index c) {
    MatrixXf m = MatrixXf::Random(r, c);
    if (nonnegative) m = m.cwiseAbs();
    return MatrixT(m.cast<T>());
  };
  const MatrixT a = draw(rows, depth), b = draw(depth, cols);
  const MatrixT c0 = update ? draw(rows, cols) : MatrixT(MatrixT::Zero(rows, cols));
  const T alpha = update ? T(-0.75f) : T(1.0f);

  MatrixT c = c0;
  c.noalias() += (alpha * a) * b;

  MatrixXd reference = c0.template cast<double>();
  for (Index j = 0; j < cols; ++j)
    for (Index l = 0; l < depth; ++l) {
      const double blj = static_cast<double>(alpha) * static_cast<double>(b(l, j));
      for (Index i = 0; i < rows; ++i) reference(i, j) += static_cast<double>(a(i, l)) * blj;
    }
  const double accumulation =
      5.0 * std::sqrt(double(3 * depth + 2)) * double(NumTraits<float>::epsilon()) *
      abs_product_update_norm(std::abs(static_cast<double>(alpha)), a, b,
                              [&c0](Index i, Index j) { return std::abs(static_cast<double>(c0(i, j))); });
  const double bound = 0.5 * double(NumTraits<T>::epsilon()) * reference.norm() +
                       0.5 * double((std::numeric_limits<T>::denorm_min)()) * std::sqrt(double(rows * cols)) +
                       accumulation;
  VERIFY(verify_within_product_bound(c.template cast<double>(), reference, bound));
}

template <typename T>
void random_values() {
  for (Index depth : {16, 64, 256, 1024, 4097})
    for (bool nonnegative : {true, false})
      for (bool update : {false, true}) {
        random_values_case<T>(64, depth, 64, nonnegative, update);
        random_values_case<T>(internal::random<Index>(2, 70), depth, internal::random<Index>(2, 70), nonnegative,
                              update);
      }
  ScopedSmallCaches small_caches;
  random_values_case<T>(64, 2049, 64, true, true);
}

// Inf and NaN in either operand and in C0 propagate through the float accumulation: Inf * 0 and Inf - Inf give NaN,
// and Inf keeps its sign. The double reference has the same class in any summation order.
template <typename T>
void special_values() {
  using MatrixT = Matrix<T, Dynamic, Dynamic>;
  const Index rows = 24, depth = 40, cols = 24;
  const T inf = NumTraits<T>::infinity(), nan = NumTraits<T>::quiet_NaN();
  MatrixT a = MatrixT::Ones(rows, depth), b = MatrixT::Ones(depth, cols);
  MatrixT c0 = MatrixT::Constant(rows, cols, T(2.0f));
  a(3, 7) = inf;
  a(5, 11) = nan;
  a(6, 1) = inf;
  a(6, 2) = -inf;
  a(8, 4) = inf;
  b(4, 9) = T(0.0f);
  b(12, 15) = -inf;
  b(13, 17) = nan;
  c0(0, 0) = nan;
  c0(1, 1) = inf;
  c0(2, 2) = -inf;
  c0(0, 15) = inf;
  const auto verify = [&](const MatrixT& c, bool add) {
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i) {
        double expected = add ? static_cast<double>(c0(i, j)) : 0.0;
        for (Index l = 0; l < depth; ++l) expected += static_cast<double>(a(i, l)) * static_cast<double>(b(l, j));
        if ((numext::isnan)(expected))
          VERIFY((numext::isnan)(c(i, j)));
        else
          VERIFY_IS_EQUAL(bits(c(i, j)), bits(T(static_cast<float>(expected))));
      }
  };
  MatrixT c(rows, cols);
  c.noalias() = a * b;
  verify(c, false);
  c = c0;
  c.noalias() += a * b;
  verify(c, true);
}

// A zero alpha leaves C unchanged bit for bit, -0 and NaN included, and reads neither operand, so that 0 * NaN cannot
// reach C (the BLAS contract). The zero test sees the factor combined in float.
template <typename T>
void zero_alpha() {
  using MatrixT = Matrix<T, Dynamic, Dynamic>;
  MatrixT a = MatrixT::Ones(30, 30), b = MatrixT::Ones(30, 30), c0(30, 30);
  a(4, 5) = NumTraits<T>::quiet_NaN();
  b(6, 7) = NumTraits<T>::infinity();
  fill_small_integers(c0, 12);
  c0(0, 0) = T(-0.0f);
  c0(1, 1) = NumTraits<T>::quiet_NaN();
  MatrixT c = c0;
  c.noalias() += (T(-0.0f) * a) * b;
  c.noalias() -= a * (T(0.0f) * b);
  for (Index j = 0; j < c.cols(); ++j)
    for (Index i = 0; i < c.rows(); ++i) VERIFY_IS_EQUAL(bits(c(i, j)), bits(c0(i, j)));
}

// C0 + A * B rounds once at the overflow threshold too. The largest half, 65504, plus 15 rounds back to it; plus 16 is
// the tie 65520, which rounds to even: Inf. The largest bfloat16, 255 * 2^120, plus 2^118 rounds back to it; plus
// 2^119 is a tie, which rounds to Inf. Either sign; every term and sum is exact in float.
template <typename T>
void overflow_threshold(Index depth, float a_value, float b_value, bool expect_inf) {
  using MatrixT = Matrix<T, Dynamic, Dynamic>;
  const double highest = static_cast<double>(static_cast<float>(NumTraits<T>::highest()));
  for (double sign : {1.0, -1.0}) {
    const MatrixT a = MatrixT::Constant(20, depth, T(static_cast<float>(sign) * a_value));
    const MatrixT b = MatrixT::Constant(depth, 20, T(b_value));
    const MatrixT c0 = MatrixT::Constant(20, 20, T(static_cast<float>(sign * highest)));
    MatrixT c = c0;
    c.noalias() += a * b;
    const T expected = T(static_cast<float>(sign * (highest + double(depth) * double(a_value) * double(b_value))));
    VERIFY_IS_EQUAL(bool((numext::isinf)(expected)), expect_inf);
    for (Index j = 0; j < c.cols(); ++j)
      for (Index i = 0; i < c.rows(); ++i) VERIFY_IS_EQUAL(bits(c(i, j)), bits(expected));
  }
}

template <typename = void>
void overflow_thresholds() {
  overflow_threshold<half>(15, 1.0f, 1.0f, false);
  overflow_threshold<half>(16, 1.0f, 1.0f, true);
  overflow_threshold<bfloat16>(2, std::ldexp(1.0f, 59), std::ldexp(1.0f, 58), false);
  overflow_threshold<bfloat16>(2, std::ldexp(1.0f, 59), std::ldexp(1.0f, 59), true);
}

// 3 * 2^-130 is a float subnormal: a bfloat16 result keeps it wherever the float GEMM does.
template <typename = void>
void special_values_bfloat16() {
  using MatrixB = Matrix<bfloat16, Dynamic, Dynamic>;
  const Index rows = 24, cols = 24;
  const MatrixB p = MatrixB::Constant(rows, 3, bfloat16(std::ldexp(1.0f, -65)));
  const MatrixB q = MatrixB::Constant(3, cols, bfloat16(std::ldexp(1.0f, -65)));
  MatrixXf float_c(rows, cols);
  float_c.noalias() = p.cast<float>() * q.cast<float>();
  MatrixB c(rows, cols);
  c.noalias() = p * q;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) {
#if defined(EIGEN_USE_OPENBLAS_BFLOAT16) && EIGEN_USE_OPENBLAS_BFLOAT16
      // sbgemm's bfloat16 dot-product instructions may flush subnormal products to zero.
      if (bits(c(i, j)) == 0) continue;
#endif
      VERIFY_IS_EQUAL(bits(c(i, j)), bits(bfloat16(float_c(i, j))));
    }
}

template <typename = void>
void special_values_half() {
  using MatrixH = Matrix<half, Dynamic, Dynamic>;
  const Index rows = 20, depth = 8, cols = 20;
  // Sums past the half range round to Inf.
  MatrixH a = MatrixH::Constant(rows, depth, half(128.0f)), b = MatrixH::Constant(depth, cols, half(128.0f));
  MatrixH c(rows, cols);
  c.noalias() = a * b;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY((numext::isinf)(c(i, j)) && !(numext::signbit)(c(i, j)));

  // Partial sums past the half range that cancel: 255^2 + 255^2 - 255^2 - 255^2 = 0, exactly, in float.
  for (Index l = 0; l < depth; ++l) a.col(l).setConstant(half(l % 4 < 2 ? 255.0f : -255.0f));
  b.setConstant(half(255.0f));
  c.noalias() = a * b;
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(bits(c(i, j)), bits(half(0.0f)));

  // Subnormal half results are normal floats. Each product 2^-13 * 2^-13 = 2^-26 is below the smallest half subnormal,
  // 2^-24, and rounds to zero in half, while the float sums k * 2^-26 do not; k = 2 is the tie 2^-25, which rounds to
  // +0, and k = 3 rounds up to 2^-24.
  a.setConstant(half(std::ldexp(1.0f, -13)));
  b.setConstant(half(std::ldexp(1.0f, -13)));
  for (Index k : {2, 3, 8}) {
    MatrixH d(rows, cols);
    d.noalias() = a.leftCols(k) * b.topRows(k);
    const half expected = half(float(k) * std::ldexp(1.0f, -26));
    for (Index j = 0; j < cols; ++j)
      for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(bits(d(i, j)), bits(expected));
  }
  VERIFY_IS_EQUAL(bits(half(std::ldexp(1.0f, -25))), bits(half(0.0f)));
}

// Scalar factors combine in float: (s A) (t B) with s = t = 2^-14 has st = 2^-28, below the smallest half subnormal.
template <typename T>
void scalar_factors_in_float() {
  using MatrixT = Matrix<T, Dynamic, Dynamic>;
  const MatrixT a = MatrixT::Constant(30, 4, T(256.0f)), b = MatrixT::Constant(4, 30, T(256.0f));
  const T s = T(std::ldexp(1.0f, -14));
  MatrixT c(30, 30);
  c.noalias() = (s * a) * (s * b);
  VERIFY_IS_EQUAL(bits(c(0, 0)), bits(T(std::ldexp(1.0f, -10))));
  VERIFY_IS_EQUAL(bits(c(29, 29)), bits(T(std::ldexp(1.0f, -10))));
}

// A fixed-size product stays off the heap where Eigen allocates on the stack and its f32 workspace fits
// EIGEN_STACK_ALLOCATION_LIMIT.
template <typename T>
void fixed_size_no_malloc() {
  constexpr int kRows = 24, kDepth = 32, kCols = 20;
  Matrix<T, kRows, kDepth> a;
  Matrix<T, kDepth, kCols> b;
  fill_small_integers(a, 6);
  fill_small_integers(b, 7);
  const MatrixXd ab = exact_integer_product(a, b);
  Matrix<T, kRows, kCols> c;
#if defined(EIGEN_ALLOCA) && !defined(EIGEN_NO_ALLOCA)
  // The workspace holds the f32 m x n result, a kc x nc RHS block and a chunk of at most m LHS rows by kc, with
  // kc <= depth and nc <= n. The bound is symmetric in m and n, so it holds for a row-major result too, and each of
  // its terms is a multiple of the 16 floats the driver aligns its buffers to.
  constexpr std::size_t kMaxWorkspaceBytes = sizeof(float) * (kRows * kCols + kDepth * kCols + kRows * kDepth);
  internal::set_is_malloc_allowed(kMaxWorkspaceBytes > EIGEN_STACK_ALLOCATION_LIMIT);
#endif
  c.noalias() = a * b;
  internal::set_is_malloc_allowed(true);
  for (Index j = 0; j < c.cols(); ++j)
    for (Index i = 0; i < c.rows(); ++i) VERIFY_IS_EQUAL(bits(c(i, j)), bits(T(static_cast<float>(ab(i, j)))));
}

// The driver writes the unrounded float sums into a float result: 3 * 3 * 4097 exceeds 2^11 and 2^8.
template <typename T>
void float_result() {
  const Index rows = 33, depth = 4097, cols = 27;
  Matrix<T, Dynamic, Dynamic, ColMajor> a(rows, depth), b(depth, cols);
  fill_small_integers(a, 8);
  fill_small_integers(b, 9);
  const MatrixXd ab = exact_integer_product(a, b);
  Matrix<float, Dynamic, Dynamic, ColMajor> c = Matrix<float, Dynamic, Dynamic, ColMajor>::Constant(rows, cols, 0.25f);
  using Gemm =
      internal::general_matrix_matrix_product<Index, T, ColMajor, false, T, ColMajor, false, ColMajor, 1, float>;
  internal::gemm_blocking_space<ColMajor, float, float, Dynamic, Dynamic, Dynamic> blocking(rows, cols, depth, 1, true);
  Gemm::run(rows, cols, depth, a.data(), a.outerStride(), b.data(), b.outerStride(), c.data(), 1, c.outerStride(), 2.0f,
            blocking);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(c(i, j), static_cast<float>(0.25 + 2.0 * ab(i, j)));
  // run_overwrite writes the sums without reading the result.
  c.setConstant(std::numeric_limits<float>::quiet_NaN());
  Gemm::run_overwrite(rows, cols, depth, a.data(), a.outerStride(), b.data(), b.outerStride(), c.data(), 1,
                      c.outerStride(), 2.0f, blocking);
  for (Index j = 0; j < cols; ++j)
    for (Index i = 0; i < rows; ++i) VERIFY_IS_EQUAL(c(i, j), static_cast<float>(2.0 * ab(i, j)));
}

template <typename = void>
void accumulator_selection() {
  using internal::product_accumulator;
  using internal::product_widens;
  STATIC_CHECK((std::is_same<product_accumulator<half, half>::type, float>::value));
  STATIC_CHECK((std::is_same<product_accumulator<bfloat16, bfloat16>::type, float>::value));
  STATIC_CHECK((product_widens<half, half>::value));
  STATIC_CHECK((product_widens<bfloat16, bfloat16>::value));
  STATIC_CHECK((!product_widens<float, float>::value));
  STATIC_CHECK((!product_widens<double, double>::value));
  STATIC_CHECK((!product_widens<std::complex<float>, std::complex<float>>::value));
  STATIC_CHECK((!product_widens<std::complex<float>, float>::value));
  STATIC_CHECK((!product_widens<int, int>::value));
  STATIC_CHECK((!product_widens<std::int8_t, std::int8_t>::value));
  STATIC_CHECK((std::is_same<product_accumulator<float, float>::type, float>::value));
  STATIC_CHECK((std::is_same<product_accumulator<std::complex<float>, float>::type, std::complex<float>>::value));
  // The widened products run the float GEMM's packers and kernel.
  using HalfGemm =
      internal::general_matrix_matrix_product<Index, half, ColMajor, false, half, RowMajor, false, RowMajor, 1>;
  STATIC_CHECK((std::is_same<HalfGemm::Traits::LhsScalar, float>::value));
  STATIC_CHECK((std::is_same<HalfGemm::Traits::RhsScalar, float>::value));
  using FloatGemm =
      internal::general_matrix_matrix_product<Index, float, ColMajor, false, float, ColMajor, false, ColMajor, 1>;
  STATIC_CHECK((std::is_same<FloatGemm::Traits::LhsScalar, float>::value));
}

}  // namespace

EIGEN_DECLARE_TEST(product_half_bfloat16) {
  CALL_SUBTEST_1(accumulator_selection<>());
  CALL_SUBTEST_1(bit_exact<half>());
  CALL_SUBTEST_2(bit_exact<bfloat16>());
  CALL_SUBTEST_3((strided_destination<half, ColMajor>()));
  CALL_SUBTEST_3((strided_destination<half, RowMajor>()));
  CALL_SUBTEST_3((strided_destination<bfloat16, ColMajor>()));
  CALL_SUBTEST_3((strided_destination<bfloat16, RowMajor>()));
  CALL_SUBTEST_3((strided_operands<half, ColMajor, RowMajor>()));
  CALL_SUBTEST_3((strided_operands<half, RowMajor, ColMajor>()));
  CALL_SUBTEST_3((strided_operands<bfloat16, ColMajor, ColMajor>()));
  CALL_SUBTEST_3((strided_operands<bfloat16, RowMajor, RowMajor>()));
  CALL_SUBTEST_3(special_values<half>());
  CALL_SUBTEST_3(special_values<bfloat16>());
  CALL_SUBTEST_3(zero_alpha<half>());
  CALL_SUBTEST_3(zero_alpha<bfloat16>());
  CALL_SUBTEST_3(overflow_thresholds<>());
  CALL_SUBTEST_3(special_values_half<>());
  CALL_SUBTEST_3(special_values_bfloat16<>());
  CALL_SUBTEST_3(scalar_factors_in_float<half>());
  CALL_SUBTEST_3(scalar_factors_in_float<bfloat16>());
  CALL_SUBTEST_3(fixed_size_no_malloc<half>());
  CALL_SUBTEST_3(fixed_size_no_malloc<bfloat16>());
  CALL_SUBTEST_3(float_result<half>());
  CALL_SUBTEST_3(float_result<bfloat16>());
  for (int i = 0; i < g_repeat; ++i) {
    CALL_SUBTEST_4(random_values<half>());
    CALL_SUBTEST_5(random_values<bfloat16>());
  }
}

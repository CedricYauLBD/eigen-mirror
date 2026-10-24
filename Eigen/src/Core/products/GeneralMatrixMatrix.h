// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_GENERAL_MATRIX_MATRIX_H
#define EIGEN_GENERAL_MATRIX_MATRIX_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename LhsScalar_, typename RhsScalar_>
class level3_blocking;

// Coalesce two heap packing buffers so glibc's dynamic thresholds learn the whole workspace size (#3201).
// Leave stack allocations and caller-provided buffers on their existing paths.
template <typename LhsScalar, typename RhsScalar>
class gemm_packing_space {
  LhsScalar* m_blockA;
  RhsScalar* m_blockB;
  aligned_stack_memory_handler<unsigned char> m_storage;
  aligned_stack_memory_handler<LhsScalar> m_lhs_lifetime;
  aligned_stack_memory_handler<RhsScalar> m_rhs_lifetime;

  static EIGEN_DONT_INLINE unsigned char* allocateCombined(std::size_t sizeA, std::size_t sizeB, LhsScalar*& blockA,
                                                           RhsScalar*& blockB) {
    check_size_for_overflow<LhsScalar>(sizeA);
    check_size_for_overflow<RhsScalar>(sizeB);
    // Not numext::maxi: under SYCL_DEVICE_ONLY its cl_ulong (std::size_t) specialization is not constexpr.
    constexpr std::size_t alignment = (std::max)(std::size_t{EIGEN_STACK_ALIGN_BYTES}, alignof(RhsScalar));
    const std::size_t max_bytes = (std::numeric_limits<std::ptrdiff_t>::max)();
    const std::size_t bytesA = sizeA * sizeof(LhsScalar);
    const std::size_t bytesB = sizeB * sizeof(RhsScalar);
    if (bytesA > max_bytes - (alignment - 1)) throw_std_bad_alloc();
    const std::size_t offsetB = first_multiple(bytesA, alignment);
    if (bytesB > max_bytes - offsetB) throw_std_bad_alloc();
    void* storage = scratch_malloc(offsetB + bytesB);
    void* storageB = static_cast<unsigned char*>(storage) + offsetB;
    blockA = static_cast<LhsScalar*>(storage);
    blockB = static_cast<RhsScalar*>(storageB);
    return static_cast<unsigned char*>(storage);
  }

  static unsigned char* allocate(std::size_t sizeA, std::size_t sizeB, LhsScalar*& blockA, RhsScalar*& blockB) {
    if (blockA != nullptr || blockB != nullptr || sizeA == 0 || sizeB == 0) return nullptr;
#if defined(EIGEN_ALLOCA) && !defined(EIGEN_NO_ALLOCA)
    if (sizeA <= EIGEN_STACK_ALLOCATION_LIMIT / sizeof(LhsScalar) ||
        sizeB <= EIGEN_STACK_ALLOCATION_LIMIT / sizeof(RhsScalar))
      return nullptr;
#endif
    // scratch_malloc guarantees only EIGEN_STACK_ALIGN_BYTES, or malloc's alignment when that is 0.
    constexpr std::size_t base_alignment =
        EIGEN_STACK_ALIGN_BYTES > 0 ? EIGEN_STACK_ALIGN_BYTES : alignof(std::max_align_t);
    if ((std::max)(alignof(LhsScalar), alignof(RhsScalar)) > base_alignment) return nullptr;
    return allocateCombined(sizeA, sizeB, blockA, blockB);
  }

 public:
  gemm_packing_space(std::size_t sizeA, std::size_t sizeB, LhsScalar* blockA, RhsScalar* blockB)
      : m_blockA(blockA),
        m_blockB(blockB),
        m_storage(allocate(sizeA, sizeB, m_blockA, m_blockB), 0, true),
        m_lhs_lifetime(blockA == nullptr ? m_blockA : nullptr, sizeA, false),
        m_rhs_lifetime(blockB == nullptr ? m_blockB : nullptr, sizeB, false) {}

  LhsScalar* blockA() const { return m_blockA; }
  RhsScalar* blockB() const { return m_blockB; }
};

// LHS-first loop order: mc -> kc -> nc. This is Eigen's usual sequential
// schedule, with a fast path that can pack RHS once for tall skinny blocking.
struct gemm_pack_lhs_first_loop_policy {
  template <typename Index, typename LhsScalar, typename RhsScalar, typename ResScalar, typename LhsMapper,
            typename RhsMapper, typename ResMapper, typename PackLhs, typename PackRhs, typename Gebp>
  static EIGEN_STRONG_INLINE void run(Index rows, Index cols, Index depth, Index kc, Index mc, Index nc,
                                      const LhsMapper& lhs, const RhsMapper& rhs, ResMapper& res, PackLhs& pack_lhs,
                                      PackRhs& pack_rhs, Gebp& gebp, LhsScalar* blockA, RhsScalar* blockB,
                                      ResScalar alpha) {
    const bool pack_rhs_once = mc != rows && kc == depth && nc == cols;

    // For each horizontal panel of the rhs, and corresponding panel of the lhs...
    for (Index i2 = 0; i2 < rows; i2 += mc) {
      const Index actual_mc = (std::min)(i2 + mc, rows) - i2;

      for (Index k2 = 0; k2 < depth; k2 += kc) {
        const Index actual_kc = (std::min)(k2 + kc, depth) - k2;

        // OK, here we have selected one horizontal panel of rhs and one vertical panel of lhs.
        // => Pack lhs's panel into a sequential chunk of memory (L2/L3 caching)
        // Note that this panel will be read as many times as the number of blocks in the rhs's
        // horizontal panel which is, in practice, a very low number.
        pack_lhs(blockA, lhs.getSubMapper(i2, k2), actual_kc, actual_mc);

        // For each kc x nc block of the rhs's horizontal panel...
        for (Index j2 = 0; j2 < cols; j2 += nc) {
          const Index actual_nc = (std::min)(j2 + nc, cols) - j2;

          // We pack the rhs's block into a sequential chunk of memory (L2 caching)
          // Note that this block will be read a very high number of times, which is equal to the number of
          // micro horizontal panel of the large rhs's panel (e.g., rows/12 times).
          if ((!pack_rhs_once) || i2 == 0) pack_rhs(blockB, rhs.getSubMapper(k2, j2), actual_kc, actual_nc);

          // Everything is packed, we can now call the panel * block kernel:
          gebp(res.getSubMapper(i2, j2), blockA, blockB, actual_mc, actual_kc, actual_nc, alpha);
        }
      }
    }
  }
};

#ifdef EIGEN_VECTORIZE_SME
// Defined in arch/SME/GeneralBlockPanelKernel.h, included after this header:
// whether the SME kernel can read this ColMajor LHS block straight from its
// source instead of a packed panel.
template <typename Scalar, typename Index>
bool sme_direct_lhs_ok(Index lhsStride, Index rows, Index depth, Index cols);
// True for the unit-stride ColMajor mapper the GEMM driver hands the packers, when it reads the kernel's own scalar:
// a source the packers convert has no LHS the kernel could read in place.
template <typename Mapper, typename KernelScalar>
struct sme_direct_lhs_mapper : std::false_type {};
template <typename Scalar, typename Index>
struct sme_direct_lhs_mapper<const_blas_data_mapper<Scalar, Index, ColMajor>, Scalar>
    : bool_constant<!NumTraits<Scalar>::IsComplex> {};
// Runs the block on the in-place LHS when the kernel can take it; the false
// overload keeps the call out of every other instantiation.
template <typename Gebp, typename ResMapper, typename LhsMapper, typename Scalar, typename ResScalar, typename Index>
EIGEN_ALWAYS_INLINE bool sme_run_direct_lhs(std::true_type, Gebp& gebp, const ResMapper& res, const LhsMapper& lhs,
                                            Index i2, Index k2, const Scalar* blockB, Index mc, Index kc, Index nc,
                                            ResScalar alpha) {
  if (!sme_direct_lhs_ok<Scalar>(lhs.stride(), mc, kc, nc)) return false;
  gebp.run_direct_lhs(res, &lhs(i2, k2), lhs.stride(), blockB, mc, kc, nc, alpha);
  return true;
}
template <typename Scalar, int LhsOrder, int RhsOrder, typename Index>
bool sme_tiny_gemm(Index rows, Index cols, Index depth, const Scalar* lhs, Index lhsStride, const Scalar* rhs,
                   Index rhsStride, Scalar* res, Index resIncr, Index resStride, Scalar alpha);
// Where the tiny-result kernel beats both the coeff-based product and the packed paths, as tuned on Apple M4: short
// depths keep the coeff-based product, and a full 2 * ps x 8 result over a long depth the SME kernel. The NEON
// small-block switches pin the paths they select, so they turn it off.
template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE bool sme_tiny_gemm_wins(Index rows, Index cols, Index depth) {
#if defined(EIGEN_SME_NO_NEON_SMALL_BLOCKS) || defined(EIGEN_SME_FORCE_NEON_SMALL_BLOCKS) || defined(EIGEN_USE_BLAS)
  EIGEN_UNUSED_VARIABLE(rows);
  EIGEN_UNUSED_VARIABLE(cols);
  EIGEN_UNUSED_VARIABLE(depth);
  return false;
#else
  const Index ps = Index(16 / sizeof(Scalar));  // scalars per NEON vector
  if (rows < 2 || cols < 2 || rows > 2 * ps || cols > 8) return false;
  if (depth < 16 && (depth < 6 || rows * cols < 24)) return false;
  return !(cols == 8 && rows > ps && depth >= 2048);
#endif
}
// Real float and double on the SME kernel take the tiny-result kernel; the false overload keeps it out of every
// other pair, including double without FEAT_SME_F64F64.
template <typename LhsScalar, typename RhsScalar>
struct sme_tiny_gemm_pair
    : bool_constant<std::is_same<LhsScalar, RhsScalar>::value &&
                    (std::is_same<LhsScalar, float>::value || std::is_same<LhsScalar, double>::value) &&
                    sme_has_gebp_kernel<LhsScalar, RhsScalar>::value> {};
template <int LhsOrder, int RhsOrder, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE bool sme_run_tiny_gemm(std::true_type, Index rows, Index cols, Index depth, const Scalar* lhs,
                                           Index lhsStride, const Scalar* rhs, Index rhsStride, Scalar* res,
                                           Index resIncr, Index resStride, Scalar alpha) {
  return sme_tiny_gemm<Scalar, LhsOrder, RhsOrder>(rows, cols, depth, lhs, lhsStride, rhs, rhsStride, res, resIncr,
                                                   resStride, alpha);
}
template <int LhsOrder, int RhsOrder, typename LhsScalar, typename RhsScalar, typename ResScalar, typename Index>
EIGEN_ALWAYS_INLINE bool sme_run_tiny_gemm(std::false_type, Index, Index, Index, const LhsScalar*, Index,
                                           const RhsScalar*, Index, ResScalar*, Index, Index, ResScalar) {
  return false;
}
template <typename Gebp, typename ResMapper, typename LhsMapper, typename Scalar, typename ResScalar, typename Index>
EIGEN_ALWAYS_INLINE bool sme_run_direct_lhs(std::false_type, Gebp&, const ResMapper&, const LhsMapper&, Index, Index,
                                            const Scalar*, Index, Index, Index, ResScalar) {
  return false;
}
template <typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE bool sme_small_gemm_wins(Index rows, Index cols, Index depth);
template <typename Scalar, int LhsOrder, int RhsOrder, bool ConjLhs, bool ConjRhs, int ResInnerStride, typename Index>
EIGEN_DONT_INLINE void sme_small_gemm(Index rows, Index cols, Index depth, const Scalar* lhs, Index lhsStride,
                                      const Scalar* rhs, Index rhsStride, Scalar* res, Index resIncr, Index resStride,
                                      Scalar alpha, Scalar* blockA, Scalar* blockB);
// The test stays inline: a call into the NEON path's frame dirties stack lines the ZA path then packs into.
template <int LhsOrder, int RhsOrder, bool ConjLhs, bool ConjRhs, int ResInnerStride, typename Scalar, typename Index,
          typename Blocking>
EIGEN_ALWAYS_INLINE bool sme_run_small_gemm(std::true_type, Index rows, Index cols, Index depth, const Scalar* lhs,
                                            Index lhsStride, const Scalar* rhs, Index rhsStride, Scalar* res,
                                            Index resIncr, Index resStride, Scalar alpha, Blocking& blocking) {
  if (!sme_small_gemm_wins<Scalar>(rows, cols, depth)) return false;
  const bool fits = depth <= blocking.kc() && rows <= blocking.mc() && cols <= blocking.nc();
  sme_small_gemm<Scalar, LhsOrder, RhsOrder, ConjLhs, ConjRhs, ResInnerStride>(
      rows, cols, depth, lhs, lhsStride, rhs, rhsStride, res, resIncr, resStride, alpha,
      fits ? blocking.blockA() : nullptr, fits ? blocking.blockB() : nullptr);
  return true;
}
template <int LhsOrder, int RhsOrder, bool ConjLhs, bool ConjRhs, int ResInnerStride, typename LhsScalar,
          typename RhsScalar, typename ResScalar, typename Index, typename Blocking>
EIGEN_ALWAYS_INLINE bool sme_run_small_gemm(std::false_type, Index, Index, Index, const LhsScalar*, Index,
                                            const RhsScalar*, Index, ResScalar*, Index, Index, ResScalar, Blocking&) {
  return false;
}
#endif

// RHS-first loop order: nc -> kc -> mc. Used by SME to stream ColMajor result
// stores through adjacent row panels; a ColMajor LHS whose MR-row slices are
// already contiguous per depth step is read from its source (no LHS packing).
struct gemm_pack_rhs_first_loop_policy {
  template <typename Index, typename LhsScalar, typename RhsScalar, typename ResScalar, typename LhsMapper,
            typename RhsMapper, typename ResMapper, typename PackLhs, typename PackRhs, typename Gebp>
  static EIGEN_STRONG_INLINE void run(Index rows, Index cols, Index depth, Index kc, Index mc, Index nc,
                                      const LhsMapper& lhs, const RhsMapper& rhs, ResMapper& res, PackLhs& pack_lhs,
                                      PackRhs& pack_rhs, Gebp& gebp, LhsScalar* blockA, RhsScalar* blockB,
                                      ResScalar alpha) {
    // Mirror of pack_rhs_once: reuse one full LHS panel across column blocks. The in-place read is decided per column
    // block, so a block that falls back to packing packs unless an earlier one already did.
    const bool pack_lhs_once = nc != cols && kc == depth && mc == rows;
    bool lhs_packed = false;

    for (Index j2 = 0; j2 < cols; j2 += nc) {
      const Index actual_nc = (std::min)(j2 + nc, cols) - j2;

      for (Index k2 = 0; k2 < depth; k2 += kc) {
        const Index actual_kc = (std::min)(k2 + kc, depth) - k2;

        // Pack one nc-strip of RHS and reuse it across all row panels.
        pack_rhs(blockB, rhs.getSubMapper(k2, j2), actual_kc, actual_nc);

        for (Index i2 = 0; i2 < rows; i2 += mc) {
          const Index actual_mc = (std::min)(i2 + mc, rows) - i2;
#ifdef EIGEN_VECTORIZE_SME
          if (sme_run_direct_lhs(bool_constant<sme_direct_lhs_mapper<LhsMapper, LhsScalar>::value>(), gebp,
                                 res.getSubMapper(i2, j2), lhs, i2, k2, blockB, actual_mc, actual_kc, actual_nc, alpha))
            continue;
#endif
          if (!pack_lhs_once || !lhs_packed) {
            pack_lhs(blockA, lhs.getSubMapper(i2, k2), actual_kc, actual_mc);
            lhs_packed = true;
          }
          gebp(res.getSubMapper(i2, j2), blockA, blockB, actual_mc, actual_kc, actual_nc, alpha);
        }
      }
    }
  }
};

// Goto's blocked GEMM on packed panels of the kernel's scalars: the cache blocking, the packed LHS that a parallel
// product's threads share, and the sequential loop order. The pack functors read from the source mappers, whose
// scalars need not be the packed ones.
template <typename Index, typename LhsScalar, typename RhsScalar, typename ResScalar, typename LhsMapper,
          typename RhsMapper, typename ResMapper, typename PackLhs, typename PackRhs, typename Gebp>
EIGEN_STRONG_INLINE void gemm_blocked_product(Index rows, Index cols, Index depth, const LhsMapper& lhs,
                                              const RhsMapper& rhs, ResMapper& res, PackLhs& pack_lhs,
                                              PackRhs& pack_rhs, Gebp& gebp, ResScalar alpha,
                                              level3_blocking<LhsScalar, RhsScalar>& blocking,
                                              GemmParallelInfo<Index>* info) {
  Index kc = blocking.kc();                    // cache block size along the K direction
  Index mc = (std::min)(rows, blocking.mc());  // cache block size along the M direction
  Index nc = (std::min)(cols, blocking.nc());  // cache block size along the N direction

#if !defined(EIGEN_USE_BLAS) && (defined(EIGEN_HAS_OPENMP) || defined(EIGEN_GEMM_THREADPOOL))
  if (info) {
    // this is the parallel version!
    int tid = info->logical_thread_id;
    int threads = info->num_threads;

    LhsScalar* blockA = blocking.blockA();
    eigen_internal_assert(blockA != 0);

    std::size_t sizeB = kc * nc;
    ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, sizeB, 0);

    // For each horizontal panel of the rhs, and corresponding vertical panel of the lhs...
    for (Index k = 0; k < depth; k += kc) {
      const Index actual_kc = (std::min)(k + kc, depth) - k;  // => rows of B', and cols of the A'

      // In order to reduce the chance that a thread has to wait for the other,
      // let's start by packing B'.
      pack_rhs(blockB, rhs.getSubMapper(k, 0), actual_kc, nc);

      // Pack A_k to A' in a parallel fashion:
      // each thread packs the sub block A_k,i to A'_i where i is the thread id.

      // However, before copying to A'_i, we have to make sure that no other thread is still using it,
      // i.e., we test that info->task_info[tid].users equals 0.
      // Then, we set info->task_info[tid].users to the number of threads to mark that all other threads are going to
      // use it.
      while (info->task_info[tid].users != 0) {
        std::this_thread::yield();
      }
      info->task_info[tid].users = threads;

      pack_lhs(blockA + info->task_info[tid].lhs_start * actual_kc, lhs.getSubMapper(info->task_info[tid].lhs_start, k),
               actual_kc, info->task_info[tid].lhs_length);

      // Notify the other threads that the part A'_i is ready to go.
      info->task_info[tid].sync = k;

      // Computes C_i += A' * B' per A'_i
      for (int shift = 0; shift < threads; ++shift) {
        int i = (tid + shift) % threads;

        // At this point we have to make sure that A'_i has been updated by the thread i,
        // we use testAndSetOrdered to mimic a volatile access.
        // However, no need to wait for the B' part which has been updated by the current thread!
        if (shift > 0) {
          while (info->task_info[i].sync != k) {
            std::this_thread::yield();
          }
        }

        gebp(res.getSubMapper(info->task_info[i].lhs_start, 0), blockA + info->task_info[i].lhs_start * actual_kc,
             blockB, info->task_info[i].lhs_length, actual_kc, nc, alpha);
      }

      // Then keep going as usual with the remaining B'
      for (Index j = nc; j < cols; j += nc) {
        const Index actual_nc = (std::min)(j + nc, cols) - j;

        // pack B_k,j to B'
        pack_rhs(blockB, rhs.getSubMapper(k, j), actual_kc, actual_nc);

        // C_j += A' * B'
        gebp(res.getSubMapper(0, j), blockA, blockB, rows, actual_kc, actual_nc, alpha);
      }

      // Release all the sub blocks A'_i of A' for the current thread,
      // i.e., we simply decrement the number of users by 1
      for (Index i = 0; i < threads; ++i) info->task_info[i].users -= 1;
    }
  } else
#endif  // !defined(EIGEN_USE_BLAS) && (defined(EIGEN_HAS_OPENMP) || defined(EIGEN_GEMM_THREADPOOL))
  {
    EIGEN_UNUSED_VARIABLE(info);

    // this is the sequential version!
    std::size_t sizeA = kc * mc;
    std::size_t sizeB = kc * nc;

    gemm_packing_space<LhsScalar, RhsScalar> packing(sizeA, sizeB, blocking.blockA(), blocking.blockB());
    ei_declare_aligned_stack_constructed_variable(LhsScalar, blockA, sizeA, packing.blockA());
    ei_declare_aligned_stack_constructed_variable(RhsScalar, blockB, sizeB, packing.blockB());

    // The SME kernel uses RHS-first order so consecutive gebp calls stream
    // through adjacent row panels of a ColMajor result. Other kernels --
    // including the scalar pairs SME does not specialize -- keep Eigen's
    // default LHS-first order.
#ifdef EIGEN_VECTORIZE_SME
    using SequentialGemmLoop = std::conditional_t<sme_has_gebp_kernel<LhsScalar, RhsScalar>::value,
                                                  gemm_pack_rhs_first_loop_policy, gemm_pack_lhs_first_loop_policy>;
#else
    using SequentialGemmLoop = gemm_pack_lhs_first_loop_policy;
#endif

    SequentialGemmLoop::run(rows, cols, depth, kc, mc, nc, lhs, rhs, res, pack_lhs, pack_rhs, gebp, blockA, blockB,
                            alpha);
  }
}

// A widened product packs its operands in its accumulator, so that the accumulator's packers and kernel serve it.
template <typename Scalar, typename LhsScalar, typename RhsScalar>
using gemm_packed_scalar = std::conditional_t<product_widens<LhsScalar, RhsScalar>::value,
                                              typename product_accumulator<LhsScalar, RhsScalar>::type, Scalar>;

// Packs a block of SourceScalar coefficients as Pack packs the same block converted to PackedScalar, converting one
// chunk at a time into a contiguous buffer. Pack places the panel that starts at row (LHS) or column (RHS) r of the
// block at r * depth, so chunks whose extents are multiples of the widest panel pack to the layout of the whole block.
// The parallel GEMM relies on the same property for the mr-aligned parts of the packed LHS that its threads pack.
template <typename Index, typename SourceScalar, typename PackedScalar, int StorageOrder, bool IsLhs, typename Pack>
class gemm_converting_pack {
 public:
  gemm_converting_pack(PackedScalar* buffer, Index chunk) : m_buffer(buffer), m_chunk(chunk) {}

  template <typename SourceMapper>
  void operator()(PackedScalar* block, const SourceMapper& source, Index depth, Index extent) const {
    using SourceMatrix = Matrix<SourceScalar, Dynamic, Dynamic, StorageOrder>;
    using PackedMatrix = Matrix<PackedScalar, Dynamic, Dynamic, StorageOrder>;
    using PackedMapper = const_blas_data_mapper<PackedScalar, Index, StorageOrder>;
    for (Index start = 0; start < extent; start += m_chunk) {
      const Index size = numext::mini(m_chunk, extent - start);
      const Index rows = IsLhs ? size : depth;
      const Index cols = IsLhs ? depth : size;
      const auto chunk = source.getSubMapper(IsLhs ? start : 0, IsLhs ? 0 : start);
      Map<PackedMatrix>(m_buffer, rows, cols) =
          Map<const SourceMatrix, 0, OuterStride<>>(chunk.data(), rows, cols, OuterStride<>(chunk.stride()))
              .template cast<PackedScalar>();
      Pack()(block + start * depth, PackedMapper(m_buffer, StorageOrder == ColMajor ? rows : cols), depth, size);
    }
  }

 private:
  PackedScalar* m_buffer;
  Index m_chunk;
};

/* Specialization for a row-major destination matrix => simple transposition of the product */
template <typename Index, typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs, typename RhsScalar,
          int RhsStorageOrder, bool ConjugateRhs, int ResInnerStride, typename ResScalar>
struct general_matrix_matrix_product<Index, LhsScalar, LhsStorageOrder, ConjugateLhs, RhsScalar, RhsStorageOrder,
                                     ConjugateRhs, RowMajor, ResInnerStride, ResScalar> {
  using AccScalar = typename product_accumulator<LhsScalar, RhsScalar>::type;
  using PackedLhsScalar = gemm_packed_scalar<LhsScalar, LhsScalar, RhsScalar>;
  using PackedRhsScalar = gemm_packed_scalar<RhsScalar, LhsScalar, RhsScalar>;
  using Traits = gebp_traits<PackedRhsScalar, PackedLhsScalar>;

  static EIGEN_STRONG_INLINE void run(Index rows, Index cols, Index depth, const LhsScalar* lhs, Index lhsStride,
                                      const RhsScalar* rhs, Index rhsStride, ResScalar* res, Index resIncr,
                                      Index resStride, AccScalar alpha,
                                      level3_blocking<PackedRhsScalar, PackedLhsScalar>& blocking,
                                      GemmParallelInfo<Index>* info = 0) {
    // transpose the product such that the result is column major
    general_matrix_matrix_product<Index, RhsScalar, RhsStorageOrder == RowMajor ? ColMajor : RowMajor, ConjugateRhs,
                                  LhsScalar, LhsStorageOrder == RowMajor ? ColMajor : RowMajor, ConjugateLhs, ColMajor,
                                  ResInnerStride, ResScalar>::run(cols, rows, depth, rhs, rhsStride, lhs, lhsStride,
                                                                  res, resIncr, resStride, alpha, blocking, info);
  }
};

/*  Specialization for a col-major destination matrix
 *    => Blocking algorithm following Goto's paper */
template <typename Index, typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs, typename RhsScalar,
          int RhsStorageOrder, bool ConjugateRhs, int ResInnerStride, typename ResScalar>
struct general_matrix_matrix_product<Index, LhsScalar, LhsStorageOrder, ConjugateLhs, RhsScalar, RhsStorageOrder,
                                     ConjugateRhs, ColMajor, ResInnerStride, ResScalar> {
  using AccScalar = typename product_accumulator<LhsScalar, RhsScalar>::type;
  using PackedLhsScalar = gemm_packed_scalar<LhsScalar, LhsScalar, RhsScalar>;
  using PackedRhsScalar = gemm_packed_scalar<RhsScalar, LhsScalar, RhsScalar>;
  using Traits = gebp_traits<PackedLhsScalar, PackedRhsScalar>;
  static_assert(std::is_same<ResScalar, typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType>::value ||
                    (product_widens<LhsScalar, RhsScalar>::value && std::is_same<ResScalar, AccScalar>::value),
                "A GEMM writes its result type, or the accumulator of a widened product.");

  static void run(Index rows, Index cols, Index depth, const LhsScalar* lhs_, Index lhsStride, const RhsScalar* rhs_,
                  Index rhsStride, ResScalar* res_, Index resIncr, Index resStride, AccScalar alpha,
                  level3_blocking<PackedLhsScalar, PackedRhsScalar>& blocking, GemmParallelInfo<Index>* info = 0) {
    // BLAS contract: if alpha == 0, the result is unchanged (and lhs/rhs need not be read).
    if (numext::is_exactly_zero(alpha)) return;
    run_impl(product_widens<LhsScalar, RhsScalar>(), rows, cols, depth, lhs_, lhsStride, rhs_, rhsStride, res_, resIncr,
             resStride, alpha, blocking, info);
  }

 private:
  using LhsMapper = const_blas_data_mapper<LhsScalar, Index, LhsStorageOrder>;
  using RhsMapper = const_blas_data_mapper<RhsScalar, Index, RhsStorageOrder>;

  static void run_impl(std::false_type, Index rows, Index cols, Index depth, const LhsScalar* lhs_, Index lhsStride,
                       const RhsScalar* rhs_, Index rhsStride, ResScalar* res_, Index resIncr, Index resStride,
                       ResScalar alpha, level3_blocking<LhsScalar, RhsScalar>& blocking,
                       GemmParallelInfo<Index>* info) {
#ifdef EIGEN_VECTORIZE_SME
    // A parallel session needs every thread to take the packed path; scaleAndAddTo runs tiny results on one thread.
    if (info == nullptr &&
        (sme_run_tiny_gemm<LhsStorageOrder, RhsStorageOrder>(
             bool_constant<sme_tiny_gemm_pair<LhsScalar, RhsScalar>::value>(), rows, cols, depth, lhs_, lhsStride, rhs_,
             rhsStride, res_, resIncr, resStride, alpha) ||
         sme_run_small_gemm<LhsStorageOrder, RhsStorageOrder, ConjugateLhs, ConjugateRhs, ResInnerStride>(
             bool_constant<sme_has_gebp_kernel<LhsScalar, RhsScalar>::value>(), rows, cols, depth, lhs_, lhsStride,
             rhs_, rhsStride, res_, resIncr, resStride, alpha, blocking)))
      return;
#endif

    using ResMapper = blas_data_mapper<typename Traits::ResScalar, Index, ColMajor, Unaligned, ResInnerStride>;
    LhsMapper lhs(lhs_, lhsStride);
    RhsMapper rhs(rhs_, rhsStride);
    ResMapper res(res_, resStride, resIncr);

    gemm_pack_lhs<LhsScalar, Index, LhsMapper, Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing,
                  LhsStorageOrder>
        pack_lhs;
    gemm_pack_rhs<RhsScalar, Index, RhsMapper, Traits::nr, RhsStorageOrder> pack_rhs;
    gebp_kernel<LhsScalar, RhsScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;

    gemm_blocked_product(rows, cols, depth, lhs, rhs, res, pack_lhs, pack_rhs, gebp, alpha, blocking, info);
  }

  // A widened product accumulates every coefficient in AccScalar over the whole depth, in a workspace W that is the
  // result itself when ResScalar is AccScalar, and rounds W to ResScalar once; rounding the result after each kc block
  // would round ceil(depth / kc) times. Each operand block is converted while it is packed, so the accumulator's
  // packers and kernel serve the product unchanged.
  static void run_impl(std::true_type, Index rows, Index cols, Index depth, const LhsScalar* lhs_, Index lhsStride,
                       const RhsScalar* rhs_, Index rhsStride, ResScalar* res_, Index resIncr, Index resStride,
                       AccScalar alpha, level3_blocking<AccScalar, AccScalar>& blocking,
                       GemmParallelInfo<Index>* info) {
    constexpr bool kResultIsWorkspace = std::is_same<ResScalar, AccScalar>::value;
    const Index kc = blocking.kc();
    const Index nc = (std::min)(cols, blocking.nc());
    // The LHS converts in chunks of whole mr-row panels sized to half of L1, the RHS a kc x nc block at a time.
    const Index mr = Traits::mr;
    const Index lhs_panels = numext::maxi<Index>(1, Index(l1CacheSize()) / (2 * mr * kc * Index(sizeof(AccScalar))));
    const Index lhs_chunk = numext::mini<Index>(rows, lhs_panels * mr);
    // Each buffer starts on a 64-byte boundary of the aligned allocation.
    const std::size_t line = numext::maxi<std::size_t>(1, 64 / sizeof(AccScalar));
    const std::size_t res_size =
        kResultIsWorkspace ? 0 : first_multiple<std::size_t>(std::size_t(rows) * std::size_t(cols), line);
    const std::size_t rhs_size = first_multiple<std::size_t>(std::size_t(kc) * std::size_t(nc), line);
    const std::size_t lhs_size = std::size_t(lhs_chunk) * std::size_t(kc);
    ei_declare_aligned_stack_constructed_variable(AccScalar, workspace, res_size + rhs_size + lhs_size, 0);
    AccScalar* const rhs_buffer = workspace + res_size;
    AccScalar* const lhs_buffer = rhs_buffer + rhs_size;

    LhsMapper lhs(lhs_, lhsStride);
    RhsMapper rhs(rhs_, rhsStride);
    using LhsPack = gemm_pack_lhs<AccScalar, Index, const_blas_data_mapper<AccScalar, Index, LhsStorageOrder>,
                                  Traits::mr, Traits::LhsProgress, typename Traits::LhsPacket4Packing, LhsStorageOrder>;
    using RhsPack = gemm_pack_rhs<AccScalar, Index, const_blas_data_mapper<AccScalar, Index, RhsStorageOrder>,
                                  Traits::nr, RhsStorageOrder>;
    gemm_converting_pack<Index, LhsScalar, AccScalar, LhsStorageOrder, true, LhsPack> pack_lhs(lhs_buffer, lhs_chunk);
    gemm_converting_pack<Index, RhsScalar, AccScalar, RhsStorageOrder, false, RhsPack> pack_rhs(rhs_buffer, nc);
    accumulate(bool_constant<kResultIsWorkspace>(), rows, cols, depth, lhs, rhs, res_, resIncr, resStride, workspace,
               pack_lhs, pack_rhs, alpha, blocking, info);
  }

  template <typename PackLhs, typename PackRhs>
  static void accumulate(std::true_type, Index rows, Index cols, Index depth, const LhsMapper& lhs,
                         const RhsMapper& rhs, ResScalar* res_, Index resIncr, Index resStride, AccScalar*,
                         PackLhs& pack_lhs, PackRhs& pack_rhs, AccScalar alpha,
                         level3_blocking<AccScalar, AccScalar>& blocking, GemmParallelInfo<Index>* info) {
    using ResMapper = blas_data_mapper<AccScalar, Index, ColMajor, Unaligned, ResInnerStride>;
    ResMapper res(res_, resStride, resIncr);
    gebp_kernel<AccScalar, AccScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;
    gemm_blocked_product(rows, cols, depth, lhs, rhs, res, pack_lhs, pack_rhs, gebp, alpha, blocking, info);
  }

  template <typename PackLhs, typename PackRhs>
  static void accumulate(std::false_type, Index rows, Index cols, Index depth, const LhsMapper& lhs,
                         const RhsMapper& rhs, ResScalar* res_, Index resIncr, Index resStride, AccScalar* workspace,
                         PackLhs& pack_lhs, PackRhs& pack_rhs, AccScalar alpha,
                         level3_blocking<AccScalar, AccScalar>& blocking, GemmParallelInfo<Index>* info) {
    using ResStride = Stride<Dynamic, ResInnerStride>;
    Map<Matrix<ResScalar, Dynamic, Dynamic, ColMajor>, 0, ResStride> result(res_, rows, cols,
                                                                            ResStride(resStride, resIncr));
    Map<Matrix<AccScalar, Dynamic, Dynamic, ColMajor>> sums(workspace, rows, cols);
    sums = result.template cast<AccScalar>();
    using ResMapper = blas_data_mapper<AccScalar, Index, ColMajor, Unaligned, 1>;
    ResMapper res(workspace, rows);
    gebp_kernel<AccScalar, AccScalar, Index, ResMapper, Traits::mr, Traits::nr, ConjugateLhs, ConjugateRhs> gebp;
    gemm_blocked_product(rows, cols, depth, lhs, rhs, res, pack_lhs, pack_rhs, gebp, alpha, blocking, info);
    result = sums.template cast<ResScalar>();
  }
};

/*********************************************************************************
 *  Specialization of generic_product_impl for "large" GEMM, i.e.,
 *  implementation of the high level wrapper to general_matrix_matrix_product
 **********************************************************************************/

template <typename AlphaScalar, typename Index, typename Gemm, typename Lhs, typename Rhs, typename Dest,
          typename BlockingType>
struct gemm_functor {
  gemm_functor(const Lhs& lhs, const Rhs& rhs, Dest& dest, const AlphaScalar& actualAlpha, BlockingType& blocking)
      : m_lhs(lhs), m_rhs(rhs), m_dest(dest), m_actualAlpha(actualAlpha), m_blocking(blocking) {}

  void initParallelSession(Index num_threads) const {
    m_blocking.initParallel(m_lhs.rows(), m_rhs.cols(), m_lhs.cols(), num_threads);
    m_blocking.allocateA();
  }

  // Whether the blocking holds no preallocated buffers, so that disjoint parts of the product can run concurrently.
  bool ownsNoBuffers() const { return m_blocking.blockA() == nullptr && m_blocking.blockB() == nullptr; }

  void operator()(Index row, Index rows, Index col = 0, Index cols = -1, GemmParallelInfo<Index>* info = 0) const {
    if (cols == -1) cols = m_rhs.cols();

    Gemm::run(rows, cols, m_lhs.cols(), &m_lhs.coeffRef(row, 0), m_lhs.outerStride(), &m_rhs.coeffRef(0, col),
              m_rhs.outerStride(), (typename Dest::Scalar*)&(m_dest.coeffRef(row, col)), m_dest.innerStride(),
              m_dest.outerStride(), m_actualAlpha, m_blocking, info);
  }

  using Traits = typename Gemm::Traits;

 protected:
  const Lhs& m_lhs;
  const Rhs& m_rhs;
  Dest& m_dest;
  AlphaScalar m_actualAlpha;
  BlockingType& m_blocking;
};

template <int StorageOrder, typename LhsScalar, typename RhsScalar, int MaxRows, int MaxCols, int MaxDepth,
          int KcFactor = 1, bool FiniteAtCompileTime = MaxRows != Dynamic && MaxCols != Dynamic && MaxDepth != Dynamic>
class gemm_blocking_space;

template <typename LhsScalar_, typename RhsScalar_>
class level3_blocking {
  using LhsScalar = LhsScalar_;
  using RhsScalar = RhsScalar_;

 protected:
  LhsScalar* m_blockA = nullptr;
  RhsScalar* m_blockB = nullptr;

  Index m_mc = 0;
  Index m_nc = 0;
  Index m_kc = 0;

 public:
  level3_blocking() = default;

  inline Index mc() const { return m_mc; }
  inline Index nc() const { return m_nc; }
  inline Index kc() const { return m_kc; }

  inline LhsScalar* blockA() { return m_blockA; }
  inline RhsScalar* blockB() { return m_blockB; }
};

template <int StorageOrder, typename LhsScalar_, typename RhsScalar_, int MaxRows, int MaxCols, int MaxDepth,
          int KcFactor>
class gemm_blocking_space<StorageOrder, LhsScalar_, RhsScalar_, MaxRows, MaxCols, MaxDepth, KcFactor,
                          true /* == FiniteAtCompileTime */>
    : public level3_blocking<std::conditional_t<StorageOrder == RowMajor, RhsScalar_, LhsScalar_>,
                             std::conditional_t<StorageOrder == RowMajor, LhsScalar_, RhsScalar_>> {
  enum {
    Transpose = StorageOrder == RowMajor,
    ActualRows = Transpose ? MaxCols : MaxRows,
    ActualCols = Transpose ? MaxRows : MaxCols
  };
  using LhsScalar = std::conditional_t<Transpose, RhsScalar_, LhsScalar_>;
  using RhsScalar = std::conditional_t<Transpose, LhsScalar_, RhsScalar_>;
  // A zero-sized fixed dimension must not declare a zero-length array, which MSVC rejects (C2229); every caller
  // returns before it touches the blocks of an empty product.
  static constexpr int SizeA = plain_enum_max(int(ActualRows) * MaxDepth, 1);
  static constexpr int SizeB = plain_enum_max(int(ActualCols) * MaxDepth, 1);

#if EIGEN_MAX_STATIC_ALIGN_BYTES >= EIGEN_DEFAULT_ALIGN_BYTES
  EIGEN_ALIGN_MAX LhsScalar m_staticA[SizeA > 0 ? SizeA : 1];
  EIGEN_ALIGN_MAX RhsScalar m_staticB[SizeB > 0 ? SizeB : 1];
#else
  EIGEN_ALIGN_MAX char m_staticA[(SizeA > 0 ? SizeA : 1) * sizeof(LhsScalar) + EIGEN_DEFAULT_ALIGN_BYTES - 1];
  EIGEN_ALIGN_MAX char m_staticB[(SizeB > 0 ? SizeB : 1) * sizeof(RhsScalar) + EIGEN_DEFAULT_ALIGN_BYTES - 1];
#endif

 public:
  gemm_blocking_space(Index /*rows*/, Index /*cols*/, Index /*depth*/, Index /*num_threads*/,
                      bool /*full_rows = false*/) {
    this->m_mc = ActualRows;
    this->m_nc = ActualCols;
    this->m_kc = MaxDepth;
#if EIGEN_MAX_STATIC_ALIGN_BYTES >= EIGEN_DEFAULT_ALIGN_BYTES
    this->m_blockA = m_staticA;
    this->m_blockB = m_staticB;
#else
    this->m_blockA = reinterpret_cast<LhsScalar*>((std::uintptr_t(m_staticA) + (EIGEN_DEFAULT_ALIGN_BYTES - 1)) &
                                                  ~std::size_t(EIGEN_DEFAULT_ALIGN_BYTES - 1));
    this->m_blockB = reinterpret_cast<RhsScalar*>((std::uintptr_t(m_staticB) + (EIGEN_DEFAULT_ALIGN_BYTES - 1)) &
                                                  ~std::size_t(EIGEN_DEFAULT_ALIGN_BYTES - 1));
#endif
  }

  void initParallel(Index, Index, Index, Index) {}

  inline void allocateA() {}
  inline void allocateB() {}
  inline void allocateAll() {}
};

template <int StorageOrder, typename LhsScalar_, typename RhsScalar_, int MaxRows, int MaxCols, int MaxDepth,
          int KcFactor>
class gemm_blocking_space<StorageOrder, LhsScalar_, RhsScalar_, MaxRows, MaxCols, MaxDepth, KcFactor, false>
    : public level3_blocking<std::conditional_t<StorageOrder == RowMajor, RhsScalar_, LhsScalar_>,
                             std::conditional_t<StorageOrder == RowMajor, LhsScalar_, RhsScalar_>> {
  enum { Transpose = StorageOrder == RowMajor };
  using LhsScalar = std::conditional_t<Transpose, RhsScalar_, LhsScalar_>;
  using RhsScalar = std::conditional_t<Transpose, LhsScalar_, RhsScalar_>;

  Index m_sizeA;
  Index m_sizeB;

 public:
  gemm_blocking_space(Index rows, Index cols, Index depth, Index num_threads, bool l3_blocking) {
    this->m_mc = Transpose ? cols : rows;
    this->m_nc = Transpose ? rows : cols;
    this->m_kc = depth;

    if (l3_blocking) {
      computeProductBlockingSizes<LhsScalar, RhsScalar, KcFactor>(this->m_kc, this->m_mc, this->m_nc, num_threads);
    } else  // no l3 blocking
    {
      Index n = this->m_nc;
      computeProductBlockingSizes<LhsScalar, RhsScalar, KcFactor>(this->m_kc, this->m_mc, n, num_threads);
    }

    m_sizeA = this->m_mc * this->m_kc;
    m_sizeB = this->m_kc * this->m_nc;
  }

  void initParallel(Index rows, Index cols, Index depth, Index num_threads) {
    this->m_mc = Transpose ? cols : rows;
    this->m_nc = Transpose ? rows : cols;
    this->m_kc = depth;

    eigen_internal_assert(this->m_blockA == 0 && this->m_blockB == 0);
    Index m = this->m_mc;
    computeProductBlockingSizes<LhsScalar, RhsScalar, KcFactor>(this->m_kc, m, this->m_nc, num_threads);
    m_sizeA = this->m_mc * this->m_kc;
    m_sizeB = this->m_kc * this->m_nc;
  }

  // The blocking buffers get the temporaries' alignment (64 bytes in SME builds, EIGEN_STACK_ALIGN_BYTES).
  void allocateA() {
    if (this->m_blockA == 0) this->m_blockA = scratch_new<LhsScalar>(m_sizeA);
  }

  void allocateB() {
    if (this->m_blockB == 0) this->m_blockB = scratch_new<RhsScalar>(m_sizeB);
  }

  void allocateAll() {
    allocateA();
    allocateB();
  }

  ~gemm_blocking_space() {
    scratch_delete(this->m_blockA, m_sizeA);
    scratch_delete(this->m_blockB, m_sizeB);
  }
};

}  // end namespace internal

namespace internal {

template <typename Lhs, typename Rhs>
struct generic_product_impl<Lhs, Rhs, DenseShape, DenseShape, GemmProduct>
    : generic_product_impl_base<Lhs, Rhs, generic_product_impl<Lhs, Rhs, DenseShape, DenseShape, GemmProduct>> {
  using Scalar = typename Product<Lhs, Rhs>::Scalar;
  using LhsScalar = typename Lhs::Scalar;
  using RhsScalar = typename Rhs::Scalar;
  using AccScalar = typename product_accumulator<LhsScalar, RhsScalar>::type;

  using LhsBlasTraits = internal::blas_traits<Lhs>;
  using ActualLhsType = typename LhsBlasTraits::DirectLinearAccessType;
  using ActualLhsTypeCleaned = internal::remove_all_t<ActualLhsType>;

  using RhsBlasTraits = internal::blas_traits<Rhs>;
  using ActualRhsType = typename RhsBlasTraits::DirectLinearAccessType;
  using ActualRhsTypeCleaned = internal::remove_all_t<ActualRhsType>;

  enum { MaxDepthAtCompileTime = min_size_prefer_fixed(Lhs::MaxColsAtCompileTime, Rhs::MaxRowsAtCompileTime) };

  using lazyproduct = generic_product_impl<Lhs, Rhs, DenseShape, DenseShape, CoeffBasedProductMode>;

  // The runtime-size heuristic comes from bug 404 and was tuned with a
  // helper program on Haswell. The threshold belongs to the kernel the GEMM
  // path would select, not to the path itself, and for the SME kernel to the
  // scalar type as well: it needs a larger product before it beats the
  // coeff-based one, by an amount that falls as the scalar widens (see
  // GeneralProduct.h). The rhs.rows() > 0 guard preserves the historical
  // empty-product path through scaleAndAddTo().
  static constexpr int kCoeffBasedThreshold =
#ifdef EIGEN_VECTORIZE_SME
      sme_has_gebp_kernel<LhsScalar, RhsScalar>::value ? sme_gemm_to_coeffbased_threshold<Scalar>::value :
#endif
                                                       EIGEN_GEMM_TO_COEFFBASED_THRESHOLD;

#ifdef EIGEN_VECTORIZE_SME
  // Second bound for the SME kernel only: a small output over a long depth.
  // The sum above grows with the depth and so never catches it, while the ZA
  // grid this kernel fills is sized by the output (see GeneralProduct.h).
  // Vector shapes are excluded -- scaleAndAddTo() routes those to GEMV, which
  // is not the path being compared here.
  static constexpr Index kCoeffBasedOutputArea = sme_has_gebp_kernel<LhsScalar, RhsScalar>::value
                                                     ? Index(EIGEN_SME_GEMM_TO_COEFFBASED_OUTPUT_AREA_THRESHOLD(Scalar))
                                                     : Index(0);

  template <typename Dst>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool outputAreaBelowThreshold(const Dst& dst) {
    // rows * cols is dst.size(), which fits Index by construction.
    return dst.rows() > 1 && dst.cols() > 1 && dst.size() <= kCoeffBasedOutputArea;
  }
#endif

#ifdef EIGEN_VECTORIZE_SME
  // Whether the tiny-result kernel takes a product the coeff-based one would; the GEMM driver sees a RowMajor
  // result transposed.
  template <typename Dst>
  static EIGEN_STRONG_INLINE bool tinyKernelWins(const Dst& dst, const Rhs& rhs) {
    constexpr bool row_major = (Dst::Flags & RowMajorBit) != 0;
    return sme_tiny_gemm_pair<LhsScalar, RhsScalar>::value &&
           sme_tiny_gemm_wins<Scalar>(row_major ? dst.cols() : dst.rows(), row_major ? dst.rows() : dst.cols(),
                                      rhs.rows());
  }
#endif

  template <typename Dst>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool useRuntimeCoeffBasedProduct(const Dst& dst, const Rhs& rhs) {
    if (rhs.rows() <= 0) return false;
#ifdef EIGEN_VECTORIZE_SME
    if ((rhs.rows() + dst.rows() + dst.cols()) < kCoeffBasedThreshold || outputAreaBelowThreshold(dst))
      return !tinyKernelWins(dst, rhs);
#else
    if ((rhs.rows() + dst.rows() + dst.cols()) < kCoeffBasedThreshold) return true;
#endif
    return false;
  }

  // BLAS contract: a zero scalar factor leaves the destination unchanged and
  // neither operand need be read, so that a non-finite coefficient cannot taint
  // the result through 0 * Inf. general_matrix_matrix_product::run enforces it
  // for the GEMM path, but the coeff-based path below has no such exit and
  // would evaluate the product, so the factor is tested before the dispatch
  // rather than inside either kernel.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool scalarFactorIsZero(const Lhs& lhs, const Rhs& rhs) {
    return scalarFactorIsZero(lhs, rhs, product_widens<LhsScalar, RhsScalar>());
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool scalarFactorIsZero(const Lhs& lhs, const Rhs& rhs,
                                                                       std::false_type) {
    return numext::is_exactly_zero(combine_scalar_factors<Scalar>(lhs, rhs));
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool scalarFactorIsZero(const Lhs& lhs, const Rhs& rhs, std::true_type) {
    return numext::is_exactly_zero(combinedScalarFactors(Scalar(1), lhs, rhs, std::true_type()));
  }

  // A widened product combines its scalar factors in its accumulator, so that s * t neither rounds nor underflows in
  // the operand type.
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AccScalar combinedScalarFactors(const Scalar& alpha, const Lhs& lhs,
                                                                               const Rhs& rhs, std::false_type) {
    return combine_scalar_factors(alpha, lhs, rhs);
  }
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE AccScalar combinedScalarFactors(const Scalar& alpha, const Lhs& lhs,
                                                                               const Rhs& rhs, std::true_type) {
    return AccScalar(alpha) * AccScalar(LhsBlasTraits::extractScalarFactor(lhs)) *
           AccScalar(RhsBlasTraits::extractScalarFactor(rhs));
  }

  template <typename Dst>
  static void evalTo(Dst& dst, const Lhs& lhs, const Rhs& rhs) {
    if (scalarFactorIsZero(lhs, rhs)) {
      dst.setZero();
      return;
    }
    if (useRuntimeCoeffBasedProduct(dst, rhs))
      lazyproduct::eval_dynamic(dst, lhs, rhs, internal::assign_op<typename Dst::Scalar, Scalar>());
    else {
      dst.setZero();
      scaleAndAddTo(dst, lhs, rhs, Scalar(1));
    }
  }

  template <typename Dst>
  static void addTo(Dst& dst, const Lhs& lhs, const Rhs& rhs) {
    if (scalarFactorIsZero(lhs, rhs)) return;
    if (useRuntimeCoeffBasedProduct(dst, rhs))
      lazyproduct::eval_dynamic(dst, lhs, rhs, internal::add_assign_op<typename Dst::Scalar, Scalar>());
    else
      scaleAndAddTo(dst, lhs, rhs, Scalar(1));
  }

  template <typename Dst>
  static void subTo(Dst& dst, const Lhs& lhs, const Rhs& rhs) {
    if (scalarFactorIsZero(lhs, rhs)) return;
    if (useRuntimeCoeffBasedProduct(dst, rhs))
      lazyproduct::eval_dynamic(dst, lhs, rhs, internal::sub_assign_op<typename Dst::Scalar, Scalar>());
    else
      scaleAndAddTo(dst, lhs, rhs, Scalar(-1));
  }

  template <typename Dest>
  static void scaleAndAddTo(Dest& dst, const Lhs& a_lhs, const Rhs& a_rhs, const Scalar& alpha) {
    EIGEN_INTERNAL_ASSERT_SAME_ROWS(dst, a_lhs);
    EIGEN_INTERNAL_ASSERT_SAME_COLS(dst, a_rhs);
    if (a_lhs.cols() == 0 || a_lhs.rows() == 0 || a_rhs.cols() == 0) return;

    if (dst.cols() == 1) {
      // Fallback to GEMV if either the lhs or rhs is a runtime vector
      typename Dest::ColXpr dst_vec(dst.col(0));
      return internal::generic_product_impl<Lhs, typename Rhs::ConstColXpr, DenseShape, DenseShape,
                                            GemvProduct>::scaleAndAddTo(dst_vec, a_lhs, a_rhs.col(0), alpha);
    } else if (dst.rows() == 1) {
      // Fallback to GEMV if either the lhs or rhs is a runtime vector
      typename Dest::RowXpr dst_vec(dst.row(0));
      return internal::generic_product_impl<typename Lhs::ConstRowXpr, Rhs, DenseShape, DenseShape,
                                            GemvProduct>::scaleAndAddTo(dst_vec, a_lhs.row(0), a_rhs, alpha);
    }

    add_const_on_value_type_t<ActualLhsType> lhs = LhsBlasTraits::extract(a_lhs);
    add_const_on_value_type_t<ActualRhsType> rhs = RhsBlasTraits::extract(a_rhs);

    const AccScalar actualAlpha = combinedScalarFactors(alpha, a_lhs, a_rhs, product_widens<LhsScalar, RhsScalar>());

    using BlockingType =
        internal::gemm_blocking_space<(Dest::Flags & RowMajorBit) ? RowMajor : ColMajor,
                                      gemm_packed_scalar<LhsScalar, LhsScalar, RhsScalar>,
                                      gemm_packed_scalar<RhsScalar, LhsScalar, RhsScalar>, Dest::MaxRowsAtCompileTime,
                                      Dest::MaxColsAtCompileTime, MaxDepthAtCompileTime>;

    using GemmFunctor = internal::gemm_functor<
        AccScalar, Index,
        internal::general_matrix_matrix_product<
            Index, LhsScalar, (ActualLhsTypeCleaned::Flags & RowMajorBit) ? RowMajor : ColMajor,
            bool(LhsBlasTraits::NeedToConjugate), RhsScalar,
            (ActualRhsTypeCleaned::Flags & RowMajorBit) ? RowMajor : ColMajor, bool(RhsBlasTraits::NeedToConjugate),
            (Dest::Flags & RowMajorBit) ? RowMajor : ColMajor, Dest::InnerStrideAtCompileTime>,
        ActualLhsTypeCleaned, ActualRhsTypeCleaned, Dest, BlockingType>;

    BlockingType blocking(dst.rows(), dst.cols(), lhs.cols(), 1, true);
#ifdef EIGEN_VECTORIZE_SME
    // A result the tiny-result kernel takes is never worth a parallel session.
    if (tinyKernelWins(dst, a_rhs))
      return internal::parallelize_gemm<false>(GemmFunctor(lhs, rhs, dst, actualAlpha, blocking), a_lhs.rows(),
                                               a_rhs.cols(), a_lhs.cols(), Dest::Flags & RowMajorBit);
#endif
    internal::parallelize_gemm<(Dest::MaxRowsAtCompileTime > 32 || Dest::MaxRowsAtCompileTime == Dynamic)>(
        GemmFunctor(lhs, rhs, dst, actualAlpha, blocking), a_lhs.rows(), a_rhs.cols(), a_lhs.cols(),
        Dest::Flags & RowMajorBit);
  }
};

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_GENERAL_MATRIX_MATRIX_H

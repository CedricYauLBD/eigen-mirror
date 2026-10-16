// Benchmarks for the memory_resource primitives against Eigen's handmade aligned malloc.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include <benchmark/benchmark.h>
#include <Eigen/Core>

using namespace Eigen;

static void BM_HandmadeAlignedMalloc(benchmark::State& state) {
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) {
    void* p = internal::handmade_aligned_malloc(bytes);
    benchmark::DoNotOptimize(p);
    internal::handmade_aligned_free(p);
  }
}

static void BM_NewDeleteResource(benchmark::State& state) {
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  memory_resource* r = new_delete_resource();
  for (auto _ : state) {
    void* p = r->allocate(bytes);
    benchmark::DoNotOptimize(p);
    r->deallocate(p, bytes);
  }
}

static void BM_ByteAllocator(benchmark::State& state) {
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  byte_allocator alloc;
  for (auto _ : state) {
    void* p = alloc.allocate(bytes);
    benchmark::DoNotOptimize(p);
    alloc.deallocate(p, bytes);
  }
}

// One timed iteration is `count` allocations of `bytes`; the arena is released between iterations.
static void BM_MonotonicBatch(benchmark::State& state) {
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  const std::size_t count = static_cast<std::size_t>(state.range(1));
  monotonic_buffer_resource arena(2 * bytes * count);
  for (auto _ : state) {
    for (std::size_t i = 0; i < count; ++i) {
      void* p = arena.allocate(bytes);
      benchmark::DoNotOptimize(p);
    }
    arena.release();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}

static void BM_HandmadeAlignedMallocBatch(benchmark::State& state) {
  const std::size_t bytes = static_cast<std::size_t>(state.range(0));
  const std::size_t count = static_cast<std::size_t>(state.range(1));
  for (auto _ : state) {
    for (std::size_t i = 0; i < count; ++i) {
      void* p = internal::handmade_aligned_malloc(bytes);
      benchmark::DoNotOptimize(p);
      internal::handmade_aligned_free(p);
    }
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}

BENCHMARK(BM_HandmadeAlignedMalloc)->Arg(128)->Arg(256)->Arg(4096);
BENCHMARK(BM_NewDeleteResource)->Arg(128)->Arg(256)->Arg(4096);
BENCHMARK(BM_ByteAllocator)->Arg(128)->Arg(256)->Arg(4096);
BENCHMARK(BM_MonotonicBatch)->Args({128, 1000})->Args({256, 1000});
BENCHMARK(BM_HandmadeAlignedMallocBatch)->Args({128, 1000})->Args({256, 1000});

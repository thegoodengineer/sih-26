// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unit tests for GPU PDHG memory estimation (#281).
//
// These tests run without a CUDA device: the estimator is pure arithmetic over model
// dimensions. Tests verify that the formula is monotone, accounts for all tracked
// allocations, and agrees with the policy constants documented in gpu_memory.hpp.

#include <gtest/gtest.h>

#include "gpu/gpu_memory.hpp"

namespace sankhya {
namespace {

using gpu::estimate_pdhg_gpu_memory;
using gpu::vram_reserve;

// ---- estimate_pdhg_gpu_memory -----------------------------------------------

TEST(GpuMemory, ZeroModelHasLibraryOverheadOnly) {
  // An empty model still needs the 16 MiB library-overhead constant.
  const std::size_t est = estimate_pdhg_gpu_memory(0, 0, 0);
  EXPECT_GE(est, 16ULL * 1024 * 1024);
}

TEST(GpuMemory, EstimateIncreasesWithNnz) {
  const std::size_t a = estimate_pdhg_gpu_memory(100, 200, 0);
  const std::size_t b = estimate_pdhg_gpu_memory(100, 200, 1000);
  const std::size_t c = estimate_pdhg_gpu_memory(100, 200, 100000);
  EXPECT_LT(a, b);
  EXPECT_LT(b, c);
}

TEST(GpuMemory, EstimateIncreasesWithRows) {
  const std::size_t a = estimate_pdhg_gpu_memory(1000, 500, 5000);
  const std::size_t b = estimate_pdhg_gpu_memory(10000, 500, 5000);
  EXPECT_LT(a, b);
}

TEST(GpuMemory, EstimateIncreasesWithCols) {
  const std::size_t a = estimate_pdhg_gpu_memory(500, 1000, 5000);
  const std::size_t b = estimate_pdhg_gpu_memory(500, 5000, 5000);
  EXPECT_LT(a, b);
}

TEST(GpuMemory, NegativeDimensionsAreClampedToZero) {
  // Should not underflow or crash.
  const std::size_t est = estimate_pdhg_gpu_memory(-1, -1, -1);
  EXPECT_GE(est, 16ULL * 1024 * 1024);
}

TEST(GpuMemory, SmallModelFormula) {
  // m=10, n=20, nnz=50:
  //   10*20*8 = 1600   n-vecs
  //    9*10*8 =  720   m-vecs
  //         8 =    8   scalar
  //    (10+1)*4 = 44   rowptr
  //    50*4   =  200   colidx
  //    50*8   =  400   vals
  //  16*1024^2        library overhead
  constexpr std::size_t kOverhead = 16ULL * 1024 * 1024;
  constexpr std::size_t expected = 1600 + 720 + 8 + 44 + 200 + 400 + kOverhead;
  EXPECT_EQ(estimate_pdhg_gpu_memory(10, 20, 50), expected);
}

TEST(GpuMemory, LargeModelIsInMibRange) {
  // 100k rows, 200k cols, 1M nonzeros: sparse storage ≈ 50 MiB (not GiB — the matrix is
  // sparse). n-vecs: 10*200k*8=16 MiB; m-vecs: 9*100k*8=7.2 MiB; CSR: ~12 MiB; 16 MiB
  // overhead. The estimate must exceed 40 MiB, well above the 16 MiB overhead constant alone.
  const std::size_t est = estimate_pdhg_gpu_memory(100000, 200000, 1000000);
  constexpr std::size_t k40Mib = 40ULL * 1024 * 1024;
  EXPECT_GT(est, k40Mib);
}

// ---- vram_reserve -----------------------------------------------------------

TEST(GpuMemory, ReserveIsAtLeast256Mib) {
  // Even with zero total VRAM the floor applies.
  EXPECT_GE(vram_reserve(0), 256ULL * 1024 * 1024);
}

TEST(GpuMemory, ReserveIsTenPercentWhenLarger) {
  // 10 GiB total → 10% = 1 GiB > 256 MiB floor.
  constexpr std::size_t total = 10ULL * 1024 * 1024 * 1024;
  EXPECT_EQ(vram_reserve(total), total / 10);
}

TEST(GpuMemory, ReserveFloorAppliesForSmallVram) {
  // 1 GiB total → 10% = 102 MiB < 256 MiB floor.
  constexpr std::size_t total = 1ULL * 1024 * 1024 * 1024;
  EXPECT_EQ(vram_reserve(total), 256ULL * 1024 * 1024);
}

TEST(GpuMemory, ReserveAt6GibVram) {
  // 6144 MiB (RTX 4050 Laptop GPU) → 10% = 614 MiB > 256 MiB floor.
  constexpr std::size_t total = 6144ULL * 1024 * 1024;
  EXPECT_EQ(vram_reserve(total), total / 10);
}

}  // namespace
}  // namespace sankhya

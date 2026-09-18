// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unit tests for GPU architecture compatibility logic (#282).
//
// Tests the pure-arithmetic capability predicate. No CUDA device or GPU needed.

#include <gtest/gtest.h>

#include "gpu/gpu_memory.hpp"

namespace sankhya {
namespace {

using gpu::is_supported_compute_capability;
using gpu::kMinComputeArch;

// ---- kMinComputeArch constant ---------------------------------------------------

TEST(GpuDevice, MinComputeArchIsNonZero) {
  EXPECT_GT(kMinComputeArch, 0);
}

// ---- is_supported_compute_capability -------------------------------------------

TEST(GpuDevice, TuringIsSupported) {
  // sm_75 (RTX 20xx, T4): exactly the minimum compiled architecture.
  EXPECT_TRUE(is_supported_compute_capability(7, 5));
}

TEST(GpuDevice, AmpereIsSupported) {
  // sm_86 (RTX 30xx)
  EXPECT_TRUE(is_supported_compute_capability(8, 6));
}

TEST(GpuDevice, AdaLovelaceIsSupported) {
  // sm_89 (RTX 40xx)
  EXPECT_TRUE(is_supported_compute_capability(8, 9));
}

TEST(GpuDevice, HopperIsSupported) {
  // sm_90 (H100) — newer than the compiled list but forward-compatible via PTX JIT.
  EXPECT_TRUE(is_supported_compute_capability(9, 0));
}

TEST(GpuDevice, PascalIsUnsupported) {
  // sm_61 (GTX 10xx) — below Turing.
  EXPECT_FALSE(is_supported_compute_capability(6, 1));
}

TEST(GpuDevice, VoltaIsUnsupported) {
  // sm_70 (V100) — just below the 7.5 threshold.
  EXPECT_FALSE(is_supported_compute_capability(7, 0));
}

TEST(GpuDevice, Compute74IsUnsupported) {
  // One minor version below the threshold.
  EXPECT_FALSE(is_supported_compute_capability(7, 4));
}

TEST(GpuDevice, ZeroCapabilityIsUnsupported) {
  EXPECT_FALSE(is_supported_compute_capability(0, 0));
}

TEST(GpuDevice, NegativeCapabilityIsUnsupported) {
  EXPECT_FALSE(is_supported_compute_capability(-1, 0));
}

}  // namespace
}  // namespace sankhya

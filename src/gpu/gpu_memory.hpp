// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU VRAM estimation for the PDHG backend (#281).
//
// The estimator is pure arithmetic over model dimensions — no CUDA calls, no allocations.
// It can therefore be used from CPU-only translation units and in unit tests that run
// without a GPU.
#pragma once

#include <cstddef>

#include "sankhya/types.hpp"

namespace sankhya::gpu {

/// Minimum NVIDIA compute capability this build can run on, as major*10 + minor. A CUDA
/// build defines SANKHYA_MIN_COMPUTE_ARCH from the lowest numeric entry of
/// SANKHYA_CUDA_ARCHITECTURES (CMakeLists.txt), so the runtime check cannot drift from what
/// was compiled; the CPU build, and a symbolic architecture list, use the documented default
/// of the default list, 75 (Turing sm_75).
#ifndef SANKHYA_MIN_COMPUTE_ARCH
#define SANKHYA_MIN_COMPUTE_ARCH 75
#endif
inline constexpr int kMinComputeArch = SANKHYA_MIN_COMPUTE_ARCH;

/// Estimate the peak device-side bytes required to run GPU PDHG on a model with the given
/// dimensions. The calculation accounts for:
///   - 10 n-element double vectors (primal iterates, sums, scratch, cost, column bounds)
///   - 9 m-element double vectors (dual iterates, sums, scratch, row bounds)
///   - CSR storage: (m+1) int32 row offsets + nnz int32 column indices + nnz double values
///   - One scalar double for reductions
///   - A conservative library-overhead constant for cuSPARSE SpMV and CUB temp buffers
///
/// The constant overheads are chosen to be safe across the architecture range (sm_75–sm_89).
/// The function is intentionally conservative; a model near the limit may succeed in practice.
[[nodiscard]] std::size_t estimate_pdhg_gpu_memory(Index rows, Index cols, Count nonzeros);

/// Return true when a device with the given compute capability (major.minor) meets the
/// minimum architecture compiled into this build (kMinComputeArch).
/// Pure arithmetic — no CUDA calls — so safe to call from CPU-only translation units.
[[nodiscard]] bool is_supported_compute_capability(int major, int minor) noexcept;

/// The safety reserve applied to free VRAM before comparing with the estimate.
/// Defined as max(10% of total VRAM, 256 MiB). Kept here so tests can match the policy.
[[nodiscard]] std::size_t vram_reserve(std::size_t total_bytes);

}  // namespace sankhya::gpu

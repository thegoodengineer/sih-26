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

/// The safety reserve applied to free VRAM before comparing with the estimate.
/// Defined as max(10% of total VRAM, 256 MiB). Kept here so tests can match the policy.
[[nodiscard]] std::size_t vram_reserve(std::size_t total_bytes);

}  // namespace sankhya::gpu

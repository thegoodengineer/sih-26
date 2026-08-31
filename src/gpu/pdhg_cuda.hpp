// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CUDA sparse matrix backend interface.

#pragma once

#include <cstddef>
#include <cstdint>

namespace sankhya::gpu {

struct PdhgCudaContext;

/// Upload a frozen CSC sparse matrix to the GPU.
[[nodiscard]] PdhgCudaContext* pdhg_cuda_create(
    std::size_t rows,
    std::size_t cols,
    std::size_t nnz,
    const std::int32_t* column_starts,
    const std::int32_t* row_indices,
    const double* values);

/// Upload a vector to the GPU x buffer.
[[nodiscard]] bool pdhg_cuda_upload_x(
    PdhgCudaContext* context,
    const double* host,
    std::size_t size);

/// Upload a vector to the GPU y buffer.
[[nodiscard]] bool pdhg_cuda_upload_y(
    PdhgCudaContext* context,
    const double* host,
    std::size_t size);

/// Download the GPU x buffer to a host vector.
[[nodiscard]] bool pdhg_cuda_download_x(
    PdhgCudaContext* context,
    double* host,
    std::size_t size);

/// Download the GPU y buffer to a host vector.
[[nodiscard]] bool pdhg_cuda_download_y(
    PdhgCudaContext* context,
    double* host,
    std::size_t size);

/// Compute y = A*x using GPU-resident vectors.
[[nodiscard]] bool pdhg_cuda_multiply_device(
    PdhgCudaContext* context,
    std::size_t x_size,
    std::size_t y_size);

/// Compute y = A^T*x using GPU-resident vectors.
[[nodiscard]] bool pdhg_cuda_transpose_multiply_device(
    PdhgCudaContext* context,
    std::size_t x_size,
    std::size_t y_size);

/// Return the device-side x buffer.
[[nodiscard]] double* pdhg_cuda_x(PdhgCudaContext* context) noexcept;

/// Return the device-side y buffer.
[[nodiscard]] double* pdhg_cuda_y(PdhgCudaContext* context) noexcept;

/// Release GPU resources.
void pdhg_cuda_destroy(PdhgCudaContext* context) noexcept;

}  // namespace sankhya::gpu
// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CUDA sparse matrix backend interface.

#pragma once

#include <cstddef>
#include <cstdint>

namespace sankhya::gpu {

struct PdhgCudaContext;

/// Create a CUDA context and upload the frozen CSC matrix, constructing CSR representation on GPU.
[[nodiscard]] PdhgCudaContext* pdhg_cuda_create(
    std::size_t rows,
    std::size_t cols,
    std::size_t nnz,
    const std::int32_t* column_starts,
    const std::int32_t* row_indices,
    const double* values);

/// Upload static problem data once.
[[nodiscard]] bool pdhg_cuda_upload_problem_data(
    PdhgCudaContext* context,
    const double* cost,
    const double* col_lower,
    const double* col_upper,
    const double* row_lower,
    const double* row_upper);

/// Upload the initial primal vector.
[[nodiscard]] bool pdhg_cuda_upload_x(
    PdhgCudaContext* context,
    const double* host,
    std::size_t size);

/// Upload the initial dual vector.
[[nodiscard]] bool pdhg_cuda_upload_y(
    PdhgCudaContext* context,
    const double* host,
    std::size_t size);

/// Calculate a candidate PDHG step on the GPU without committing to x and y.
/// Computes movement and interaction scalars on GPU and returns them to host.
[[nodiscard]] bool pdhg_cuda_step(
    PdhgCudaContext* context,
    double tau,
    double sigma,
    double omega,
    double* movement,
    double* interaction);

/// Accept and commit the candidate step on the GPU (x <- x_next, y <- y_next)
/// and accumulate running sum vectors for averaging.
[[nodiscard]] bool pdhg_cuda_accept_step(
    PdhgCudaContext* context);

/// Download the running average vectors (x_sum / count, y_sum / count).
[[nodiscard]] bool pdhg_cuda_download_average(
    PdhgCudaContext* context,
    double* host_x_avg,
    double* host_y_avg,
    std::size_t count);

/// Reset running average sum vectors on the GPU (used on restart).
[[nodiscard]] bool pdhg_cuda_reset_sum(
    PdhgCudaContext* context);

/// Download the current primal vector.
[[nodiscard]] bool pdhg_cuda_download_x(
    PdhgCudaContext* context,
    double* host,
    std::size_t size);

/// Download the current dual vector.
[[nodiscard]] bool pdhg_cuda_download_y(
    PdhgCudaContext* context,
    double* host,
    std::size_t size);

/// Return the device-side primal vector.
[[nodiscard]] double* pdhg_cuda_x(
    PdhgCudaContext* context) noexcept;

/// Return the device-side dual vector.
[[nodiscard]] double* pdhg_cuda_y(
    PdhgCudaContext* context) noexcept;

/// Release all CUDA resources.
void pdhg_cuda_destroy(
    PdhgCudaContext* context) noexcept;

}  // namespace sankhya::gpu
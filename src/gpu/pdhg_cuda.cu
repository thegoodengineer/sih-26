// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CUDA sparse matrix backend.

#include "gpu/pdhg_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sankhya::gpu {

struct PdhgCudaContext {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t nnz = 0;

  std::int32_t* d_column_starts = nullptr;
  std::int32_t* d_row_indices = nullptr;
  double* d_values = nullptr;

  // Persistent GPU-resident vectors.
  //
  // d_x:
  //   Input vector for matrix-vector operations.
  //
  // d_y:
  //   General persistent vector.
  //
  // d_y_work:
  //   Output buffer for A*x and A^T*x.
  //
  // d_x_work is reserved for future GPU-side PDHG operations.
  double* d_x = nullptr;
  double* d_y = nullptr;

  double* d_x_work = nullptr;
  double* d_y_work = nullptr;
};

namespace {

__global__ void csc_multiply_kernel(
    std::size_t cols,
    const std::int32_t* column_starts,
    const std::int32_t* row_indices,
    const double* values,
    const double* x,
    double* y) {
  const std::size_t col =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (col >= cols) {
    return;
  }

  const std::int32_t begin = column_starts[col];
  const std::int32_t end = column_starts[col + 1];

  const double x_value = x[col];

  for (std::int32_t k = begin; k < end; ++k) {
    const std::int32_t row = row_indices[k];

    atomicAdd(
        &y[row],
        values[k] * x_value);
  }
}

__global__ void csc_transpose_multiply_kernel(
    std::size_t cols,
    const std::int32_t* column_starts,
    const std::int32_t* row_indices,
    const double* values,
    const double* x,
    double* y) {
  const std::size_t col =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (col >= cols) {
    return;
  }

  const std::int32_t begin = column_starts[col];
  const std::int32_t end = column_starts[col + 1];

  double sum = 0.0;

  for (std::int32_t k = begin; k < end; ++k) {
    const std::int32_t row = row_indices[k];

    sum += values[k] * x[row];
  }

  y[col] = sum;
}

bool check_cuda(cudaError_t status) {
  return status == cudaSuccess;
}

}  // namespace

PdhgCudaContext* pdhg_cuda_create(
    std::size_t rows,
    std::size_t cols,
    std::size_t nnz,
    const std::int32_t* column_starts,
    const std::int32_t* row_indices,
    const double* values) {
  if (column_starts == nullptr ||
      row_indices == nullptr ||
      values == nullptr) {
    return nullptr;
  }

  auto* context = new PdhgCudaContext{};

  context->rows = rows;
  context->cols = cols;
  context->nnz = nnz;

  const std::size_t vector_size =
      rows > cols ? rows : cols;

  if (!check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_column_starts),
          (cols + 1) * sizeof(std::int32_t))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_row_indices),
          nnz * sizeof(std::int32_t))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_values),
          nnz * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_x),
          vector_size * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_y),
          vector_size * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_x_work),
          vector_size * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_y_work),
          vector_size * sizeof(double)))) {
    pdhg_cuda_destroy(context);
    return nullptr;
  }

  if (!check_cuda(cudaMemcpy(
          context->d_column_starts,
          column_starts,
          (cols + 1) * sizeof(std::int32_t),
          cudaMemcpyHostToDevice)) ||
      !check_cuda(cudaMemcpy(
          context->d_row_indices,
          row_indices,
          nnz * sizeof(std::int32_t),
          cudaMemcpyHostToDevice)) ||
      !check_cuda(cudaMemcpy(
          context->d_values,
          values,
          nnz * sizeof(double),
          cudaMemcpyHostToDevice))) {
    pdhg_cuda_destroy(context);
    return nullptr;
  }

  return context;
}

bool pdhg_cuda_upload_x(
    PdhgCudaContext* context,
    const double* host,
    std::size_t size) {
  if (context == nullptr || host == nullptr) {
    return false;
  }

  const std::size_t capacity =
      context->rows > context->cols
          ? context->rows
          : context->cols;

  if (size > capacity) {
    return false;
  }

  return check_cuda(cudaMemcpy(
      context->d_x,
      host,
      size * sizeof(double),
      cudaMemcpyHostToDevice));
}

bool pdhg_cuda_upload_y(
    PdhgCudaContext* context,
    const double* host,
    std::size_t size) {
  if (context == nullptr || host == nullptr) {
    return false;
  }

  const std::size_t capacity =
      context->rows > context->cols
          ? context->rows
          : context->cols;

  if (size > capacity) {
    return false;
  }

  return check_cuda(cudaMemcpy(
      context->d_y,
      host,
      size * sizeof(double),
      cudaMemcpyHostToDevice));
}

bool pdhg_cuda_download_x(
    PdhgCudaContext* context,
    double* host,
    std::size_t size) {
  if (context == nullptr || host == nullptr) {
    return false;
  }

  const std::size_t capacity =
      context->rows > context->cols
          ? context->rows
          : context->cols;

  if (size > capacity) {
    return false;
  }

  return check_cuda(cudaMemcpy(
      host,
      context->d_x,
      size * sizeof(double),
      cudaMemcpyDeviceToHost));
}

bool pdhg_cuda_download_y(
    PdhgCudaContext* context,
    double* host,
    std::size_t size) {
  if (context == nullptr || host == nullptr) {
    return false;
  }

  const std::size_t capacity =
      context->rows > context->cols
          ? context->rows
          : context->cols;

  if (size > capacity) {
    return false;
  }

  // Matrix multiplication results are written to d_y_work.
  return check_cuda(cudaMemcpy(
      host,
      context->d_y_work,
      size * sizeof(double),
      cudaMemcpyDeviceToHost));
}

bool pdhg_cuda_multiply_device(
    PdhgCudaContext* context,
    std::size_t x_size,
    std::size_t y_size) {
  if (context == nullptr ||
      x_size != context->cols ||
      y_size != context->rows) {
    return false;
  }

  // A*x writes into d_y_work.
  if (!check_cuda(cudaMemset(
          context->d_y_work,
          0,
          y_size * sizeof(double)))) {
    return false;
  }

  constexpr int block_size = 256;

  const int grid_size =
      static_cast<int>(
          (context->cols + block_size - 1) /
          block_size);

  csc_multiply_kernel<<<grid_size, block_size>>>(
      context->cols,
      context->d_column_starts,
      context->d_row_indices,
      context->d_values,
      context->d_x,
      context->d_y_work);

  if (!check_cuda(cudaGetLastError())) {
    return false;
  }

  if (!check_cuda(cudaDeviceSynchronize())) {
    return false;
  }

  return true;
}

bool pdhg_cuda_transpose_multiply_device(
    PdhgCudaContext* context,
    std::size_t x_size,
    std::size_t y_size) {
  if (context == nullptr ||
      x_size != context->rows ||
      y_size != context->cols) {
    return false;
  }

  // A^T*x writes into d_y_work.
  if (!check_cuda(cudaMemset(
          context->d_y_work,
          0,
          y_size * sizeof(double)))) {
    return false;
  }

  constexpr int block_size = 256;

  const int grid_size =
      static_cast<int>(
          (context->cols + block_size - 1) /
          block_size);

  csc_transpose_multiply_kernel<<<grid_size, block_size>>>(
      context->cols,
      context->d_column_starts,
      context->d_row_indices,
      context->d_values,
      context->d_x,
      context->d_y_work);

  if (!check_cuda(cudaGetLastError())) {
    return false;
  }

  if (!check_cuda(cudaDeviceSynchronize())) {
    return false;
  }

  return true;
}

double* pdhg_cuda_x(
    PdhgCudaContext* context) noexcept {
  return context != nullptr
             ? context->d_x
             : nullptr;
}

double* pdhg_cuda_y(
    PdhgCudaContext* context) noexcept {
  return context != nullptr
             ? context->d_y
             : nullptr;
}

void pdhg_cuda_destroy(
    PdhgCudaContext* context) noexcept {
  if (context == nullptr) {
    return;
  }

  cudaFree(context->d_column_starts);
  cudaFree(context->d_row_indices);
  cudaFree(context->d_values);

  cudaFree(context->d_x);
  cudaFree(context->d_y);

  cudaFree(context->d_x_work);
  cudaFree(context->d_y_work);

  delete context;
}

}  // namespace sankhya::gpu
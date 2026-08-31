// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CUDA sparse matrix backend.

#include "gpu/pdhg_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sankhya::gpu {

struct PdhgCudaContext {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t nnz = 0;

  // CSC matrix representation (for A^T * y)
  std::int32_t* d_column_starts = nullptr;
  std::int32_t* d_row_indices = nullptr;
  double* d_values = nullptr;

  // CSR matrix representation (for A * x_bar and A * dx without atomics)
  std::int32_t* d_row_starts = nullptr;
  std::int32_t* d_col_indices = nullptr;
  double* d_csr_values = nullptr;

  // Static problem data
  double* d_cost = nullptr;
  double* d_col_lower = nullptr;
  double* d_col_upper = nullptr;
  double* d_row_lower = nullptr;
  double* d_row_upper = nullptr;

  // GPU-resident iterate vectors
  double* d_x = nullptr;
  double* d_y = nullptr;
  double* d_x_next = nullptr;
  double* d_y_next = nullptr;
  double* d_x_bar = nullptr;
  double* d_dx = nullptr;
  double* d_dy = nullptr;

  // Running sum vectors for ergodic averaging
  double* d_x_sum = nullptr;
  double* d_y_sum = nullptr;

  // Scalar reduction buffers:
  // [0] = ||dx||^2, [1] = ||dy||^2, [2] = dy^T (A * dx)
  double* d_scalars = nullptr;
  double* h_scalars = nullptr;
};

namespace {

constexpr int kBlockSize = 256;

bool check_cuda(cudaError_t status) {
  return status == cudaSuccess;
}

__device__ inline double project_device(
    double value,
    double lower,
    double upper) {
  if (isfinite(lower) && fabs(lower) < 1e30 && value < lower) return lower;
  if (isfinite(upper) && fabs(upper) < 1e30 && value > upper) return upper;
  return value;
}

// Reset reduction scalars on device before fused steps
__global__ void zero_scalars_kernel(double* scalars) {
  if (threadIdx.x == 0) {
    scalars[0] = 0.0;
    scalars[1] = 0.0;
    scalars[2] = 0.0;
  }
}

// Fused kernel 1: A^T * y + primal update + dx + x_bar + ||dx||^2 reduction
__global__ void primal_fused_kernel(
    std::size_t cols,
    double tau,
    const std::int32_t* column_starts,
    const std::int32_t* row_indices,
    const double* values,
    const double* cost,
    const double* col_lower,
    const double* col_upper,
    const double* x,
    const double* y,
    double* x_next,
    double* dx,
    double* x_bar,
    double* scalars) {
  __shared__ double sdata[kBlockSize];
  const unsigned int tid = threadIdx.x;
  const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  double local_diff_sq = 0.0;

  if (j < cols) {
    const std::int32_t begin = column_starts[j];
    const std::int32_t end = column_starts[j + 1];

    double at_y = 0.0;
    for (std::int32_t k = begin; k < end; ++k) {
      at_y += values[k] * y[row_indices[k]];
    }

    const double grad = cost[j] + at_y;
    const double cur_x = x[j];
    const double next_x = project_device(cur_x - tau * grad, col_lower[j], col_upper[j]);

    x_next[j] = next_x;
    const double diff = next_x - cur_x;
    dx[j] = diff;
    x_bar[j] = 2.0 * next_x - cur_x;

    local_diff_sq = diff * diff;
  }

  sdata[tid] = local_diff_sq;
  __syncthreads();

  for (unsigned int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0 && sdata[0] != 0.0) {
    atomicAdd(&scalars[0], sdata[0]);
  }
}

// Fused kernel 2: A * x_bar via CSR + dual Moreau prox update + dy + ||dy||^2 reduction
__global__ void dual_fused_kernel(
    std::size_t rows,
    double sigma,
    const std::int32_t* row_starts,
    const std::int32_t* col_indices,
    const double* csr_values,
    const double* row_lower,
    const double* row_upper,
    const double* y,
    const double* x_bar,
    double* y_next,
    double* dy,
    double* scalars) {
  __shared__ double sdata[kBlockSize];
  const unsigned int tid = threadIdx.x;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  double local_diff_sq = 0.0;

  if (i < rows) {
    const std::int32_t begin = row_starts[i];
    const std::int32_t end = row_starts[i + 1];

    double ax_bar = 0.0;
    for (std::int32_t k = begin; k < end; ++k) {
      ax_bar += csr_values[k] * x_bar[col_indices[k]];
    }

    const double cur_y = y[i];
    const double v = cur_y + sigma * ax_bar;
    const double next_y = v - sigma * project_device(v / sigma, row_lower[i], row_upper[i]);

    y_next[i] = next_y;
    const double diff = next_y - cur_y;
    dy[i] = diff;

    local_diff_sq = diff * diff;
  }

  sdata[tid] = local_diff_sq;
  __syncthreads();

  for (unsigned int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0 && sdata[0] != 0.0) {
    atomicAdd(&scalars[1], sdata[0]);
  }
}

// Fused kernel 3: A * dx via CSR + dy^T (A * dx) reduction
__global__ void interaction_fused_kernel(
    std::size_t rows,
    const std::int32_t* row_starts,
    const std::int32_t* col_indices,
    const double* csr_values,
    const double* dx,
    const double* dy,
    double* scalars) {
  __shared__ double sdata[kBlockSize];
  const unsigned int tid = threadIdx.x;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  double local_interaction = 0.0;

  if (i < rows) {
    const std::int32_t begin = row_starts[i];
    const std::int32_t end = row_starts[i + 1];

    double adx = 0.0;
    for (std::int32_t k = begin; k < end; ++k) {
      adx += csr_values[k] * dx[col_indices[k]];
    }

    local_interaction = dy[i] * adx;
  }

  sdata[tid] = local_interaction;
  __syncthreads();

  for (unsigned int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0 && sdata[0] != 0.0) {
    atomicAdd(&scalars[2], sdata[0]);
  }
}

// Fused kernel 4: Commit candidate (x <- x_next, y <- y_next) & accumulate running sums
__global__ void accept_and_accumulate_fused_kernel(
    std::size_t cols,
    std::size_t rows,
    const double* x_next,
    const double* y_next,
    double* x,
    double* y,
    double* x_sum,
    double* y_sum) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < cols) {
    const double vx = x_next[idx];
    x[idx] = vx;
    if (x_sum != nullptr) {
      x_sum[idx] += vx;
    }
  }

  if (idx < rows) {
    const double vy = y_next[idx];
    y[idx] = vy;
    if (y_sum != nullptr) {
      y_sum[idx] += vy;
    }
  }
}

// Download running average: x_avg = x_sum / count, y_avg = y_sum / count
__global__ void compute_average_kernel(
    std::size_t cols,
    std::size_t rows,
    double inv_count,
    const double* x_sum,
    const double* y_sum,
    double* x_avg_out,
    double* y_avg_out) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < cols && x_sum != nullptr && x_avg_out != nullptr) {
    x_avg_out[idx] = x_sum[idx] * inv_count;
  }

  if (idx < rows && y_sum != nullptr && y_avg_out != nullptr) {
    y_avg_out[idx] = y_sum[idx] * inv_count;
  }
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
      (cols > 0 && row_indices == nullptr && nnz > 0) ||
      (cols > 0 && values == nullptr && nnz > 0)) {
    return nullptr;
  }

  auto* context = new PdhgCudaContext{};
  context->rows = rows;
  context->cols = cols;
  context->nnz = nnz;

  // 1. Build CSR representation on host from input CSC matrix
  std::vector<std::int32_t> h_row_starts(rows + 1, 0);
  std::vector<std::int32_t> h_col_indices(nnz, 0);
  std::vector<double> h_csr_values(nnz, 0.0);

  if (rows > 0 && cols > 0 && nnz > 0) {
    for (std::size_t k = 0; k < nnz; ++k) {
      const auto r = static_cast<std::size_t>(row_indices[k]);
      if (r < rows) {
        h_row_starts[r + 1]++;
      }
    }

    for (std::size_t i = 0; i < rows; ++i) {
      h_row_starts[i + 1] += h_row_starts[i];
    }

    std::vector<std::int32_t> next_pos = h_row_starts;
    for (std::size_t j = 0; j < cols; ++j) {
      const std::int32_t begin = column_starts[j];
      const std::int32_t end = column_starts[j + 1];
      for (std::int32_t k = begin; k < end; ++k) {
        const auto r = static_cast<std::size_t>(row_indices[k]);
        const std::int32_t p = next_pos[r]++;
        h_col_indices[static_cast<std::size_t>(p)] = static_cast<std::int32_t>(j);
        h_csr_values[static_cast<std::size_t>(p)] = values[k];
      }
    }
  }

  // 2. Allocate all GPU device memory
  if (!check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_column_starts),
          (cols + 1) * sizeof(std::int32_t))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_row_indices),
          std::max<std::size_t>(nnz, 1) * sizeof(std::int32_t))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_values),
          std::max<std::size_t>(nnz, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_row_starts),
          (rows + 1) * sizeof(std::int32_t))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_col_indices),
          std::max<std::size_t>(nnz, 1) * sizeof(std::int32_t))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_csr_values),
          std::max<std::size_t>(nnz, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_cost),
          std::max<std::size_t>(cols, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_col_lower),
          std::max<std::size_t>(cols, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_col_upper),
          std::max<std::size_t>(cols, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_row_lower),
          std::max<std::size_t>(rows, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_row_upper),
          std::max<std::size_t>(rows, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_x),
          std::max<std::size_t>(cols, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_y),
          std::max<std::size_t>(rows, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_x_next),
          std::max<std::size_t>(cols, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_y_next),
          std::max<std::size_t>(rows, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_x_bar),
          std::max<std::size_t>(cols, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_dx),
          std::max<std::size_t>(cols, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_dy),
          std::max<std::size_t>(rows, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_x_sum),
          std::max<std::size_t>(cols, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_y_sum),
          std::max<std::size_t>(rows, 1) * sizeof(double))) ||
      !check_cuda(cudaMalloc(
          reinterpret_cast<void**>(&context->d_scalars),
          3 * sizeof(double))) ||
      !check_cuda(cudaHostAlloc(
          reinterpret_cast<void**>(&context->h_scalars),
          3 * sizeof(double),
          cudaHostAllocDefault))) {
    pdhg_cuda_destroy(context);
    return nullptr;
  }

  // 3. Upload CSC & CSR matrices to GPU
  if (!check_cuda(cudaMemcpy(
          context->d_column_starts,
          column_starts,
          (cols + 1) * sizeof(std::int32_t),
          cudaMemcpyHostToDevice)) ||
      (nnz > 0 && !check_cuda(cudaMemcpy(
                      context->d_row_indices,
                      row_indices,
                      nnz * sizeof(std::int32_t),
                      cudaMemcpyHostToDevice))) ||
      (nnz > 0 && !check_cuda(cudaMemcpy(
                      context->d_values,
                      values,
                      nnz * sizeof(double),
                      cudaMemcpyHostToDevice))) ||
      !check_cuda(cudaMemcpy(
          context->d_row_starts,
          h_row_starts.data(),
          (rows + 1) * sizeof(std::int32_t),
          cudaMemcpyHostToDevice)) ||
      (nnz > 0 && !check_cuda(cudaMemcpy(
                      context->d_col_indices,
                      h_col_indices.data(),
                      nnz * sizeof(std::int32_t),
                      cudaMemcpyHostToDevice))) ||
      (nnz > 0 && !check_cuda(cudaMemcpy(
                      context->d_csr_values,
                      h_csr_values.data(),
                      nnz * sizeof(double),
                      cudaMemcpyHostToDevice)))) {
    pdhg_cuda_destroy(context);
    return nullptr;
  }

  // Initialize running sums to 0
  if (cols > 0) {
    cudaMemset(context->d_x_sum, 0, cols * sizeof(double));
  }
  if (rows > 0) {
    cudaMemset(context->d_y_sum, 0, rows * sizeof(double));
  }

  return context;
}

bool pdhg_cuda_upload_problem_data(
    PdhgCudaContext* context,
    const double* cost,
    const double* col_lower,
    const double* col_upper,
    const double* row_lower,
    const double* row_upper) {
  if (context == nullptr || cost == nullptr || col_lower == nullptr ||
      col_upper == nullptr || row_lower == nullptr || row_upper == nullptr) {
    return false;
  }

  if (context->cols > 0) {
    if (!check_cuda(cudaMemcpy(
            context->d_cost,
            cost,
            context->cols * sizeof(double),
            cudaMemcpyHostToDevice)) ||
        !check_cuda(cudaMemcpy(
            context->d_col_lower,
            col_lower,
            context->cols * sizeof(double),
            cudaMemcpyHostToDevice)) ||
        !check_cuda(cudaMemcpy(
            context->d_col_upper,
            col_upper,
            context->cols * sizeof(double),
            cudaMemcpyHostToDevice))) {
      return false;
    }
  }

  if (context->rows > 0) {
    if (!check_cuda(cudaMemcpy(
            context->d_row_lower,
            row_lower,
            context->rows * sizeof(double),
            cudaMemcpyHostToDevice)) ||
        !check_cuda(cudaMemcpy(
            context->d_row_upper,
            row_upper,
            context->rows * sizeof(double),
            cudaMemcpyHostToDevice))) {
      return false;
    }
  }

  return true;
}

bool pdhg_cuda_upload_x(
    PdhgCudaContext* context,
    const double* host,
    std::size_t size) {
  if (context == nullptr || host == nullptr || size != context->cols) {
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
  if (context == nullptr || host == nullptr || size != context->rows) {
    return false;
  }

  return check_cuda(cudaMemcpy(
      context->d_y,
      host,
      size * sizeof(double),
      cudaMemcpyHostToDevice));
}

bool pdhg_cuda_step(
    PdhgCudaContext* context,
    double tau,
    double sigma,
    double omega,
    double* movement,
    double* interaction) {
  if (context == nullptr || movement == nullptr || interaction == nullptr) {
    return false;
  }

  const int col_grid =
      static_cast<int>(
          (context->cols + kBlockSize - 1) /
          kBlockSize);

  const int row_grid =
      static_cast<int>(
          (context->rows + kBlockSize - 1) /
          kBlockSize);

  // 1. Asynchronously reset reduction scalar accumulators on device
  zero_scalars_kernel<<<1, 1>>>(context->d_scalars);

  // 2. Fused primal step (A^T * y + primal update + dx + x_bar + ||dx||^2 reduction)
  if (context->cols > 0) {
    primal_fused_kernel<<<col_grid, kBlockSize>>>(
        context->cols,
        tau,
        context->d_column_starts,
        context->d_row_indices,
        context->d_values,
        context->d_cost,
        context->d_col_lower,
        context->d_col_upper,
        context->d_x,
        context->d_y,
        context->d_x_next,
        context->d_dx,
        context->d_x_bar,
        context->d_scalars);
  }

  // 3. Fused dual step (A * x_bar via CSR + dual prox update + dy + ||dy||^2 reduction)
  if (context->rows > 0) {
    dual_fused_kernel<<<row_grid, kBlockSize>>>(
        context->rows,
        sigma,
        context->d_row_starts,
        context->d_col_indices,
        context->d_csr_values,
        context->d_row_lower,
        context->d_row_upper,
        context->d_y,
        context->d_x_bar,
        context->d_y_next,
        context->d_dy,
        context->d_scalars);

    // 4. Fused interaction step (A * dx via CSR + dy^T (A*dx) reduction)
    interaction_fused_kernel<<<row_grid, kBlockSize>>>(
        context->rows,
        context->d_row_starts,
        context->d_col_indices,
        context->d_csr_values,
        context->d_dx,
        context->d_dy,
        context->d_scalars);
  }

  // 5. Transfer only 3 scalar values to pinned host memory (one synchronous DMA)
  if (!check_cuda(cudaMemcpy(
          context->h_scalars,
          context->d_scalars,
          3 * sizeof(double),
          cudaMemcpyDeviceToHost))) {
    return false;
  }

  const double dx_norm_sq = context->h_scalars[0];
  const double dy_norm_sq = context->h_scalars[1];
  const double dy_dot_adx = context->h_scalars[2];

  *movement = 0.5 * omega * dx_norm_sq + 0.5 * dy_norm_sq / omega;
  *interaction = std::fabs(dy_dot_adx);

  return true;
}

bool pdhg_cuda_accept_step(
    PdhgCudaContext* context) {
  if (context == nullptr) {
    return false;
  }

  const std::size_t max_dim =
      context->cols > context->rows
          ? context->cols
          : context->rows;

  if (max_dim == 0) {
    return true;
  }

  const int grid =
      static_cast<int>(
          (max_dim + kBlockSize - 1) /
          kBlockSize);

  accept_and_accumulate_fused_kernel<<<grid, kBlockSize>>>(
      context->cols,
      context->rows,
      context->d_x_next,
      context->d_y_next,
      context->d_x,
      context->d_y,
      context->d_x_sum,
      context->d_y_sum);

  return true;
}

bool pdhg_cuda_download_average(
    PdhgCudaContext* context,
    double* host_x_avg,
    double* host_y_avg,
    std::size_t count) {
  if (context == nullptr || host_x_avg == nullptr || host_y_avg == nullptr || count == 0) {
    return false;
  }

  const std::size_t max_dim =
      context->cols > context->rows
          ? context->cols
          : context->rows;

  if (max_dim > 0) {
    const int grid =
        static_cast<int>(
            (max_dim + kBlockSize - 1) /
            kBlockSize);

    const double inv_count = 1.0 / static_cast<double>(count);

    // Reuse d_x_bar and d_dx as temporary output buffers on device for average
    compute_average_kernel<<<grid, kBlockSize>>>(
        context->cols,
        context->rows,
        inv_count,
        context->d_x_sum,
        context->d_y_sum,
        context->d_x_bar,
        context->d_dx);
  }

  if (context->cols > 0) {
    if (!check_cuda(cudaMemcpy(
            host_x_avg,
            context->d_x_bar,
            context->cols * sizeof(double),
            cudaMemcpyDeviceToHost))) {
      return false;
    }
  }

  if (context->rows > 0) {
    if (!check_cuda(cudaMemcpy(
            host_y_avg,
            context->d_dx,
            context->rows * sizeof(double),
            cudaMemcpyDeviceToHost))) {
      return false;
    }
  }

  return true;
}

bool pdhg_cuda_reset_sum(
    PdhgCudaContext* context) {
  if (context == nullptr) {
    return false;
  }

  if (context->cols > 0) {
    if (!check_cuda(cudaMemset(
            context->d_x_sum,
            0,
            context->cols * sizeof(double)))) {
      return false;
    }
  }

  if (context->rows > 0) {
    if (!check_cuda(cudaMemset(
            context->d_y_sum,
            0,
            context->rows * sizeof(double)))) {
      return false;
    }
  }

  return true;
}

bool pdhg_cuda_download_x(
    PdhgCudaContext* context,
    double* host,
    std::size_t size) {
  if (context == nullptr || host == nullptr || size != context->cols) {
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
  if (context == nullptr || host == nullptr || size != context->rows) {
    return false;
  }

  return check_cuda(cudaMemcpy(
      host,
      context->d_y,
      size * sizeof(double),
      cudaMemcpyDeviceToHost));
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

  if (context->h_scalars != nullptr) {
    cudaFreeHost(context->h_scalars);
  }

  cudaFree(context->d_column_starts);
  cudaFree(context->d_row_indices);
  cudaFree(context->d_values);

  cudaFree(context->d_row_starts);
  cudaFree(context->d_col_indices);
  cudaFree(context->d_csr_values);

  cudaFree(context->d_cost);
  cudaFree(context->d_col_lower);
  cudaFree(context->d_col_upper);
  cudaFree(context->d_row_lower);
  cudaFree(context->d_row_upper);

  cudaFree(context->d_x);
  cudaFree(context->d_y);
  cudaFree(context->d_x_next);
  cudaFree(context->d_y_next);
  cudaFree(context->d_x_bar);
  cudaFree(context->d_dx);
  cudaFree(context->d_dy);

  cudaFree(context->d_x_sum);
  cudaFree(context->d_y_sum);
  cudaFree(context->d_scalars);

  delete context;
}

}  // namespace sankhya::gpu
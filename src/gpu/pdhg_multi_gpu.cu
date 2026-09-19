// SPDX-License-Identifier: Apache-2.0
// SANKHYA - multi-GPU restarted PDHG for LP (#295).
//
// References (derived from the papers; no solver source consulted):
//   [CP11]   Chambolle & Pock, JMIV 40(1) 2011.
//   [PDLP]   Applegate et al., NeurIPS 2021.
//   [cuPDLP] Lu & Yang, arXiv:2311.12180.
//
// Partitioning scheme:
//   - Rows of A are split into K disjoint blocks: device k owns rows [r_k, r_{k+1}).
//   - Primal vector x (n-element) and problem constants are REPLICATED on every device.
//   - Dual vector y (m-element) is PARTITIONED: device k holds y[r_k..r_{k+1}).
//
// Per-iteration communication (host-mediated):
//   1. A^T * y_k  → partial_aty_k (n-vector) — download, sum on CPU, upload to all.
//   2. Two scalars: movement_y_k, interaction_k — download from each, sum on CPU.
//
// Fallback: if K==1 or device setup fails, delegates to solve_pdhg_gpu().

#include "pdhg_multi_gpu.hpp"

#include "../pdhg/pdhg_evaluate.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// fmt/format.h uses Unicode string literals that nvcc cannot parse; use snprintf instead.
#include <cstdio>
#include <string>

#include "../core/resource_limits.hpp"
#include "../core/stop_controller.hpp"
#include "../la/scaling.hpp"
#include "device.hpp"
#include "multi_device.hpp"
#include "pdhg_gpu.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya::gpu {
namespace {

constexpr Count kEvaluationInterval = 40;
constexpr int kBlockSize = 256;

// ---- Error-check macros --------------------------------------------------
#define CUDA_CHECK(expr)                     \
  do {                                       \
    if ((expr) != cudaSuccess) return false; \
  } while (0)

#define CS_CHECK(expr)                                   \
  do {                                                   \
    if ((expr) != CUSPARSE_STATUS_SUCCESS) return false; \
  } while (0)

static bool launch_ok() { return cudaPeekAtLastError() == cudaSuccess; }

static const double kOne = 1.0;
static const double kZero = 0.0;

// ---- CUDA kernels (identical math to pdhg_gpu.cu; separate TU) ----------

__global__ void k_primal_fused_mg(const double* __restrict__ x, const double* __restrict__ at_y,
                                   const double* __restrict__ cost,
                                   const double* __restrict__ col_lo,
                                   const double* __restrict__ col_hi, double* __restrict__ x_next,
                                   double* __restrict__ extrapolated, double* __restrict__ dx,
                                   double* d_mv_x, double tau, double omega, int n) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  double mv_thread = 0.0;
  if (j < n) {
    double xnj = x[j] - tau * (cost[j] + at_y[j]);
    if (!isinf(col_lo[j]) && xnj < col_lo[j]) xnj = col_lo[j];
    if (!isinf(col_hi[j]) && xnj > col_hi[j]) xnj = col_hi[j];
    x_next[j] = xnj;
    const double dxj = xnj - x[j];
    extrapolated[j] = 2.0 * xnj - x[j];
    dx[j] = dxj;
    mv_thread = 0.5 * omega * dxj * dxj;
  }
  const double bsum = BlockReduce(temp).Sum(mv_thread);
  if (threadIdx.x == 0) atomicAdd(d_mv_x, bsum);
}

__global__ void k_dual_fused_mg(const double* __restrict__ y, const double* __restrict__ a_x,
                                 const double* __restrict__ row_lo,
                                 const double* __restrict__ row_hi, double* __restrict__ y_next,
                                 double* __restrict__ dy, double* d_mv_y, double sigma,
                                 double omega, int m) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double mv_thread = 0.0;
  if (i < m) {
    const double v = y[i] + sigma * a_x[i];
    double pv = v / sigma;
    if (!isinf(row_lo[i]) && pv < row_lo[i]) pv = row_lo[i];
    if (!isinf(row_hi[i]) && pv > row_hi[i]) pv = row_hi[i];
    const double yni = v - sigma * pv;
    y_next[i] = yni;
    const double dyi = yni - y[i];
    dy[i] = dyi;
    mv_thread = 0.5 * dyi * dyi / omega;
  }
  const double bsum = BlockReduce(temp).Sum(mv_thread);
  if (threadIdx.x == 0) atomicAdd(d_mv_y, bsum);
}

__global__ void k_interaction_fused_mg(const double* __restrict__ dy,
                                        const double* __restrict__ adx, double* d_interaction,
                                        int m) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const double val = (i < m) ? dy[i] * adx[i] : 0.0;
  const double bsum = BlockReduce(temp).Sum(val);
  if (threadIdx.x == 0) atomicAdd(d_interaction, bsum);
}

__global__ void k_accum_n_mg(const double* __restrict__ v, double* __restrict__ s, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) s[j] += v[j];
}

__global__ void k_accum_m_mg(const double* __restrict__ v, double* __restrict__ s, int m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) s[i] += v[i];
}

// ---- Device-memory helpers -----------------------------------------------

static double* dev_zeros_mg(std::size_t count) {
  if (count == 0) return nullptr;
  double* p = nullptr;
  if (cudaMalloc(&p, count * sizeof(double)) != cudaSuccess) return nullptr;
  if (cudaMemset(p, 0, count * sizeof(double)) != cudaSuccess) {
    cudaFree(p);
    return nullptr;
  }
  return p;
}

static int* dev_int_mg(std::size_t count) {
  if (count == 0) return nullptr;
  int* p = nullptr;
  if (cudaMalloc(&p, count * sizeof(int)) != cudaSuccess) return nullptr;
  return p;
}

template <typename T>
static bool hd_copy_mg(const T* host, T* dev, std::size_t count) {
  if (count == 0) return true;
  return cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

static bool dh_copy_mg(const double* dev, double* host, std::size_t count) {
  if (count == 0) return true;
  return cudaMemcpy(host, dev, count * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
}

// ---- Per-device state ---------------------------------------------------

struct DeviceState {
  // Full primal vectors (n-length, REPLICATED on every device)
  double *d_x{}, *d_xn{}, *d_ext{}, *d_dx{}, *d_aty{}, *d_xsum{};
  // Scalar accumulators: [0]=mv_x, [1]=mv_y_partial, [2]=interaction_partial
  double* d_scalars{};
  // Problem constants (n-length, replicated)
  double *d_cost{}, *d_clo{}, *d_chi{};
  // Local dual vectors (local_m-length, PARTITIONED)
  double *d_y{}, *d_yn{}, *d_dy{}, *d_ax{}, *d_adx{}, *d_ysum{};
  // Local row bounds (local_m-length)
  double *d_rlo{}, *d_rhi{};
  // Local CSR row slice
  int *d_rowptr{}, *d_colidx{};
  double* d_vals{};
  // cuSPARSE
  void* d_spmv{};
  std::size_t spmv_bytes{};
  cusparseHandle_t cs{};
  cusparseSpMatDescr_t mat{};
  cusparseDnVecDescr_t vn{};  // n-element
  cusparseDnVecDescr_t vm{};  // local_m-element

  int device_id = -1;
  int n{}, local_m{}, local_nnz{};
  int row_start{};

  DeviceState() = default;
  DeviceState(const DeviceState&) = delete;
  DeviceState& operator=(const DeviceState&) = delete;

  ~DeviceState() {
    if (device_id < 0) return;
    cudaSetDevice(device_id);
    if (vm) cusparseDestroyDnVec(vm);
    if (vn) cusparseDestroyDnVec(vn);
    if (mat) cusparseDestroySpMat(mat);
    if (cs) cusparseDestroy(cs);
    cudaFree(d_x);    cudaFree(d_xn);   cudaFree(d_ext);  cudaFree(d_dx);
    cudaFree(d_aty);  cudaFree(d_xsum); cudaFree(d_scalars);
    cudaFree(d_cost); cudaFree(d_clo);  cudaFree(d_chi);
    cudaFree(d_y);    cudaFree(d_yn);   cudaFree(d_dy);
    cudaFree(d_ax);   cudaFree(d_adx);  cudaFree(d_ysum);
    cudaFree(d_rlo);  cudaFree(d_rhi);
    cudaFree(d_rowptr); cudaFree(d_colidx); cudaFree(d_vals);
    cudaFree(d_spmv);
  }

  [[nodiscard]] bool alloc_ok() const {
    return d_x && d_xn && d_ext && d_dx && d_aty && d_xsum && d_scalars && d_cost && d_clo &&
           d_chi;
  }
};

// ---- cuSPARSE SpMV on DeviceState ----------------------------------------
// Uses local_m for the row dimension.

static bool spmv_nt_mg(DeviceState& d, double* d_in_n, double* d_out_m) {
  if (d.local_m == 0 || d.local_nnz == 0) {
    if (d.local_m > 0)
      cudaMemset(d_out_m, 0, static_cast<std::size_t>(d.local_m) * sizeof(double));
    return true;
  }
  CS_CHECK(cusparseDnVecSetValues(d.vn, d_in_n));
  CS_CHECK(cusparseDnVecSetValues(d.vm, d_out_m));
  CS_CHECK(cusparseSpMV(d.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, d.mat, d.vn, &kZero,
                        d.vm, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d.d_spmv));
  return true;
}

static bool spmv_t_mg(DeviceState& d, double* d_in_m, double* d_out_n) {
  if (d.local_m == 0 || d.local_nnz == 0) {
    if (d.n > 0)
      cudaMemset(d_out_n, 0, static_cast<std::size_t>(d.n) * sizeof(double));
    return true;
  }
  CS_CHECK(cusparseDnVecSetValues(d.vm, d_in_m));
  CS_CHECK(cusparseDnVecSetValues(d.vn, d_out_n));
  CS_CHECK(cusparseSpMV(d.cs, CUSPARSE_OPERATION_TRANSPOSE, &kOne, d.mat, d.vm, &kZero, d.vn,
                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d.d_spmv));
  return true;
}

using pdhg::euclidean_norm;
using pdhg::evaluate;
using pdhg::Problem;
using pdhg::Residuals;

// ---- Device setup --------------------------------------------------------
// Build a local CSR slice for rows [row_start, row_end) from the full scaled CsrView.

static bool setup_device(DeviceState& d, const CsrView& full_csr, const Scaling& scaling,
                          const std::vector<double>& x0_scaled, int row_start, int row_end,
                          int n_cols) {
  cudaSetDevice(d.device_id);

  const int lm = row_end - row_start;
  d.n = n_cols;
  d.local_m = lm;
  d.row_start = row_start;

  // Allocate primal vectors (full n, replicated)
  const std::size_t n = static_cast<std::size_t>(n_cols);
  d.d_x = dev_zeros_mg(n);
  d.d_xn = dev_zeros_mg(n);
  d.d_ext = dev_zeros_mg(n);
  d.d_dx = dev_zeros_mg(n);
  d.d_aty = dev_zeros_mg(n);
  d.d_xsum = dev_zeros_mg(n);
  d.d_scalars = dev_zeros_mg(3);
  d.d_cost = dev_zeros_mg(n);
  d.d_clo = dev_zeros_mg(n);
  d.d_chi = dev_zeros_mg(n);
  if (!d.alloc_ok()) return false;

  // Upload problem constants
  if (!hd_copy_mg(scaling.cost.data(), d.d_cost, n) ||
      !hd_copy_mg(scaling.col_lower.data(), d.d_clo, n) ||
      !hd_copy_mg(scaling.col_upper.data(), d.d_chi, n) ||
      !hd_copy_mg(x0_scaled.data(), d.d_x, n))
    return false;

  // Allocate local dual vectors
  const std::size_t lm_sz = static_cast<std::size_t>(lm);
  if (lm > 0) {
    d.d_y = dev_zeros_mg(lm_sz);
    d.d_yn = dev_zeros_mg(lm_sz);
    d.d_dy = dev_zeros_mg(lm_sz);
    d.d_ax = dev_zeros_mg(lm_sz);
    d.d_adx = dev_zeros_mg(lm_sz);
    d.d_ysum = dev_zeros_mg(lm_sz);
    d.d_rlo = dev_zeros_mg(lm_sz);
    d.d_rhi = dev_zeros_mg(lm_sz);
    if (!d.d_y || !d.d_yn || !d.d_dy || !d.d_ax || !d.d_adx || !d.d_ysum || !d.d_rlo ||
        !d.d_rhi)
      return false;
    if (!hd_copy_mg(scaling.row_lower.data() + row_start, d.d_rlo, lm_sz) ||
        !hd_copy_mg(scaling.row_upper.data() + row_start, d.d_rhi, lm_sz))
      return false;
  }

  // Build local CSR slice: rows [row_start, row_end) of full_csr
  const auto& full_rp = full_csr.row_starts();
  const auto& full_ci = full_csr.column_indices();
  const auto& full_cv = full_csr.values();

  const int nnz_start = full_rp[static_cast<std::size_t>(row_start)];
  const int nnz_end =
      (row_end <= static_cast<int>(full_rp.size()) - 1)
          ? full_rp[static_cast<std::size_t>(row_end)]
          : static_cast<int>(full_ci.size());
  d.local_nnz = nnz_end - nnz_start;

  // Build local row-pointer array (re-based to local coordinates)
  std::vector<int> local_rp(static_cast<std::size_t>(lm + 1));
  for (int i = 0; i <= lm; ++i)
    local_rp[static_cast<std::size_t>(i)] =
        full_rp[static_cast<std::size_t>(row_start + i)] - nnz_start;

  d.d_rowptr = dev_int_mg(static_cast<std::size_t>(lm + 1));
  if (!d.d_rowptr) return false;
  if (!hd_copy_mg(local_rp.data(), d.d_rowptr, static_cast<std::size_t>(lm + 1))) return false;

  if (d.local_nnz > 0) {
    const std::size_t lnnz = static_cast<std::size_t>(d.local_nnz);
    d.d_colidx = dev_int_mg(lnnz);
    d.d_vals = dev_zeros_mg(lnnz);
    if (!d.d_colidx || !d.d_vals) return false;
    if (!hd_copy_mg(full_ci.data() + nnz_start, d.d_colidx, lnnz) ||
        !hd_copy_mg(full_cv.data() + nnz_start, d.d_vals, lnnz))
      return false;
  }

  // cuSPARSE
  if (cusparseCreate(&d.cs) != CUSPARSE_STATUS_SUCCESS) return false;
  if (cusparseCreateCsr(&d.mat, static_cast<int64_t>(lm), static_cast<int64_t>(n_cols),
                        static_cast<int64_t>(d.local_nnz), d.d_rowptr, d.d_colidx, d.d_vals,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
    return false;

  void* vn_init = d.d_x ? d.d_x : d.d_scalars;
  void* vm_init = (lm > 0 && d.d_y) ? d.d_y : d.d_scalars;
  if (cusparseCreateDnVec(&d.vn, static_cast<int64_t>(n_cols), vn_init, CUDA_R_64F) !=
      CUSPARSE_STATUS_SUCCESS)
    return false;
  if (lm > 0) {
    if (cusparseCreateDnVec(&d.vm, static_cast<int64_t>(lm), vm_init, CUDA_R_64F) !=
        CUSPARSE_STATUS_SUCCESS)
      return false;
  }

  // SpMV buffer
  if (lm > 0 && d.local_nnz > 0) {
    std::size_t bytes_nt = 0, bytes_t = 0;
    cusparseDnVecSetValues(d.vn, d.d_ext);
    cusparseDnVecSetValues(d.vm, d.d_ax);
    if (cusparseSpMV_bufferSize(d.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, d.mat, d.vn,
                                &kZero, d.vm, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                &bytes_nt) != CUSPARSE_STATUS_SUCCESS)
      return false;
    cusparseDnVecSetValues(d.vm, d.d_y);
    cusparseDnVecSetValues(d.vn, d.d_aty);
    if (cusparseSpMV_bufferSize(d.cs, CUSPARSE_OPERATION_TRANSPOSE, &kOne, d.mat, d.vm, &kZero,
                                d.vn, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                &bytes_t) != CUSPARSE_STATUS_SUCCESS)
      return false;
    d.spmv_bytes = std::max(bytes_nt, bytes_t);
    if (d.spmv_bytes > 0 && cudaMalloc(&d.d_spmv, d.spmv_bytes) != cudaSuccess) return false;
  }
  return true;
}

}  // anonymous namespace

// ---- Public entry point --------------------------------------------------

Solution solve_pdhg_multi_gpu(const Model& model, const Options& options,
                               const std::vector<int>& device_ids, Logger& logger,
                               SolveControl* control) {
  // Single-device: delegate directly to the single-GPU solver.
  if (device_ids.size() <= 1) {
    return solve_pdhg_gpu(model, options, logger, control);
  }

  // Validate each requested device exists and meets architecture requirements.
  const int total_devices = device_count();
  for (int id : device_ids) {
    if (id >= total_devices) {
      logger.warning(
          "Multi-GPU PDHG: device {} requested but only {} devices are present; "
          "falling back to single-GPU",
          id, total_devices);
      return solve_pdhg_gpu(model, options, logger, control);
    }
  }

  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "pdhg-cuda-multi";

  const std::string validation = model.validate();
  if (!validation.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = validation;
    return solution;
  }

  const Index rows = model.num_rows();
  const Index cols = model.num_cols();
  const double sense = model.sense_multiplier();
  const auto n = static_cast<std::size_t>(cols);
  const auto m = static_cast<std::size_t>(rows);
  const int ni = static_cast<int>(n);
  const int mi = static_cast<int>(m);
  const int K = static_cast<int>(device_ids.size());

  // ---- Unscaled problem --------------------------------------------------
  Problem prob;
  prob.model = &model;
  prob.cost.resize(n);
  for (Index j = 0; j < cols; ++j)
    prob.cost[static_cast<std::size_t>(j)] =
        sense * model.col_cost[static_cast<std::size_t>(j)];
  prob.cost_norm = euclidean_norm(prob.cost);
  {
    double bsq = 0.0;
    for (Index i = 0; i < rows; ++i) {
      const auto u = static_cast<std::size_t>(i);
      const double b = is_finite_bound(model.row_lower[u])
                           ? model.row_lower[u]
                           : (is_finite_bound(model.row_upper[u]) ? model.row_upper[u] : 0.0);
      bsq += b * b;
    }
    prob.bound_norm = std::sqrt(bsq);
  }

  // ---- Precondition (CPU, one-off) ---------------------------------------
  constexpr int kRuizIter = 10;
  constexpr int kPowerIter = 30;
  const Scaling scaling = build_scaling(model, prob.cost, kRuizIter);
  const double spectral_norm = estimate_spectral_norm(
      scaling.matrix, kPowerIter, static_cast<unsigned>(options.get_int("random_seed")) + 1u);

  // ---- Build CSR on CPU --------------------------------------------------
  const CsrView full_csr(scaling.matrix);

  // ---- Solver parameters -------------------------------------------------
  const double tolerance = options.get_double("pdhg_tolerance");
  const ResourceLimits limits(options, logger);
  const Count iteration_limit =
      limits.iteration_limit() < 0 ? 1000000 : static_cast<Count>(limits.iteration_limit());
  const bool use_restarts = options.get_bool("pdhg_restart");
  const bool stop_at_request = options.get_bool("pdhg_stop_at_request");

  // ---- Initial x (scaled) ------------------------------------------------
  std::vector<double> x0_scaled(n);
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    double v = 0.0;
    if (!std::isinf(scaling.col_lower[u]) && v < scaling.col_lower[u])
      v = scaling.col_lower[u];
    if (!std::isinf(scaling.col_upper[u]) && v > scaling.col_upper[u])
      v = scaling.col_upper[u];
    x0_scaled[u] = v;
  }

  // ---- Row partitioning --------------------------------------------------
  const std::vector<RowPartition> parts = partition_rows(mi, device_ids);

  logger.info(
      "Solving LP with CUDA multi-GPU ({} devices) restarted PDHG: {} rows, {} columns, {} "
      "nonzeros",
      K, rows, cols, model.num_nonzeros());
  for (int k = 0; k < K; ++k)
    logger.verbose("  device {}: rows [{}, {})", parts[static_cast<std::size_t>(k)].device_id,
                   parts[static_cast<std::size_t>(k)].row_start,
                   parts[static_cast<std::size_t>(k)].row_end);
  logger.info("Estimated ||A||_2 = {:.4e}, target tolerance {:.1e}, restarts {}", spectral_norm,
              tolerance, use_restarts ? "on" : "off");

  // ---- Allocate and set up per-device state ------------------------------
  std::vector<DeviceState> devices(static_cast<std::size_t>(K));
  bool setup_ok = true;
  for (int k = 0; k < K && setup_ok; ++k) {
    const RowPartition& part = parts[static_cast<std::size_t>(k)];
    devices[static_cast<std::size_t>(k)].device_id = part.device_id;
    setup_ok = setup_device(devices[static_cast<std::size_t>(k)], full_csr, scaling, x0_scaled,
                             part.row_start, part.row_end, ni);
  }
  if (!setup_ok) {
    logger.warning(
        "Multi-GPU PDHG: device setup failed; falling back to single-GPU on device {}",
        device_ids[0]);
    return solve_pdhg_gpu(model, options, logger, control);
  }

  // ---- Host buffers for allreduce ----------------------------------------
  std::vector<double> h_partial_aty(n, 0.0);  // scratch for each device's A^T y partial
  std::vector<double> h_aty(n, 0.0);          // accumulated sum

  // ---- CPU-side iterate buffers ------------------------------------------
  std::vector<double> h_x(n), h_xsum(n);
  std::vector<double> h_y(m), h_ysum(m);
  std::vector<double> x_unscaled(n), y_unscaled(m);
  std::vector<double> activity(m), reduced_costs(n);
  std::vector<double> x_restart(x0_scaled), y_restart(m, 0.0);

  auto unscale = [&](const std::vector<double>& xs, const std::vector<double>& ys) {
    for (Index j = 0; j < cols; ++j)
      x_unscaled[static_cast<std::size_t>(j)] =
          xs[static_cast<std::size_t>(j)] * scaling.column[static_cast<std::size_t>(j)];
    for (Index i = 0; i < rows; ++i)
      y_unscaled[static_cast<std::size_t>(i)] =
          ys[static_cast<std::size_t>(i)] * scaling.row[static_cast<std::size_t>(i)];
  };

  // Assemble full y from partitioned device memory into h_y.
  auto gather_y = [&](bool use_ysum, std::vector<double>& dest) -> bool {
    for (int k = 0; k < K; ++k) {
      const DeviceState& d = devices[static_cast<std::size_t>(k)];
      if (d.local_m == 0) continue;
      cudaSetDevice(d.device_id);
      double* src = use_ysum ? d.d_ysum : d.d_y;
      if (!dh_copy_mg(src, dest.data() + d.row_start,
                      static_cast<std::size_t>(d.local_m)))
        return false;
    }
    return true;
  };

  // ---- Grid dimensions ---------------------------------------------------
  const int bn = (ni + kBlockSize - 1) / kBlockSize;

  // ---- PDHG main loop ----------------------------------------------------
  double eta = spectral_norm > 0.0 ? 1.0 / spectral_norm : 1.0;
  double omega = 1.0;
  Count iteration = 0, restarts = 0, last_restart = 0, averaged = 0;
  double restart_kkt = std::numeric_limits<double>::infinity();

  Residuals best;
  best.primal = best.dual = best.gap = std::numeric_limits<double>::infinity();
  std::vector<double> best_x(n, 0.0), best_y(m, 0.0);

  bool converged = false, gpu_error = false, logged_table = false;
  StopController stop(control, timer, limits);
  SolveStatus stop_status = SolveStatus::kIterationLimit;

  while (!gpu_error) {
    if (iteration >= iteration_limit) break;
    if (stop.should_stop(
            [&]() {
              Progress p;
              p.phase = Progress::Phase::kLp;
              p.iterations = iteration;
              p.objective = best.primal;
              p.best_bound = sense * best.dual_objective + model.objective_offset;
              return p;
            },
            &stop_status))
      break;

    const double tau = eta / omega;
    const double sigma = eta * omega;

    // -- Step 1: A_k^T * y_k → d_aty (partial) on each device, then allreduce --
    // Launch transpose SpMV on all devices simultaneously.
    for (int k = 0; k < K; ++k) {
      DeviceState& d = devices[static_cast<std::size_t>(k)];
      cudaSetDevice(d.device_id);
      if (d.local_m > 0) {
        if (!spmv_t_mg(d, d.d_y, d.d_aty)) {
          gpu_error = true;
          break;
        }
      } else {
        if (d.n > 0) cudaMemset(d.d_aty, 0, n * sizeof(double));
      }
    }
    if (gpu_error) break;

    // Download partial A^T y from each device, sum on CPU, upload sum to all devices.
    std::fill(h_aty.begin(), h_aty.end(), 0.0);
    for (int k = 0; k < K; ++k) {
      DeviceState& d = devices[static_cast<std::size_t>(k)];
      cudaSetDevice(d.device_id);
      cudaDeviceSynchronize();
      if (!dh_copy_mg(d.d_aty, h_partial_aty.data(), n)) {
        gpu_error = true;
        break;
      }
      for (std::size_t j = 0; j < n; ++j) h_aty[j] += h_partial_aty[j];
    }
    if (gpu_error) break;
    for (int k = 0; k < K; ++k) {
      DeviceState& d = devices[static_cast<std::size_t>(k)];
      cudaSetDevice(d.device_id);
      if (!hd_copy_mg(h_aty.data(), d.d_aty, n)) {
        gpu_error = true;
        break;
      }
    }
    if (gpu_error) break;

    // -- Step 2: zero scalars and run primal update on all devices --
    for (int k = 0; k < K; ++k) {
      DeviceState& d = devices[static_cast<std::size_t>(k)];
      cudaSetDevice(d.device_id);
      if (cudaMemset(d.d_scalars, 0, 3 * sizeof(double)) != cudaSuccess) {
        gpu_error = true;
        break;
      }
      if (ni > 0)
        k_primal_fused_mg<<<bn, kBlockSize>>>(d.d_x, d.d_aty, d.d_cost, d.d_clo, d.d_chi,
                                               d.d_xn, d.d_ext, d.d_dx, d.d_scalars + 0, tau,
                                               omega, ni);
      if (!launch_ok()) {
        gpu_error = true;
        break;
      }
    }
    if (gpu_error) break;

    // -- Step 3: A_k * ext → ax_k (local dual input) --
    for (int k = 0; k < K; ++k) {
      DeviceState& d = devices[static_cast<std::size_t>(k)];
      cudaSetDevice(d.device_id);
      if (!spmv_nt_mg(d, d.d_ext, d.d_ax)) {
        gpu_error = true;
        break;
      }
    }
    if (gpu_error) break;

    // -- Step 4: dual update + A_k * dx → adx_k --
    for (int k = 0; k < K; ++k) {
      DeviceState& d = devices[static_cast<std::size_t>(k)];
      cudaSetDevice(d.device_id);
      const int bm = (d.local_m + kBlockSize - 1) / kBlockSize;
      if (d.local_m > 0) {
        k_dual_fused_mg<<<bm, kBlockSize>>>(d.d_y, d.d_ax, d.d_rlo, d.d_rhi, d.d_yn, d.d_dy,
                                             d.d_scalars + 1, sigma, omega, d.local_m);
        if (!launch_ok()) {
          gpu_error = true;
          break;
        }
      }
    }
    if (gpu_error) break;

    for (int k = 0; k < K; ++k) {
      DeviceState& d = devices[static_cast<std::size_t>(k)];
      cudaSetDevice(d.device_id);
      if (!spmv_nt_mg(d, d.d_dx, d.d_adx)) {
        gpu_error = true;
        break;
      }
    }
    if (gpu_error) break;

    // -- Step 5: interaction reduction, download scalars, reduce across devices --
    for (int k = 0; k < K; ++k) {
      DeviceState& d = devices[static_cast<std::size_t>(k)];
      cudaSetDevice(d.device_id);
      const int bm = (d.local_m + kBlockSize - 1) / kBlockSize;
      if (d.local_m > 0) {
        k_interaction_fused_mg<<<bm, kBlockSize>>>(d.d_dy, d.d_adx, d.d_scalars + 2, d.local_m);
        if (!launch_ok()) {
          gpu_error = true;
          break;
        }
      }
    }
    if (gpu_error) break;

    // Download scalars from all devices and reduce.
    // mv_x: identical on all devices (replicated primal update) — use device 0.
    // mv_y, interaction: sum partial contributions.
    double h_mv_x = 0.0, h_mv_y = 0.0, h_interaction = 0.0;
    for (int k = 0; k < K; ++k) {
      DeviceState& d = devices[static_cast<std::size_t>(k)];
      cudaSetDevice(d.device_id);
      double h_sc[3] = {};
      if (!dh_copy_mg(d.d_scalars, h_sc, 3)) {
        gpu_error = true;
        break;
      }
      if (k == 0) h_mv_x = h_sc[0];
      h_mv_y += h_sc[1];
      h_interaction += h_sc[2];
    }
    if (gpu_error) break;

    const double movement = h_mv_x + h_mv_y;
    const double interaction = std::fabs(h_interaction);

    // -- Step 6: adaptive step size [PDLP §3.1] --
    const bool no_info = (interaction <= 0.0);
    const double limit =
        no_info ? std::numeric_limits<double>::infinity() : movement / interaction;
    const double exp_val = static_cast<double>(std::max<Count>(2, iteration + 1));
    const double shrink = 1.0 - std::pow(exp_val, -0.3);
    const double grow = 1.0 + std::pow(exp_val, -0.6);
    const double proposed = std::min(shrink * limit, grow * eta);

    if (eta <= limit) {
      // Accept: pointer-swap x and y on all devices, accumulate running sums.
      for (int k = 0; k < K; ++k) {
        DeviceState& d = devices[static_cast<std::size_t>(k)];
        cudaSetDevice(d.device_id);
        std::swap(d.d_x, d.d_xn);
        std::swap(d.d_y, d.d_yn);
        if (ni > 0) k_accum_n_mg<<<bn, kBlockSize>>>(d.d_x, d.d_xsum, ni);
        const int bm = (d.local_m + kBlockSize - 1) / kBlockSize;
        if (d.local_m > 0) k_accum_m_mg<<<bm, kBlockSize>>>(d.d_y, d.d_ysum, d.local_m);
        if (!launch_ok()) {
          gpu_error = true;
          break;
        }
      }
      if (gpu_error) break;
      ++averaged;
      ++iteration;
    }
    const double eta_ceil = 1.0e3 / std::max(spectral_norm, 1e-12);
    if (!no_info) eta = std::clamp(proposed, 1e-12, eta_ceil);

    // -- Step 7: convergence check (CPU, every kEvaluationInterval) --
    if (iteration == 0) continue;
    if (iteration % kEvaluationInterval != 0 && !no_info) continue;

    // Download x from device 0 (same on all), gather y from all devices.
    const DeviceState& d0 = devices[0];
    cudaSetDevice(d0.device_id);
    if (!dh_copy_mg(d0.d_x, h_x.data(), n)) {
      gpu_error = true;
      break;
    }
    if (!gather_y(false, h_y)) {
      gpu_error = true;
      break;
    }

    unscale(h_x, h_y);
    std::vector<double> cur_x = x_unscaled, cur_y = y_unscaled;
    const Residuals cur = evaluate(prob, cur_x, cur_y, activity, reduced_costs);

    const Residuals* chosen = &cur;
    const std::vector<double>* chosen_x = &cur_x;
    const std::vector<double>* chosen_y = &cur_y;

    Residuals avg;
    std::vector<double> avg_x, avg_y;
    if (averaged > 0) {
      cudaSetDevice(d0.device_id);
      if (!dh_copy_mg(d0.d_xsum, h_xsum.data(), n)) {
        gpu_error = true;
        break;
      }
      if (!gather_y(true, h_ysum)) {
        gpu_error = true;
        break;
      }
      const double cnt = static_cast<double>(averaged);
      std::vector<double> xav(n), yav(m);
      for (std::size_t j = 0; j < n; ++j) xav[j] = h_xsum[j] / cnt;
      for (std::size_t i = 0; i < m; ++i) yav[i] = h_ysum[i] / cnt;
      unscale(xav, yav);
      avg_x = x_unscaled;
      avg_y = y_unscaled;
      avg = evaluate(prob, avg_x, avg_y, activity, reduced_costs);
      if (avg.worst() < cur.worst()) {
        chosen = &avg;
        chosen_x = &avg_x;
        chosen_y = &avg_y;
      }
    }

    const Residuals& better = *chosen;
    if (better.worst() < best.worst()) {
      best = better;
      best_x = *chosen_x;
      best_y = *chosen_y;
    }

    if (!logged_table) {
      logger.begin_iteration_table();
      logged_table = true;
    }
    logger.iteration(iteration, sense * better.primal_objective + model.objective_offset,
                     better.primal, better.dual, timer.elapsed_seconds());

    const bool stop_here =
        better.meets_request(tolerance) && (stop_at_request || better.meets_project_standard());
    if (stop_here) {
      best = better;
      best_x = *chosen_x;
      best_y = *chosen_y;
      converged = true;
      break;
    }

    // Restart logic [PDLP §4.3]
    if (use_restarts) {
      const double kkt = better.worst();
      const Count since = iteration - last_restart;
      const bool sufficient = kkt <= 0.2 * restart_kkt;
      const bool artificial =
          since >= std::max<Count>(kEvaluationInterval,
                                   static_cast<Count>(0.36 * static_cast<double>(iteration)));
      if (sufficient || artificial) {
        std::vector<double> dx_rs(n), dy_rs(m);
        for (std::size_t j = 0; j < n; ++j) dx_rs[j] = h_x[j] - x_restart[j];
        for (std::size_t i = 0; i < m; ++i) dy_rs[i] = h_y[i] - y_restart[i];
        const double dxn = euclidean_norm(dx_rs), dyn = euclidean_norm(dy_rs);
        if (dxn > 1e-12 && dyn > 1e-12) {
          constexpr double theta = 0.5;
          omega = std::exp(theta * std::log(dyn / dxn) + (1.0 - theta) * std::log(omega));
          omega = std::clamp(omega, 1e-6, 1e6);
        }
        // Zero running sums on all devices.
        for (int k = 0; k < K; ++k) {
          DeviceState& d = devices[static_cast<std::size_t>(k)];
          cudaSetDevice(d.device_id);
          cudaMemset(d.d_xsum, 0, n * sizeof(double));
          if (d.local_m > 0)
            cudaMemset(d.d_ysum, 0, static_cast<std::size_t>(d.local_m) * sizeof(double));
        }
        averaged = 0;
        x_restart = h_x;
        y_restart = h_y;
        restart_kkt = kkt;
        last_restart = iteration;
        ++restarts;
        logger.verbose("restart {} at iteration {}: KKT {:.3e}, primal weight {:.3e}", restarts,
                       iteration, kkt, omega);
      }
    }
  }  // end while

  if (gpu_error) {
    Options remaining = options;
    if (limits.has_time_limit())
      remaining.set_double("time_limit", limits.remaining_seconds(timer.elapsed_seconds()));
    logger.warning(
        "Multi-GPU PDHG: CUDA error after {:.2f}s; falling back to CPU PDHG on remaining "
        "budget",
        timer.elapsed_seconds());
    return pdhg::solve_pdhg(model, remaining, logger, control);
  }

  // ---- Extract solution --------------------------------------------------
  for (Index j = 0; j < cols; ++j)
    solution.col_value[static_cast<std::size_t>(j)] =
        best_x.empty() ? 0.0 : best_x[static_cast<std::size_t>(j)];

  const Residuals final_r = evaluate(prob, best_x, best_y, activity, reduced_costs);
  for (Index j = 0; j < cols; ++j)
    solution.col_dual[static_cast<std::size_t>(j)] =
        sense * reduced_costs[static_cast<std::size_t>(j)];
  for (Index i = 0; i < rows; ++i)
    solution.row_dual[static_cast<std::size_t>(i)] =
        sense * (-best_y[static_cast<std::size_t>(i)]);

  solution.iterations = iteration;
  solution.solve_seconds = timer.elapsed_seconds();

  const bool verifiable = converged && final_r.meets_project_standard();
  if (verifiable) {
    solution.status = SolveStatus::kOptimal;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CUDA multi-GPU PDHG (%d devices) converged after %ld iterations and %ld restarts; "
        "absolute primal %.3e, dual %.3e, relative gap %.3e",
        K, iteration, restarts, final_r.absolute_primal, final_r.absolute_dual,
        final_r.gap_as_verified);
    solution.message = buf;
  } else if (converged) {
    solution.status = SolveStatus::kFeasible;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CUDA multi-GPU PDHG (%d devices) met requested tolerance %.1e after %ld iterations "
        "but NOT project standard",
        K, tolerance, iteration);
    solution.message = buf;
  } else {
    solution.status =
        (stop_status == SolveStatus::kTimeLimit || stop_status == SolveStatus::kInterrupted)
            ? stop_status
            : SolveStatus::kIterationLimit;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CUDA multi-GPU PDHG (%d devices) stopped at relative primal %.3e, dual %.3e, "
        "gap %.3e after %ld iterations and %ld restarts (target %.1e)",
        K, final_r.primal, final_r.dual, final_r.gap, iteration, restarts, tolerance);
    solution.message = buf;
  }

  if (verifiable) {
    solution.dual_bound = sense * final_r.dual_objective + model.objective_offset;
  } else {
    solution.dual_bound = model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  }
  solution.recompute_quality(model);

  logger.info("");
  logger.info(
      "Status: {}  objective {:.10e}  iterations {}  restarts {}  devices {}  time {:.3f}s",
      to_string(solution.status), solution.objective, solution.iterations, restarts, K,
      solution.solve_seconds);
  logger.info("Relative residuals: primal {:.3e}, dual {:.3e}, gap {:.3e}", final_r.primal,
              final_r.dual, final_r.gap);
  if (!solution.message.empty()) logger.info("{}", solution.message);
  return solution;
}

}  // namespace sankhya::gpu

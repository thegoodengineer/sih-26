// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU-accelerated restarted PDHG for LP.
//
// References (written from the papers; per ENGINEERING_RULES.md no solver source was
// consulted):
//   [CP11]  Chambolle & Pock, "A first-order primal-dual algorithm for convex problems with
//           applications to imaging", JMIV 40(1), 2011. Algorithm 1 is the iteration below.
//   [PDLP]  Applegate et al., "Practical Large-Scale LP using PDHG", NeurIPS 2021.
//           Sections 3.1 (adaptive step size), 3.2 (primal weight), 4.3 (restarts).
//   [cuPDLP] Lu & Yang, "cuPDLP.jl: A GPU Implementation of Restarted PDHG for LP",
//           arXiv:2311.12180. GPU design reference.
//
// CPU/GPU split:
//   GPU  SpMVs (cuSPARSE), primal/dual coordinate updates, running-sum accumulation,
//        movement and interaction reductions (CUB BlockReduce + atomicAdd, #280).
//   CPU  Preconditioning (one-off), convergence evaluation (every 40 iterations),
//        restart logic and scalar step-size arithmetic.
//
// The CSR matrix is built on CPU from CsrView(scaling.matrix) and uploaded once.
// The GPU VRAM ceiling on the primary test machine (RTX 5050) is 6 GB.

#include "pdhg_gpu.hpp"

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

// ---- Error-check macros -------------------------------------------------
// Each returns false from the enclosing function, which the caller treats as a GPU failure
// and either logs + falls back, or frees resources before returning.

#define CUDA_CHECK(expr)                     \
  do {                                       \
    if ((expr) != cudaSuccess) return false; \
  } while (0)

#define CS_CHECK(expr)                                   \
  do {                                                   \
    if ((expr) != CUSPARSE_STATUS_SUCCESS) return false; \
  } while (0)

// A kernel launch reports its failure (bad configuration, out of resources) through
// cudaPeekAtLastError, not through the launch statement, and not reliably through a later
// copy. Checked after every launch below and mapped to the same fallback as any other
// device failure.
static bool launch_ok() {
  return cudaPeekAtLastError() == cudaSuccess;
}

// ---- cuSPARSE scalar constants (host pointers, valid as alpha/beta) -----
static const double kOne = 1.0;
static const double kZero = 0.0;

// ---- CUDA kernels -------------------------------------------------------

// [CP11] Algorithm 1, primal half-step, fused with [PDLP] §3.1 movement_x reduction (#280).
// Combining the update with the reduction eliminates one kernel launch and one global-memory
// round-trip (write dx to d_tmpn, read it back for CUB DeviceReduce) per iteration.
//   x_next = proj_[col_lo, col_hi](x - tau*(cost + at_y))
//   extrapolated = 2*x_next - x    ([CP11] over-relaxation)
//   dx = x_next - x
//   d_mv_x += sum_j 0.5 * omega * dx[j]^2   (via CUB BlockReduce + atomicAdd)
__global__ void k_primal_fused(const double* __restrict__ x, const double* __restrict__ at_y,
                               const double* __restrict__ cost, const double* __restrict__ col_lo,
                               const double* __restrict__ col_hi, double* __restrict__ x_next,
                               double* __restrict__ extrapolated, double* __restrict__ dx,
                               double* d_mv_x, double tau, double omega, int n) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  double mv_thread = 0.0;
  if (j < n) {
    double xnj = x[j] - tau * (cost[j] + at_y[j]);
    // project onto [col_lo[j], col_hi[j]]; infinite sides (IEEE) are treated as absent
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

// [CP11] Algorithm 1, dual half-step, fused with [PDLP] §3.1 movement_y reduction (#280).
//   v = y + sigma * a_x
//   y_next = v - sigma * proj_[row_lo, row_hi](v / sigma)   (Moreau identity)
//   dy = y_next - y
//   d_mv_y += sum_i 0.5 * dy[i]^2 / omega   (via CUB BlockReduce + atomicAdd)
__global__ void k_dual_fused(const double* __restrict__ y, const double* __restrict__ a_x,
                             const double* __restrict__ row_lo, const double* __restrict__ row_hi,
                             double* __restrict__ y_next, double* __restrict__ dy, double* d_mv_y,
                             double sigma, double omega, int m) {
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

// [PDLP] §3.1 interaction: d_interaction += sum_i dy[i] * adx[i]
// Replaces the previous k_pointwise_mul + CUB DeviceReduce pair (#280).
__global__ void k_interaction_fused(const double* __restrict__ dy,
                                    const double* __restrict__ adx, double* d_interaction,
                                    int m) {
  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const double val = (i < m) ? dy[i] * adx[i] : 0.0;
  const double bsum = BlockReduce(temp).Sum(val);
  if (threadIdx.x == 0) atomicAdd(d_interaction, bsum);
}

// Accumulate into running sums (for the average iterate)
__global__ void k_accum_n(const double* __restrict__ v, double* __restrict__ sum, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  sum[j] += v[j];
}

__global__ void k_accum_m(const double* __restrict__ v, double* __restrict__ sum, int m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  sum[i] += v[i];
}

// ---- Device-memory helpers ----------------------------------------------

static double* dev_zeros(std::size_t count) {
  if (count == 0) return nullptr;
  double* p = nullptr;
  if (cudaMalloc(&p, count * sizeof(double)) != cudaSuccess) return nullptr;
  if (cudaMemset(p, 0, count * sizeof(double)) != cudaSuccess) {
    cudaFree(p);
    return nullptr;
  }
  return p;
}

static int* dev_int(std::size_t count) {
  if (count == 0) return nullptr;
  int* p = nullptr;
  if (cudaMalloc(&p, count * sizeof(int)) != cudaSuccess) return nullptr;
  return p;
}

template <typename T>
static bool hd_copy(const T* host, T* dev, std::size_t count) {
  if (count == 0) return true;
  return cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

static bool dh_copy(const double* dev, double* host, std::size_t count) {
  if (count == 0) return true;
  return cudaMemcpy(host, dev, count * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
}

// ---- RAII wrapper for all GPU resources ---------------------------------

struct GpuState {
  // Iterates (scaled space)
  double *d_x{}, *d_xn{}, *d_ext{}, *d_dx{}, *d_aty{}, *d_xsum{};
  double *d_y{}, *d_yn{}, *d_dy{}, *d_ax{}, *d_adx{}, *d_ysum{};
  // Per-iteration scalar accumulators: [0]=movement_x, [1]=movement_y, [2]=interaction.
  // Zeroed by cudaMemset at the top of each iteration; updated via atomicAdd inside the
  // fused primal/dual kernels; downloaded in one transfer to avoid per-scalar sync stalls.
  double* d_scalars{};
  // Problem constants (device)
  double *d_cost{}, *d_clo{}, *d_chi{}, *d_rlo{}, *d_rhi{};
  // CSR matrix (device)
  int *d_rowptr{}, *d_colidx{};
  double* d_vals{};
  // cuSPARSE SpMV buffer (single buffer, sized to max of NT and T)
  void* d_spmv{};
  std::size_t spmv_bytes{};
  // cuSPARSE handles
  cusparseHandle_t cs{};
  cusparseSpMatDescr_t mat{};
  cusparseDnVecDescr_t vn{};  // n-element dense vector
  cusparseDnVecDescr_t vm{};  // m-element dense vector

  int n{}, m{}, nnz{};

  ~GpuState() {
    if (vm) cusparseDestroyDnVec(vm);
    if (vn) cusparseDestroyDnVec(vn);
    if (mat) cusparseDestroySpMat(mat);
    if (cs) cusparseDestroy(cs);
    cudaFree(d_x);
    cudaFree(d_xn);
    cudaFree(d_ext);
    cudaFree(d_dx);
    cudaFree(d_aty);
    cudaFree(d_xsum);
    cudaFree(d_y);
    cudaFree(d_yn);
    cudaFree(d_dy);
    cudaFree(d_ax);
    cudaFree(d_adx);
    cudaFree(d_ysum);
    cudaFree(d_scalars);
    cudaFree(d_cost);
    cudaFree(d_clo);
    cudaFree(d_chi);
    cudaFree(d_rlo);
    cudaFree(d_rhi);
    cudaFree(d_rowptr);
    cudaFree(d_colidx);
    cudaFree(d_vals);
    cudaFree(d_spmv);
  }

  [[nodiscard]] bool alloc_ok() const {
    return d_x && d_xn && d_ext && d_dx && d_aty && d_xsum && d_y && d_yn && d_dy && d_ax &&
           d_adx && d_ysum && d_scalars && d_cost && d_clo && d_chi;
  }
};

// ---- cuSPARSE SpMV helpers ----------------------------------------------

// Non-transpose SpMV: d_out_m = A * d_in_n
static bool spmv_nt(GpuState& g, double* d_in_n, double* d_out_m) {
  if (g.m == 0 || g.nnz == 0) {
    if (g.m > 0) cudaMemset(d_out_m, 0, static_cast<std::size_t>(g.m) * sizeof(double));
    return true;
  }
  CS_CHECK(cusparseDnVecSetValues(g.vn, d_in_n));
  CS_CHECK(cusparseDnVecSetValues(g.vm, d_out_m));
  CS_CHECK(cusparseSpMV(g.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, g.mat, g.vn, &kZero,
                        g.vm, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, g.d_spmv));
  return true;
}

// Transpose SpMV: d_out_n = A^T * d_in_m
static bool spmv_t(GpuState& g, double* d_in_m, double* d_out_n) {
  if (g.m == 0 || g.nnz == 0) {
    if (g.n > 0) cudaMemset(d_out_n, 0, static_cast<std::size_t>(g.n) * sizeof(double));
    return true;
  }
  CS_CHECK(cusparseDnVecSetValues(g.vm, d_in_m));
  CS_CHECK(cusparseDnVecSetValues(g.vn, d_out_n));
  CS_CHECK(cusparseSpMV(g.cs, CUSPARSE_OPERATION_TRANSPOSE, &kOne, g.mat, g.vm, &kZero, g.vn,
                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, g.d_spmv));
  return true;
}

using pdhg::euclidean_norm;
using pdhg::evaluate;
using pdhg::Problem;
using pdhg::Residuals;

}  // anonymous namespace

// ---- Main solver --------------------------------------------------------

Solution solve_pdhg_gpu(const Model& model, const Options& options, Logger& logger,
                        SolveControl* control) {
  std::string device_desc;
  if (!device_available(&device_desc)) {
    logger.warning("GPU PDHG: no CUDA device ({}); falling back to CPU solver", device_desc);
    return pdhg::solve_pdhg(model, options, logger, control);
  }

  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "pdhg-cuda";

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
  const int ni = static_cast<int>(n), mi = static_cast<int>(m);

  // ---- Unscaled minimise-space problem ------------------------------------
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

  // ---- Preconditioning (CPU, one-off) ------------------------------------
  constexpr int kRuizIter = 10;
  constexpr int kPowerIter = 30;
  const Scaling scaling = build_scaling(model, prob.cost, kRuizIter);
  const double spectral_norm = estimate_spectral_norm(
      scaling.matrix, kPowerIter, static_cast<unsigned>(options.get_int("random_seed")) + 1u);

  // ---- Build CSR on CPU via CsrView --------------------------------------
  const CsrView csr(scaling.matrix);
  const int nnz = static_cast<int>(csr.values().size());

  // ---- Solver parameters -------------------------------------------------
  const double tolerance = options.get_double("pdhg_tolerance");
  const ResourceLimits limits(options, logger);
  const Count iteration_limit =
      limits.iteration_limit() < 0 ? 1000000 : static_cast<Count>(limits.iteration_limit());
  const bool use_restarts = options.get_bool("pdhg_restart");
  const bool stop_at_request = options.get_bool("pdhg_stop_at_request");

  logger.info("Solving LP with CUDA restarted PDHG on {}: {} rows, {} columns, {} nonzeros",
              device_desc, rows, cols, model.num_nonzeros());
  logger.info("Scaled matrix entries in [{:.3e}, {:.3e}], estimated ||A||_2 = {:.4e}",
              scaling.min_abs, scaling.max_abs, spectral_norm);
  logger.info("Target relative tolerance {:.1e}, restarts {}", tolerance,
              use_restarts ? "on" : "off");

  // ---- GPU resource allocation -------------------------------------------
  GpuState g;
  g.n = ni;
  g.m = mi;
  g.nnz = nnz;

  g.d_x = dev_zeros(n);
  g.d_xn = dev_zeros(n);
  g.d_ext = dev_zeros(n);
  g.d_dx = dev_zeros(n);
  g.d_aty = dev_zeros(n);
  g.d_xsum = dev_zeros(n);
  g.d_y = dev_zeros(m);
  g.d_yn = dev_zeros(m);
  g.d_dy = dev_zeros(m);
  g.d_ax = dev_zeros(m);
  g.d_adx = dev_zeros(m);
  g.d_ysum = dev_zeros(m);
  g.d_scalars = dev_zeros(3);  // [0]=mv_x, [1]=mv_y, [2]=interaction
  g.d_cost = dev_zeros(n);
  g.d_clo = dev_zeros(n);
  g.d_chi = dev_zeros(n);
  if (m > 0) {
    g.d_rlo = dev_zeros(m);
    g.d_rhi = dev_zeros(m);
  }
  g.d_rowptr = dev_int(m + 1);
  g.d_colidx = nnz > 0 ? dev_int(static_cast<std::size_t>(nnz)) : nullptr;
  g.d_vals = nnz > 0 ? dev_zeros(static_cast<std::size_t>(nnz)) : nullptr;

  if (!g.alloc_ok() || !g.d_rowptr || (m > 0 && (!g.d_rlo || !g.d_rhi)) ||
      (nnz > 0 && (!g.d_colidx || !g.d_vals))) {
    logger.warning("GPU PDHG: device allocation failed; falling back to CPU solver");
    return pdhg::solve_pdhg(model, options, logger, control);
  }

  // ---- Upload matrix and problem data ------------------------------------
  {
    const auto& rs = csr.row_starts();
    const auto& ci = csr.column_indices();
    const auto& cv = csr.values();
    // Index = int32_t matches CUSPARSE_INDEX_32I; both vectors hold plain ints on the wire
    if (!hd_copy(rs.data(), g.d_rowptr, rs.size()) ||
        (nnz > 0 && (!hd_copy(ci.data(), g.d_colidx, ci.size()) ||
                     !hd_copy(cv.data(), g.d_vals, cv.size()))) ||
        !hd_copy(scaling.cost.data(), g.d_cost, n) ||
        !hd_copy(scaling.col_lower.data(), g.d_clo, n) ||
        !hd_copy(scaling.col_upper.data(), g.d_chi, n) ||
        (m > 0 && (!hd_copy(scaling.row_lower.data(), g.d_rlo, m) ||
                   !hd_copy(scaling.row_upper.data(), g.d_rhi, m)))) {
      logger.warning("GPU PDHG: data upload failed; falling back to CPU solver");
      return pdhg::solve_pdhg(model, options, logger, control);
    }

    // Initial x: projection of 0 onto column bounds (same as CPU PDHG)
    std::vector<double> x0(n);
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      double v = 0.0;
      if (!std::isinf(scaling.col_lower[u]) && v < scaling.col_lower[u])
        v = scaling.col_lower[u];
      if (!std::isinf(scaling.col_upper[u]) && v > scaling.col_upper[u])
        v = scaling.col_upper[u];
      x0[u] = v;
    }
    if (!hd_copy(x0.data(), g.d_x, n)) {
      logger.warning("GPU PDHG: initial iterate upload failed; falling back to CPU solver");
      return pdhg::solve_pdhg(model, options, logger, control);
    }
  }

  // ---- cuSPARSE setup ----------------------------------------------------
  if (cusparseCreate(&g.cs) != CUSPARSE_STATUS_SUCCESS) {
    logger.warning("GPU PDHG: cusparseCreate failed; falling back to CPU solver");
    return pdhg::solve_pdhg(model, options, logger, control);
  }

  auto cs_failed = [&]() -> Solution {
    logger.warning("GPU PDHG: cuSPARSE setup failed; falling back to CPU solver");
    return pdhg::solve_pdhg(model, options, logger, control);
  };

  // Create CSR matrix descriptor.  When nnz=0, d_colidx/d_vals are nullptr;
  // cusparseCreateCsr accepts null pointers when nnz=0.
  if (cusparseCreateCsr(&g.mat, static_cast<int64_t>(mi), static_cast<int64_t>(ni),
                        static_cast<int64_t>(nnz), g.d_rowptr, g.d_colidx, g.d_vals,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_64F) != CUSPARSE_STATUS_SUCCESS)
    return cs_failed();

  // Create dense vector descriptors; dimension is pinned at construction, data updated later.
  // vn: n-element (input to NT SpMV, output of T SpMV)
  // vm: m-element (output of NT SpMV, input to T SpMV)
  void* vn_init = (g.d_x ? g.d_x : g.d_scalars);  // non-null placeholder
  void* vm_init = (g.d_y ? g.d_y : g.d_scalars);
  if (cusparseCreateDnVec(&g.vn, static_cast<int64_t>(ni), vn_init, CUDA_R_64F) !=
      CUSPARSE_STATUS_SUCCESS)
    return cs_failed();
  if (mi > 0) {
    if (cusparseCreateDnVec(&g.vm, static_cast<int64_t>(mi), vm_init, CUDA_R_64F) !=
        CUSPARSE_STATUS_SUCCESS)
      return cs_failed();
  }

  // Query SpMV buffer sizes and allocate a single buffer for both operations.
  if (m > 0 && nnz > 0) {
    std::size_t bytes_nt = 0, bytes_t = 0;
    cusparseDnVecSetValues(g.vn, g.d_ext);
    cusparseDnVecSetValues(g.vm, g.d_ax);
    if (cusparseSpMV_bufferSize(g.cs, CUSPARSE_OPERATION_NON_TRANSPOSE, &kOne, g.mat, g.vn,
                                &kZero, g.vm, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                &bytes_nt) != CUSPARSE_STATUS_SUCCESS)
      return cs_failed();
    cusparseDnVecSetValues(g.vm, g.d_y);
    cusparseDnVecSetValues(g.vn, g.d_aty);
    if (cusparseSpMV_bufferSize(g.cs, CUSPARSE_OPERATION_TRANSPOSE, &kOne, g.mat, g.vm, &kZero,
                                g.vn, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                &bytes_t) != CUSPARSE_STATUS_SUCCESS)
      return cs_failed();
    g.spmv_bytes = std::max(bytes_nt, bytes_t);
    if (g.spmv_bytes > 0) {
      if (cudaMalloc(&g.d_spmv, g.spmv_bytes) != cudaSuccess) return cs_failed();
    }
  }

  // ---- Main iteration loop -----------------------------------------------
  const int bn = (ni + kBlockSize - 1) / kBlockSize;
  const int bm = (mi + kBlockSize - 1) / kBlockSize;

  // CPU-side vectors for convergence evaluation
  std::vector<double> h_x(n), h_y(m), h_xsum(n), h_ysum(m);
  std::vector<double> x_unscaled(n), y_unscaled(m);
  std::vector<double> activity(m), reduced_costs(n);
  // Restart reference point (scaled)
  std::vector<double> x_restart(n, 0.0), y_restart(m, 0.0);
  {
    // init x_restart = x0 (same as initial iterate)
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      double v = 0.0;
      if (!std::isinf(scaling.col_lower[u]) && v < scaling.col_lower[u])
        v = scaling.col_lower[u];
      if (!std::isinf(scaling.col_upper[u]) && v > scaling.col_upper[u])
        v = scaling.col_upper[u];
      x_restart[u] = v;
    }
  }

  auto unscale = [&](const std::vector<double>& xs, const std::vector<double>& ys) {
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      x_unscaled[u] = xs[u] * scaling.column[u];
    }
    for (Index i = 0; i < rows; ++i) {
      const auto u = static_cast<std::size_t>(i);
      y_unscaled[u] = ys[u] * scaling.row[u];
    }
  };

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

    // Zero per-iteration scalar accumulators before the fused kernels write into them.
    if (cudaMemset(g.d_scalars, 0, 3 * sizeof(double)) != cudaSuccess) {
      gpu_error = true;
      break;
    }

    // 1. A^T y  [cuPDLP §3, transpose SpMV]
    if (!spmv_t(g, g.d_y, g.d_aty)) {
      gpu_error = true;
      break;
    }

    // 2. Primal update + movement_x reduction  [CP11 Alg.1, PDLP §3.1]
    if (ni > 0)
      k_primal_fused<<<bn, kBlockSize>>>(g.d_x, g.d_aty, g.d_cost, g.d_clo, g.d_chi, g.d_xn,
                                         g.d_ext, g.d_dx, g.d_scalars + 0, tau, omega, ni);
    if (!launch_ok()) {
      gpu_error = true;
      break;
    }

    // 3. A * extrapolated  [CP11 Alg.1 dual step]
    if (!spmv_nt(g, g.d_ext, g.d_ax)) {
      gpu_error = true;
      break;
    }

    // 4. Dual update + movement_y reduction  [CP11 Alg.1, PDLP §3.1]
    if (mi > 0)
      k_dual_fused<<<bm, kBlockSize>>>(g.d_y, g.d_ax, g.d_rlo, g.d_rhi, g.d_yn, g.d_dy,
                                       g.d_scalars + 1, sigma, omega, mi);
    if (!launch_ok()) {
      gpu_error = true;
      break;
    }

    // 5. A * dx  [PDLP §3.1 interaction term]
    if (!spmv_nt(g, g.d_dx, g.d_adx)) {
      gpu_error = true;
      break;
    }

    // 6. Interaction + download all three scalars in one transfer  [PDLP §3.1]
    // movement_x (d_scalars[0]) and movement_y (d_scalars[1]) were accumulated by steps 2/4.
    if (mi > 0) {
      k_interaction_fused<<<bm, kBlockSize>>>(g.d_dy, g.d_adx, g.d_scalars + 2, mi);
      if (!launch_ok()) {
        gpu_error = true;
        break;
      }
    }
    double h_scalars[3] = {};
    if (!dh_copy(g.d_scalars, h_scalars, 3)) {
      gpu_error = true;
      break;
    }
    const double movement = h_scalars[0] + h_scalars[1];
    const double interaction = std::fabs(h_scalars[2]);

    // 7. Adaptive step size  [PDLP §3.1]
    const bool no_info = (interaction <= 0.0);
    const double limit =
        no_info ? std::numeric_limits<double>::infinity() : movement / interaction;
    // exponent floor of 2 avoids the shrink=0 collapse on the first iteration  [PDLP §3.1]
    const double exp = static_cast<double>(std::max<Count>(2, iteration + 1));
    const double shrink = 1.0 - std::pow(exp, -0.3);
    const double grow = 1.0 + std::pow(exp, -0.6);
    const double proposed = std::min(shrink * limit, grow * eta);

    if (eta <= limit) {
      // Accept: swap iterates via pointer swap
      std::swap(g.d_x, g.d_xn);
      std::swap(g.d_y, g.d_yn);
      // Accumulate running sums
      if (ni > 0) k_accum_n<<<bn, kBlockSize>>>(g.d_x, g.d_xsum, ni);
      if (mi > 0) k_accum_m<<<bm, kBlockSize>>>(g.d_y, g.d_ysum, mi);
      if (!launch_ok()) {
        gpu_error = true;
        break;
      }
      ++averaged;
      ++iteration;
    }
    const double eta_ceil = 1.0e3 / std::max(spectral_norm, 1e-12);
    if (!no_info) eta = std::clamp(proposed, 1e-12, eta_ceil);

    // 8. Convergence and restart check (CPU, every kEvaluationInterval accepted steps)
    if (iteration == 0) continue;
    if (iteration % kEvaluationInterval != 0 && !no_info) continue;

    // Download current iterates
    if (!dh_copy(g.d_x, h_x.data(), n) || !dh_copy(g.d_y, h_y.data(), m)) {
      gpu_error = true;
      break;
    }

    // Evaluate current iterate
    unscale(h_x, h_y);
    std::vector<double> cur_x = x_unscaled, cur_y = y_unscaled;
    const Residuals cur = evaluate(prob, cur_x, cur_y, activity, reduced_costs);

    // Evaluate running average  [PDLP §4.3]
    const Residuals* chosen = &cur;
    const std::vector<double>* chosen_x = &cur_x;
    const std::vector<double>* chosen_y = &cur_y;

    Residuals avg;
    std::vector<double> avg_x, avg_y;
    if (averaged > 0) {
      if (!dh_copy(g.d_xsum, h_xsum.data(), n) || !dh_copy(g.d_ysum, h_ysum.data(), m)) {
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

    // Restart logic  [PDLP §4.3]
    if (use_restarts) {
      const double kkt = better.worst();
      const Count since = iteration - last_restart;
      const bool sufficient = kkt <= 0.2 * restart_kkt;
      const bool artificial =
          since >= std::max<Count>(kEvaluationInterval,
                                   static_cast<Count>(0.36 * static_cast<double>(iteration)));

      if (sufficient || artificial) {
        // Primal weight update towards observed ratio of dual/primal movement  [PDLP §3.2]
        std::vector<double> dx_rs(n), dy_rs(m);
        for (std::size_t j = 0; j < n; ++j) dx_rs[j] = h_x[j] - x_restart[j];
        for (std::size_t i = 0; i < m; ++i) dy_rs[i] = h_y[i] - y_restart[i];
        const double dxn = euclidean_norm(dx_rs), dyn = euclidean_norm(dy_rs);
        if (dxn > 1e-12 && dyn > 1e-12) {
          constexpr double theta = 0.5;
          omega = std::exp(theta * std::log(dyn / dxn) + (1.0 - theta) * std::log(omega));
          omega = std::clamp(omega, 1e-6, 1e6);
        }

        cudaMemset(g.d_xsum, 0, n * sizeof(double));
        cudaMemset(g.d_ysum, 0, m * sizeof(double));
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
    // The CPU engine takes over on what the budget has left (#289): the seconds already
    // spent on the device are not handed out a second time.
    Options remaining = options;
    if (limits.has_time_limit()) {
      remaining.set_double("time_limit", limits.remaining_seconds(timer.elapsed_seconds()));
    }
    logger.warning(
        "GPU PDHG: CUDA error during solve after {:.2f}s; falling back to the CPU "
        "solver on the remaining budget",
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
        "CUDA PDHG converged after %ld iterations and %ld restarts; absolute primal %.3e, "
        "dual %.3e, relative gap %.3e",
        iteration, restarts, final_r.absolute_primal, final_r.absolute_dual,
        final_r.gap_as_verified);
    solution.message = buf;
  } else if (converged) {
    solution.status = SolveStatus::kFeasible;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CUDA PDHG met requested tolerance %.1e after %ld iterations but NOT project standard "
        "(primal %.3e vs %.1e, dual %.3e vs %.1e, gap %.3e vs %.1e)",
        tolerance, iteration, final_r.absolute_primal, tol::kPrimalFeasibility,
        final_r.absolute_dual, tol::kDualFeasibility, final_r.gap_as_verified,
        tol::kDualityGap);
    solution.message = buf;
  } else {
    solution.status =
        (stop_status == SolveStatus::kTimeLimit || stop_status == SolveStatus::kInterrupted)
            ? stop_status
            : SolveStatus::kIterationLimit;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "CUDA PDHG stopped at relative primal %.3e, dual %.3e, gap %.3e after %ld "
        "iterations and %ld restarts (target %.1e)",
        final_r.primal, final_r.dual, final_r.gap, iteration, restarts, tolerance);
    solution.message = buf;
  }

  if (verifiable) {
    solution.dual_bound = sense * final_r.dual_objective + model.objective_offset;
  } else {
    solution.dual_bound = model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  }
  solution.recompute_quality(model);

  logger.info("");
  logger.info("Status: {}   objective {:.10e}   iterations {}   restarts {}   time {:.3f}s",
              to_string(solution.status), solution.objective, solution.iterations, restarts,
              solution.solve_seconds);
  logger.info("Relative residuals: primal {:.3e}, dual {:.3e}, gap {:.3e}", final_r.primal,
              final_r.dual, final_r.gap);
  if (!solution.message.empty()) logger.info("{}", solution.message);
  return solution;
}

}  // namespace sankhya::gpu

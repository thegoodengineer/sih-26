// SPDX-License-Identifier: Apache-2.0
// SANKHYA - diagonal preconditioning. See scaling.hpp for the references.

#include "scaling.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace sankhya::pdhg {
namespace {

/// Multiply the accumulated scaling by a new pass and rebuild the scaled matrix.
void apply_pass(const std::vector<double>& row_pass, const std::vector<double>& column_pass,
                Scaling* scaling) {
  const Index rows = scaling->matrix.num_rows();
  const Index cols = scaling->matrix.num_cols();

  SparseMatrix rebuilt(rows, cols);
  rebuilt.reserve(static_cast<std::size_t>(scaling->matrix.num_nonzeros()));
  for (Index j = 0; j < cols; ++j) {
    const ColumnView column = scaling->matrix.column(j);
    const double cj = column_pass[static_cast<std::size_t>(j)];
    for (Index k = 0; k < column.size; ++k) {
      const Index i = column.rows[k];
      rebuilt.add_entry(i, j, column.values[k] * row_pass[static_cast<std::size_t>(i)] * cj);
    }
  }
  // Drop tolerance zero: a scaling pass must never change the sparsity pattern, or the
  // scaled problem stops being the same problem.
  rebuilt.finalize(0.0);
  scaling->matrix = std::move(rebuilt);

  for (Index i = 0; i < rows; ++i) {
    scaling->row[static_cast<std::size_t>(i)] *= row_pass[static_cast<std::size_t>(i)];
  }
  for (Index j = 0; j < cols; ++j) {
    scaling->column[static_cast<std::size_t>(j)] *= column_pass[static_cast<std::size_t>(j)];
  }
}

/// One Ruiz round: scale each row and column by the inverse square root of its largest
/// absolute entry, which drives both towards an infinity norm of 1.
void ruiz_round(Scaling* scaling) {
  const Index rows = scaling->matrix.num_rows();
  const Index cols = scaling->matrix.num_cols();
  std::vector<double> row_max(static_cast<std::size_t>(rows), 0.0);
  std::vector<double> column_max(static_cast<std::size_t>(cols), 0.0);

  for (Index j = 0; j < cols; ++j) {
    const ColumnView column = scaling->matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const double magnitude = std::fabs(column.values[k]);
      const auto i = static_cast<std::size_t>(column.rows[k]);
      row_max[i] = std::max(row_max[i], magnitude);
      column_max[static_cast<std::size_t>(j)] =
          std::max(column_max[static_cast<std::size_t>(j)], magnitude);
    }
  }

  // An empty row or column has nothing to equilibrate; leave it at 1 rather than dividing
  // by zero.
  std::vector<double> row_pass(static_cast<std::size_t>(rows), 1.0);
  std::vector<double> column_pass(static_cast<std::size_t>(cols), 1.0);
  for (std::size_t i = 0; i < row_max.size(); ++i) {
    if (row_max[i] > 0.0) row_pass[i] = 1.0 / std::sqrt(row_max[i]);
  }
  for (std::size_t j = 0; j < column_max.size(); ++j) {
    if (column_max[j] > 0.0) column_pass[j] = 1.0 / std::sqrt(column_max[j]);
  }
  apply_pass(row_pass, column_pass, scaling);
}

/// Pock-Chambolle with alpha = 1: row i by 1/sqrt(sum_j |a_ij|), column j by
/// 1/sqrt(sum_i |a_ij|). Pock & Chambolle (2011), section 4.
void pock_chambolle_round(Scaling* scaling) {
  const Index rows = scaling->matrix.num_rows();
  const Index cols = scaling->matrix.num_cols();
  std::vector<double> row_sum(static_cast<std::size_t>(rows), 0.0);
  std::vector<double> column_sum(static_cast<std::size_t>(cols), 0.0);

  for (Index j = 0; j < cols; ++j) {
    const ColumnView column = scaling->matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const double magnitude = std::fabs(column.values[k]);
      row_sum[static_cast<std::size_t>(column.rows[k])] += magnitude;
      column_sum[static_cast<std::size_t>(j)] += magnitude;
    }
  }

  std::vector<double> row_pass(static_cast<std::size_t>(rows), 1.0);
  std::vector<double> column_pass(static_cast<std::size_t>(cols), 1.0);
  for (std::size_t i = 0; i < row_sum.size(); ++i) {
    if (row_sum[i] > 0.0) row_pass[i] = 1.0 / std::sqrt(row_sum[i]);
  }
  for (std::size_t j = 0; j < column_sum.size(); ++j) {
    if (column_sum[j] > 0.0) column_pass[j] = 1.0 / std::sqrt(column_sum[j]);
  }
  apply_pass(row_pass, column_pass, scaling);
}

/// A bound scales by the multiplier when finite, and stays infinite when not. Multiplying
/// an infinity would produce an infinity anyway, but doing it explicitly keeps a NaN from
/// appearing if a multiplier is ever zero.
double scale_bound(double bound, double multiplier) {
  return is_finite_bound(bound) ? bound * multiplier : bound;
}

}  // namespace

Scaling build_scaling(const Model& model, const std::vector<double>& min_space_cost,
                      int ruiz_iterations) {
  const Index rows = model.num_rows();
  const Index cols = model.num_cols();

  Scaling scaling;
  scaling.row.assign(static_cast<std::size_t>(rows), 1.0);
  scaling.column.assign(static_cast<std::size_t>(cols), 1.0);
  scaling.matrix = model.matrix;

  for (int pass = 0; pass < ruiz_iterations; ++pass) ruiz_round(&scaling);
  pock_chambolle_round(&scaling);

  // Apply the accumulated scaling to the vectors.
  //   xhat = Dc^-1 x   so   lhat = l / Dc,  chat = Dc c
  //   Ahat xhat = Dr A x   so   rlhat = Dr rl
  scaling.cost.resize(static_cast<std::size_t>(cols));
  scaling.col_lower.resize(static_cast<std::size_t>(cols));
  scaling.col_upper.resize(static_cast<std::size_t>(cols));
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double dc = scaling.column[u];
    scaling.cost[u] = min_space_cost[u] * dc;
    scaling.col_lower[u] = scale_bound(model.col_lower[u], 1.0 / dc);
    scaling.col_upper[u] = scale_bound(model.col_upper[u], 1.0 / dc);
  }

  scaling.row_lower.resize(static_cast<std::size_t>(rows));
  scaling.row_upper.resize(static_cast<std::size_t>(rows));
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double dr = scaling.row[u];
    scaling.row_lower[u] = scale_bound(model.row_lower[u], dr);
    scaling.row_upper[u] = scale_bound(model.row_upper[u], dr);
  }

  scaling.max_abs = 0.0;
  scaling.min_abs = 0.0;
  bool first = true;
  for (const double value : scaling.matrix.values()) {
    const double magnitude = std::fabs(value);
    scaling.max_abs = std::max(scaling.max_abs, magnitude);
    if (first || magnitude < scaling.min_abs) {
      scaling.min_abs = magnitude;
      first = false;
    }
  }
  return scaling;
}

double estimate_spectral_norm(const SparseMatrix& matrix, int iterations, unsigned seed) {
  const Index rows = matrix.num_rows();
  const Index cols = matrix.num_cols();
  if (rows == 0 || cols == 0 || matrix.num_nonzeros() == 0) return 1.0;

  // Power iteration on A^T A. A random start avoids the pathological case of a vector that
  // happens to be orthogonal to the dominant singular vector.
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> spread(-1.0, 1.0);
  std::vector<double> v(static_cast<std::size_t>(cols));
  for (double& value : v) value = spread(rng);

  std::vector<double> av(static_cast<std::size_t>(rows), 0.0);
  double norm = 0.0;
  for (int step = 0; step < iterations; ++step) {
    double length = 0.0;
    for (const double value : v) length += value * value;
    length = std::sqrt(length);
    if (length == 0.0) return 1.0;
    for (double& value : v) value /= length;

    matrix.multiply(v.data(), av.data());
    matrix.transpose_multiply(av.data(), v.data());

    double next = 0.0;
    for (const double value : v) next += value * value;
    norm = std::sqrt(std::sqrt(next));  // ||A^T A v|| ^ (1/2) approaches ||A||_2
  }

  // Round up. PDHG's convergence needs eta <= 1/||A||_2, so an UNDERESTIMATE of the norm
  // produces an oversized step and divergence; an overestimate merely costs iterations,
  // and the adaptive rule recovers those.
  return norm > 0.0 ? norm * 1.01 : 1.0;
}

}  // namespace sankhya::pdhg

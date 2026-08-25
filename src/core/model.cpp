// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Model and Solution implementation.
//
// recompute_quality() is the important function in this file. CLAUDE.md forbids reporting a
// number the solver merely believes; every engine calls this immediately before returning,
// so the infeasibility figures in the log and in the JSON blob are recomputed from the
// primal and dual vectors rather than accumulated during the solve. An engine that has
// drifted is caught by its own report.

#include "sankhya/model.hpp"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>

namespace sankhya {

const char* to_string(SolveStatus status) noexcept {
  switch (status) {
    case SolveStatus::kNotSolved: return "not_solved";
    case SolveStatus::kOptimal: return "optimal";
    case SolveStatus::kInfeasible: return "infeasible";
    case SolveStatus::kUnbounded: return "unbounded";
    case SolveStatus::kInfeasibleOrUnbounded: return "infeasible_or_unbounded";
    case SolveStatus::kFeasible: return "feasible";
    case SolveStatus::kIterationLimit: return "iteration_limit";
    case SolveStatus::kTimeLimit: return "time_limit";
    case SolveStatus::kNodeLimit: return "node_limit";
    case SolveStatus::kNumericalError: return "numerical_error";
    case SolveStatus::kModelError: return "model_error";
  }
  return "unknown";
}

const char* to_string(BasisStatus status) noexcept {
  switch (status) {
    case BasisStatus::kUnknown: return "unknown";
    case BasisStatus::kBasic: return "basic";
    case BasisStatus::kAtLower: return "at_lower";
    case BasisStatus::kAtUpper: return "at_upper";
    case BasisStatus::kNonbasicFree: return "free";
    case BasisStatus::kFixed: return "fixed";
  }
  return "unknown";
}

const char* to_string(VarType type) noexcept {
  switch (type) {
    case VarType::kContinuous: return "continuous";
    case VarType::kInteger: return "integer";
  }
  return "unknown";
}

// =========================================================================================
// Model
// =========================================================================================

bool Model::has_integrality() const noexcept {
  return std::any_of(col_type.begin(), col_type.end(),
                     [](VarType t) { return t == VarType::kInteger; });
}

Index Model::num_integer_columns() const noexcept {
  return static_cast<Index>(std::count(col_type.begin(), col_type.end(), VarType::kInteger));
}

bool Model::has_quadratic_objective() const noexcept {
  return hessian.num_nonzeros() > 0;
}

bool Model::is_fixed_column(Index j) const noexcept {
  return col_lower[static_cast<std::size_t>(j)] == col_upper[static_cast<std::size_t>(j)];
}

bool Model::is_equality_row(Index i) const noexcept {
  return row_lower[static_cast<std::size_t>(i)] == row_upper[static_cast<std::size_t>(i)];
}

void Model::resize_columns(Index n) {
  const auto u = static_cast<std::size_t>(n);
  col_cost.resize(u, 0.0);
  // The MPS default column bounds are [0, +inf). Matching that here means a reader only
  // has to write the bounds it actually sees in the BOUNDS section.
  col_lower.resize(u, 0.0);
  col_upper.resize(u, kInfinity);
  col_type.resize(u, VarType::kContinuous);
  if (!col_names.empty()) col_names.resize(u);
}

void Model::resize_rows(Index m) {
  const auto u = static_cast<std::size_t>(m);
  row_lower.resize(u, -kInfinity);
  row_upper.resize(u, kInfinity);
  if (!row_names.empty()) row_names.resize(u);
}

double Model::evaluate_objective(const double* x) const {
  double linear = objective_offset;
  const Index n = num_cols();
  for (Index j = 0; j < n; ++j) {
    linear += col_cost[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
  }
  if (!has_quadratic_objective()) return linear;

  // Q is stored lower-triangular including the diagonal, and the objective term is
  // 0.5 * x^T Q x with Q symmetric. Off-diagonal stored entries therefore each stand for
  // two entries of the full symmetric matrix, giving them a full weight of 1.0 while the
  // diagonal keeps its 0.5.
  double quadratic = 0.0;
  for (Index j = 0; j < hessian.num_cols(); ++j) {
    const ColumnView c = hessian.column(j);
    for (Index k = 0; k < c.size; ++k) {
      const Index i = c.rows[k];
      const double v = c.values[k];
      const double xi = x[static_cast<std::size_t>(i)];
      const double xj = x[static_cast<std::size_t>(j)];
      quadratic += (i == j) ? 0.5 * v * xi * xj : v * xi * xj;
    }
  }
  return linear + quadratic;
}

std::string Model::validate() const {
  const Index n = num_cols();
  const Index m = num_rows();

  if (static_cast<Index>(col_lower.size()) != n) return "col_lower length != num_cols";
  if (static_cast<Index>(col_upper.size()) != n) return "col_upper length != num_cols";
  if (static_cast<Index>(col_type.size()) != n) return "col_type length != num_cols";
  if (!col_names.empty() && static_cast<Index>(col_names.size()) != n) {
    return "col_names is neither empty nor num_cols long";
  }
  if (static_cast<Index>(row_upper.size()) != m) return "row_upper length != num_rows";
  if (!row_names.empty() && static_cast<Index>(row_names.size()) != m) {
    return "row_names is neither empty nor num_rows long";
  }

  if (!matrix.frozen()) return "constraint matrix is not finalized";
  if (matrix.num_rows() != m)
    return fmt::format("matrix has {} rows, model has {}", matrix.num_rows(), m);
  if (matrix.num_cols() != n)
    return fmt::format("matrix has {} columns, model has {}", matrix.num_cols(), n);

  // Coefficient VALUES, not just the shape. Bounds, costs and the offset were already
  // checked for NaN below; the matrix was the one numeric input nothing inspected, and it
  // is the worst place to leave unguarded. A NaN or infinite entry does not crash the
  // simplex - it propagates through the ratio test into a plausible-looking optimum for a
  // problem nobody posed. An infinity is equally meaningless as a coefficient: MPS uses
  // 1e30 to mean "no bound", which is a statement about a BOUND, never about an entry of A.
  //
  // This check also covers models built programmatically through the C API in Phase 10,
  // which never pass through a reader at all, so it belongs here rather than only in the
  // readers - validate() is the last gate before any engine sees the model.
  for (Index j = 0; j < n; ++j) {
    const ColumnView c = matrix.column(j);
    for (Index k = 0; k < c.size; ++k) {
      if (!std::isfinite(c.values[k])) {
        return fmt::format("matrix entry ({}, {}) is {}, which is not a usable coefficient",
                           c.rows[k], j, c.values[k]);
      }
    }
  }

  if (has_quadratic_objective()) {
    if (!hessian.frozen()) return "hessian is not finalized";
    if (hessian.num_rows() != n || hessian.num_cols() != n) {
      return "hessian is not num_cols x num_cols";
    }
    for (Index j = 0; j < hessian.num_cols(); ++j) {
      const ColumnView c = hessian.column(j);
      for (Index k = 0; k < c.size; ++k) {
        if (c.rows[k] < j) {
          return fmt::format(
              "hessian entry ({}, {}) is above the diagonal; only the lower "
              "triangle is stored",
              c.rows[k], j);
        }
        if (!std::isfinite(c.values[k])) {
          return fmt::format("hessian entry ({}, {}) is {}, which is not a usable coefficient",
                             c.rows[k], j, c.values[k]);
        }
      }
    }
  }

  for (Index j = 0; j < n; ++j) {
    const double lo = col_lower[static_cast<std::size_t>(j)];
    const double hi = col_upper[static_cast<std::size_t>(j)];
    if (std::isnan(lo) || std::isnan(hi)) return fmt::format("column {} has a NaN bound", j);
    if (lo > hi) {
      return fmt::format("column {} has lower bound {:g} above upper bound {:g}", j, lo, hi);
    }
    if (std::isnan(col_cost[static_cast<std::size_t>(j)])) {
      return fmt::format("column {} has a NaN objective coefficient", j);
    }
  }

  for (Index i = 0; i < m; ++i) {
    const double lo = row_lower[static_cast<std::size_t>(i)];
    const double hi = row_upper[static_cast<std::size_t>(i)];
    if (std::isnan(lo) || std::isnan(hi)) return fmt::format("row {} has a NaN bound", i);
    if (lo > hi) {
      return fmt::format("row {} has lower bound {:g} above upper bound {:g}", i, lo, hi);
    }
  }

  if (std::isnan(objective_offset)) return "objective offset is NaN";
  return {};
}

// =========================================================================================
// Solution
// =========================================================================================

void Solution::allocate_for(const Model& model) {
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  col_value.assign(n, 0.0);
  col_dual.assign(n, 0.0);
  col_status.assign(n, BasisStatus::kUnknown);
  row_activity.assign(m, 0.0);
  row_dual.assign(m, 0.0);
  row_status.assign(m, BasisStatus::kUnknown);
}

void Solution::recompute_quality(const Model& model) {
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  if (static_cast<Index>(col_value.size()) != n) return;

  // Row activities, recomputed from the matrix rather than carried along by the engine.
  row_activity.assign(static_cast<std::size_t>(m), 0.0);
  if (m > 0) model.matrix.multiply(col_value.data(), row_activity.data());

  primal_infeasibility = 0.0;
  integrality_violation = 0.0;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double x = col_value[u];
    if (is_finite_bound(model.col_lower[u])) {
      primal_infeasibility = std::max(primal_infeasibility, model.col_lower[u] - x);
    }
    if (is_finite_bound(model.col_upper[u])) {
      primal_infeasibility = std::max(primal_infeasibility, x - model.col_upper[u]);
    }
    if (model.col_type[u] == VarType::kInteger) {
      integrality_violation = std::max(integrality_violation, std::fabs(x - std::round(x)));
    }
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double a = row_activity[u];
    if (is_finite_bound(model.row_lower[u])) {
      primal_infeasibility = std::max(primal_infeasibility, model.row_lower[u] - a);
    }
    if (is_finite_bound(model.row_upper[u])) {
      primal_infeasibility = std::max(primal_infeasibility, a - model.row_upper[u]);
    }
  }

  objective = model.evaluate_objective(col_value.data());

  // Dual quality only when the engine produced duals of the right length.
  if (static_cast<Index>(row_dual.size()) == m && static_cast<Index>(col_dual.size()) == n) {
    dual_infeasibility = 0.0;
    complementarity_violation = 0.0;
    const double sense = model.sense_multiplier();
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      // Sign conditions for a minimization problem: d_j >= 0 at the lower bound, d_j <= 0
      // at the upper bound, d_j == 0 strictly between. sense folds a maximization model
      // into the same test.
      const double d = sense * col_dual[u];
      const double x = col_value[u];
      const double lo = model.col_lower[u];
      const double hi = model.col_upper[u];
      const bool at_lower = is_finite_bound(lo) && std::fabs(x - lo) <= tol::kPrimalFeasibility;
      const bool at_upper = is_finite_bound(hi) && std::fabs(x - hi) <= tol::kPrimalFeasibility;
      if (at_lower && at_upper) continue;  // fixed column: any reduced cost is admissible
      if (at_lower) {
        dual_infeasibility = std::max(dual_infeasibility, -d);
      } else if (at_upper) {
        dual_infeasibility = std::max(dual_infeasibility, d);
      } else {
        dual_infeasibility = std::max(dual_infeasibility, std::fabs(d));
      }
      if (!at_lower && !at_upper) {
        complementarity_violation = std::max(complementarity_violation, std::fabs(d));
      }
    }

    // The same conditions on the ROWS. Checking only the columns leaves half the KKT system
    // unmeasured: an engine could report a row price on a constraint that is not even
    // active, which is a genuine optimality failure, and this function would still print a
    // dual infeasibility of zero. Since these numbers are what the log, the .sol file and
    // the benchmark CSVs all quote, a half-measured self-report is worse than none - it
    // reads as a clean bill of health.
    for (Index i = 0; i < m; ++i) {
      const auto u = static_cast<std::size_t>(i);
      const double y = sense * row_dual[u];
      const double a = row_activity[u];
      const double lo = model.row_lower[u];
      const double hi = model.row_upper[u];
      const bool at_lower = is_finite_bound(lo) && std::fabs(a - lo) <= tol::kPrimalFeasibility;
      const bool at_upper = is_finite_bound(hi) && std::fabs(a - hi) <= tol::kPrimalFeasibility;
      if (at_lower && at_upper) continue;  // equality row: any multiplier is admissible
      if (at_lower) {
        dual_infeasibility = std::max(dual_infeasibility, -y);
      } else if (at_upper) {
        dual_infeasibility = std::max(dual_infeasibility, y);
      } else {
        // Slack row. Complementary slackness forces its price to zero.
        dual_infeasibility = std::max(dual_infeasibility, std::fabs(y));
        complementarity_violation = std::max(complementarity_violation, std::fabs(y));
      }
    }
  }

  absolute_gap = std::fabs(objective - dual_bound);
  const double scale = std::max(1.0, std::fabs(objective));
  relative_gap = absolute_gap / scale;
}

}  // namespace sankhya

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - checking the two certificates. See include/sankhya/certificate.hpp for what they
// are and why the solver checks its own.

#include "sankhya/certificate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

[[nodiscard]] bool finite(double value) noexcept {
  return std::isfinite(value);
}

void explain(std::string* why, std::string text) {
  if (why != nullptr) *why = std::move(text);
}

}  // namespace

bool farkas_proves_infeasible(const Model& model, const std::vector<double>& y,
                              std::string* why) {
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  if (static_cast<Index>(y.size()) != m) {
    explain(why, fmt::format("the certificate has {} multipliers for {} rows", y.size(), m));
    return false;
  }
  if (std::none_of(y.begin(), y.end(), [](double v) { return v != 0.0; })) {
    explain(why, "every multiplier is zero, which aggregates to 0 >= 0");
    return false;
  }

  // A multiplier may only lean on a bound the row actually has. Silently treating a missing
  // bound as zero would weaken the aggregate into something that proves nothing while still
  // looking like a proof.
  double required = 0.0;
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (y[u] > 0.0) {
      if (!finite(model.row_lower[u])) {
        explain(why, fmt::format("multiplier {:.6g} on row {} uses a lower bound the row does "
                                 "not have",
                                 y[u], i));
        return false;
      }
      required += y[u] * model.row_lower[u];
    } else if (y[u] < 0.0) {
      if (!finite(model.row_upper[u])) {
        explain(why, fmt::format("multiplier {:.6g} on row {} uses an upper bound the row "
                                 "does not have",
                                 y[u], i));
        return false;
      }
      required += y[u] * model.row_upper[u];
    }
  }

  // d = A'y, then the most the column box can give the aggregate.
  std::vector<double> d(static_cast<std::size_t>(n), 0.0);
  double scale = 0.0;
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(j);
    double sum = 0.0;
    for (Index k = 0; k < column.size; ++k) {
      const double term = column.values[k] * y[static_cast<std::size_t>(column.rows[k])];
      sum += term;
      scale = std::max(scale, std::fabs(term));
    }
    d[static_cast<std::size_t>(j)] = sum;
  }

  double reachable = 0.0;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    // A coefficient at rounding size is not evidence of anything; treating it as zero is the
    // conservative reading, because it can only make `reachable` smaller and so can only make
    // the proof harder to pass.
    if (d[u] > tol::kZeroDrop * std::max(1.0, scale)) {
      if (!finite(model.col_upper[u])) {
        explain(why, fmt::format("column {} is free upward and the aggregate leans on it, so "
                                 "the aggregate has no upper limit",
                                 j));
        return false;
      }
      reachable += d[u] * model.col_upper[u];
    } else if (d[u] < -tol::kZeroDrop * std::max(1.0, scale)) {
      if (!finite(model.col_lower[u])) {
        explain(why, fmt::format("column {} is free downward and the aggregate leans on it, "
                                 "so the aggregate has no upper limit",
                                 j));
        return false;
      }
      reachable += d[u] * model.col_lower[u];
    }
  }

  const double margin = required - reachable;
  const double comparison = std::max({1.0, std::fabs(required), std::fabs(reachable), scale});
  if (margin <= tol::kPrimalFeasibility * comparison) {
    explain(why, fmt::format("the rows aggregate to at least {:.12g} and the column bounds "
                             "allow {:.12g}, which is not a contradiction beyond rounding "
                             "(margin {:.3e})",
                             required, reachable, margin));
    return false;
  }
  explain(why, fmt::format("the rows aggregate to at least {:.12g} but the column bounds "
                           "allow at most {:.12g}",
                           required, reachable));
  return true;
}

bool ray_proves_unbounded(const Model& model, const std::vector<double>& d, std::string* why) {
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  if (static_cast<Index>(d.size()) != n) {
    explain(why, fmt::format("the ray has {} entries for {} columns", d.size(), n));
    return false;
  }
  double magnitude = 0.0;
  for (const double v : d) magnitude = std::max(magnitude, std::fabs(v));
  if (magnitude == 0.0) {
    explain(why, "the ray is all zeros, which is not a direction");
    return false;
  }
  // Scaled to the ray's own size: a component a billion times smaller than the largest is
  // rounding in the direction, not a move the bounds have to allow.
  const double moving = tol::kPrimalFeasibility * magnitude;

  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (d[u] > moving && finite(model.col_upper[u])) {
      explain(why, fmt::format("the ray raises column {}, which has an upper bound", j));
      return false;
    }
    if (d[u] < -moving && finite(model.col_lower[u])) {
      explain(why, fmt::format("the ray lowers column {}, which has a lower bound", j));
      return false;
    }
  }

  std::vector<double> activity(static_cast<std::size_t>(m), 0.0);
  double scale = 0.0;
  for (Index j = 0; j < n; ++j) {
    const double step = d[static_cast<std::size_t>(j)];
    if (step == 0.0) continue;
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const double term = column.values[k] * step;
      activity[static_cast<std::size_t>(column.rows[k])] += term;
      scale = std::max(scale, std::fabs(term));
    }
  }
  const double row_moving = tol::kPrimalFeasibility * std::max(1.0, scale);
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (activity[u] > row_moving && finite(model.row_upper[u])) {
      explain(why, fmt::format("the ray raises row {}, which has an upper bound", i));
      return false;
    }
    if (activity[u] < -row_moving && finite(model.row_lower[u])) {
      explain(why, fmt::format("the ray lowers row {}, which has a lower bound", i));
      return false;
    }
  }

  // In minimise space, whatever the model's sense.
  const double sense = model.sense_multiplier();
  double improvement = 0.0;
  double cost_scale = 0.0;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double term = sense * model.col_cost[u] * d[u];
    improvement += term;
    cost_scale = std::max(cost_scale, std::fabs(term));
  }
  if (improvement >= -tol::kPrimalFeasibility * std::max(1.0, cost_scale)) {
    explain(why, fmt::format("the objective changes by {:.12g} per unit step, which is not an "
                             "improvement",
                             improvement));
    return false;
  }

  // A quadratic objective curving back up makes the improvement local, not unbounded.
  // The Hessian is stored as its LOWER TRIANGLE, so an off-diagonal entry stands for two
  // entries of Q and contributes to both products - the same expansion src/qp/ uses.
  if (model.hessian.num_nonzeros() > 0) {
    double curvature = 0.0;
    for (Index j = 0; j < model.hessian.num_cols(); ++j) {
      const ColumnView column = model.hessian.column(j);
      const auto uj = static_cast<std::size_t>(j);
      for (Index k = 0; k < column.size; ++k) {
        const auto ui = static_cast<std::size_t>(column.rows[k]);
        curvature += column.values[k] * d[uj] * d[ui];
        if (column.rows[k] != j) curvature += column.values[k] * d[ui] * d[uj];
      }
    }
    if (curvature > tol::kPrimalFeasibility * std::max(1.0, std::fabs(curvature))) {
      explain(why, fmt::format("d'Qd = {:.12g} is positive, so the objective curves back up "
                               "along the ray",
                               curvature));
      return false;
    }
  }

  explain(why, fmt::format("the objective improves by {:.12g} per unit step and no bound "
                           "blocks the direction",
                           improvement));
  return true;
}

}  // namespace sankhya

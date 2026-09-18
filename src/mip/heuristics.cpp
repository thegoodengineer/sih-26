// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MILP primal heuristics (#290). References on the declarations.

#include "mip/heuristics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

double clamp_to(double value, double lower, double upper) {
  if (is_finite_bound(lower)) value = std::max(value, std::ceil(lower - 1e-9));
  if (is_finite_bound(upper)) value = std::min(value, std::floor(upper + 1e-9));
  return value;
}

/// How far `activity` is outside [lower, upper].
double violation(double activity, double lower, double upper) {
  double v = 0.0;
  if (is_finite_bound(lower)) v = std::max(v, lower - activity);
  if (is_finite_bound(upper)) v = std::max(v, activity - upper);
  return v;
}

}  // namespace

Locks compute_locks(const Model& model) {
  Locks locks;
  const auto n = static_cast<std::size_t>(model.num_cols());
  locks.up.assign(n, 0);
  locks.down.assign(n, 0);
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView col = model.matrix.column(j);
    const auto u = static_cast<std::size_t>(j);
    for (Index k = 0; k < col.size; ++k) {
      const auto i = static_cast<std::size_t>(col.rows[k]);
      const double a = col.values[k];
      if (a == 0.0) continue;
      // Moving the column up moves a positive-coefficient row's activity up, towards its
      // upper bound; a negative coefficient's down, towards its lower bound.
      const bool lower_matters = is_finite_bound(model.row_lower[i]);
      const bool upper_matters = is_finite_bound(model.row_upper[i]);
      if ((a > 0.0 && upper_matters) || (a < 0.0 && lower_matters)) ++locks.up[u];
      if ((a > 0.0 && lower_matters) || (a < 0.0 && upper_matters)) ++locks.down[u];
    }
  }
  return locks;
}

std::vector<double> lock_round(const Model& model, const Locks& locks,
                               const std::vector<Index>& integer_columns,
                               const std::vector<double>& x) {
  std::vector<double> rounded = x;
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    const double v = x[u];
    double r = std::round(v);
    if (std::fabs(v - r) > tol::kIntegrality) {
      // A direction no row locks can be taken without breaking any row that the LP point
      // satisfied; with both directions locked, nearest is as good a guess as any.
      if (locks.down[u] == 0) {
        r = std::floor(v);
      } else if (locks.up[u] == 0) {
        r = std::ceil(v);
      }
    }
    rounded[u] = clamp_to(r, model.col_lower[u], model.col_upper[u]);
  }
  return rounded;
}

bool repair(const Model& model, const std::vector<Index>& integer_columns,
            std::vector<double>* x_io, int max_moves, double tolerance, Count* moves) {
  std::vector<double>& x = *x_io;
  *moves = 0;
  const auto m = static_cast<std::size_t>(model.num_rows());
  std::vector<double> activity(m, 0.0);
  model.matrix.multiply_add(x.data(), activity.data());
  std::vector<double> v(m, 0.0);
  double total = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    v[i] = violation(activity[i], model.row_lower[i], model.row_upper[i]);
    total += v[i];
  }
  std::vector<char> is_integer(static_cast<std::size_t>(model.num_cols()), 0);
  for (const Index j : integer_columns) is_integer[static_cast<std::size_t>(j)] = 1;
  const CsrView by_row(model.matrix);
  const double sense = model.sense_multiplier();

  while (*moves < max_moves) {
    std::size_t worst = m;
    double worst_v = tolerance;
    for (std::size_t i = 0; i < m; ++i) {
      if (v[i] > worst_v) {
        worst_v = v[i];
        worst = i;
      }
    }
    if (worst == m) return true;  // every row within tolerance

    // The best one-unit shift of an integer column in the worst row: the most violation
    // removed over ALL rows the column touches, then the smallest objective cost, then the
    // lowest index - so the result does not depend on anything but the model and the point.
    Index best_column = -1;
    double best_step = 0.0;
    double best_total = total;
    double best_cost = std::numeric_limits<double>::infinity();
    const ColumnView row = by_row.row(static_cast<Index>(worst));
    for (Index k = 0; k < row.size; ++k) {
      const Index j = row.rows[k];
      const auto u = static_cast<std::size_t>(j);
      if (is_integer[u] == 0) continue;
      for (const double step : {1.0, -1.0}) {
        const double moved = x[u] + step;
        if (is_finite_bound(model.col_lower[u]) && moved < model.col_lower[u] - 1e-9) continue;
        if (is_finite_bound(model.col_upper[u]) && moved > model.col_upper[u] + 1e-9) continue;
        double after = total;
        const ColumnView col = model.matrix.column(j);
        for (Index t = 0; t < col.size; ++t) {
          const auto i = static_cast<std::size_t>(col.rows[t]);
          after -= v[i];
          after += violation(activity[i] + step * col.values[t], model.row_lower[i],
                             model.row_upper[i]);
        }
        const double cost = sense * model.col_cost[u] * step;
        const bool better = after < best_total - 1e-12 ||
                            (after <= best_total + 1e-12 && best_column >= 0 &&
                             (cost < best_cost || (cost == best_cost && j < best_column)));
        if (better && after < total - 1e-12) {
          best_column = j;
          best_step = step;
          best_total = after;
          best_cost = cost;
        }
      }
    }
    if (best_column < 0) return false;  // stuck: no shift in the worst row helps

    const auto u = static_cast<std::size_t>(best_column);
    x[u] += best_step;
    const ColumnView col = model.matrix.column(best_column);
    for (Index t = 0; t < col.size; ++t) {
      const auto i = static_cast<std::size_t>(col.rows[t]);
      activity[i] += best_step * col.values[t];
      total -= v[i];
      v[i] = violation(activity[i], model.row_lower[i], model.row_upper[i]);
      total += v[i];
    }
    ++*moves;
  }
  // Out of moves: the same test the loop makes, on the rows as they now stand.
  return std::all_of(v.begin(), v.end(), [&](double value) { return value <= tolerance; });
}

bool rins_submodel(const Model& model, const std::vector<Index>& integer_columns,
                   const std::vector<double>& relaxation, const std::vector<double>& incumbent,
                   double min_fixed_fraction, double tolerance, Model* out, Count* fixed) {
  *fixed = 0;
  std::vector<Index> agree;
  for (const Index j : integer_columns) {
    const auto u = static_cast<std::size_t>(j);
    if (std::fabs(relaxation[u] - incumbent[u]) <= tolerance) agree.push_back(j);
  }
  if (integer_columns.empty() ||
      static_cast<double>(agree.size()) <
          min_fixed_fraction * static_cast<double>(integer_columns.size())) {
    return false;
  }
  *out = model;
  for (const Index j : agree) {
    const auto u = static_cast<std::size_t>(j);
    const double value = std::round(incumbent[u]);
    out->col_lower[u] = value;
    out->col_upper[u] = value;
  }
  *fixed = static_cast<Count>(agree.size());
  return true;
}

std::vector<double> feasibility_pump(const Model& model,
                                     const std::vector<Index>& integer_columns,
                                     const std::vector<double>& start,
                                     const Options& lp_options, int max_rounds,
                                     double integrality_tolerance, Count* lp_solves) {
  *lp_solves = 0;
  if (integer_columns.empty()) return {};
  // The projection LP: the model's rows and bounds, every column continuous, and an
  // objective replaced round by round with the L1 distance to the current rounding.
  Model lp = model;
  std::fill(lp.col_type.begin(), lp.col_type.end(), VarType::kContinuous);
  lp.sense = ObjSense::kMinimize;
  lp.objective_offset = 0.0;
  lp.hessian.reset(model.num_cols(), model.num_cols());
  lp.hessian.finalize();

  std::vector<double> x = start;
  std::vector<double> previous_rounding;
  for (int round = 0; round < max_rounds; ++round) {
    // Round, and test whether the LP point is already integral on the integer columns.
    std::vector<double> rounding(x.size(), 0.0);
    bool integral = true;
    for (const Index j : integer_columns) {
      const auto u = static_cast<std::size_t>(j);
      rounding[u] = clamp_to(std::round(x[u]), model.col_lower[u], model.col_upper[u]);
      integral = integral && std::fabs(x[u] - rounding[u]) <= integrality_tolerance;
    }
    if (integral) {
      // The LP point is feasible (the LP said so) and integral where it must be: done. The
      // integer columns are snapped to their integers so the caller's check sees them exact.
      for (const Index j : integer_columns) {
        x[static_cast<std::size_t>(j)] = rounding[static_cast<std::size_t>(j)];
      }
      return x;
    }
    // A CYCLE - the same rounding as last round - is broken by flipping the columns whose
    // LP value is furthest from their rounding, the pump's standard perturbation. Ordered by
    // that distance and then by index, so a rerun makes the same flips.
    if (rounding == previous_rounding) {
      std::vector<std::pair<double, Index>> far;
      for (const Index j : integer_columns) {
        const auto u = static_cast<std::size_t>(j);
        far.emplace_back(-std::fabs(x[u] - rounding[u]), j);
      }
      std::sort(far.begin(), far.end());
      const std::size_t flips = std::max<std::size_t>(1, integer_columns.size() / 10);
      for (std::size_t k = 0; k < std::min(flips, far.size()); ++k) {
        const auto u = static_cast<std::size_t>(far[k].second);
        const double away = x[u] > rounding[u] ? rounding[u] + 1.0 : rounding[u] - 1.0;
        rounding[u] = clamp_to(away, model.col_lower[u], model.col_upper[u]);
      }
    }
    previous_rounding = rounding;

    // The L1 distance to the rounding is linear in x for a column rounded to one of its
    // bounds: x - l at the lower, u - x at the upper. A general integer rounded strictly
    // between its bounds would need an auxiliary column and is left out of the distance.
    std::fill(lp.col_cost.begin(), lp.col_cost.end(), 0.0);
    for (const Index j : integer_columns) {
      const auto u = static_cast<std::size_t>(j);
      if (is_finite_bound(model.col_lower[u]) && rounding[u] <= model.col_lower[u] + 1e-9) {
        lp.col_cost[u] = 1.0;
      } else if (is_finite_bound(model.col_upper[u]) &&
                 rounding[u] >= model.col_upper[u] - 1e-9) {
        lp.col_cost[u] = -1.0;
      }
    }
    const Solution projected = solve(lp, lp_options);
    ++*lp_solves;
    if (projected.status != SolveStatus::kOptimal) return {};
    x = projected.col_value;
  }
  return {};
}

}  // namespace sankhya::mip

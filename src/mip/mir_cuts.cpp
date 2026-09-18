// SPDX-License-Identifier: Apache-2.0
// SANKHYA - mixed-integer rounding cuts (#221). See mir_cuts.hpp for the inequality and
// the citations.
//
// AGGREGATION. A single model row rarely yields a useful MIR cut when it carries continuous
// variables that sit strictly inside their bounds at the LP point: the bound substitution
// leaves them with a positive slack that the rounding either drops (a positive coefficient)
// or scales up (a negative one), and the cut is weak or not violated at all. Marchand &
// Wolsey's remedy is to ELIMINATE such a variable by adding a multiple of another row that
// contains it, and to try the MIR inequality on every intermediate aggregate, up to a small
// depth. The aggregate is a valid inequality because each row is added with the sign its
// own bound permits (a multiple lambda > 0 of `sum a x <= u`, or lambda < 0 of
// `sum a x >= l`, both of which are `<=` inequalities after the multiplication).

#include "mir_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

/// f0 outside this band gives a cut that is either numerically meaningless (f0 near 0: the
/// rounding barely moves the right-hand side while 1/(1 - f0) stays finite) or dominated by
/// the base inequality (f0 near 1: 1/(1 - f0) blows the continuous coefficient up).
constexpr double kMinFraction = 0.01;
/// Violation, relative to the cut's right-hand side, below which the cut is not worth a row.
constexpr double kMinViolation = 1e-4;
/// Largest ratio between the largest and smallest coefficient a cut may carry (the issue's
/// numerical filter, applied here as well as in the shared filter so a bad divisor is not
/// even proposed).
constexpr double kMaxDynamism = 1e6;
/// How many rows may be added to a starting row before the search gives up on it
/// (Marchand & Wolsey use 6; the issue names the same number).
constexpr int kMaxAggregation = 6;
/// A continuous variable is "inside" its bounds, and so worth eliminating, when its slack to
/// the nearer bound exceeds this.
constexpr double kInsideTolerance = 1e-6;

[[nodiscard]] double fractional_part(double v) {
  return v - std::floor(v);
}

[[nodiscard]] bool integral(double v) {
  return std::fabs(v - std::round(v)) <= tol::kIntegrality;
}

/// One row of the model in row-wise form: (column, coefficient) pairs.
struct RowEntries {
  std::vector<Index> columns;
  std::vector<double> values;
};

std::vector<RowEntries> rows_of(const Model& model) {
  std::vector<RowEntries> rows(static_cast<std::size_t>(model.num_rows()));
  for (Index j = 0; j < model.num_cols(); ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      RowEntries& row = rows[static_cast<std::size_t>(column.rows[k])];
      row.columns.push_back(j);
      row.values.push_back(column.values[k]);
    }
  }
  return rows;
}

/// A base inequality sum coef_j x_j <= rhs over the model's columns, sparse.
struct BaseRow {
  std::vector<Index> columns;
  std::vector<double> values;
  double rhs = 0.0;
};

/// The MIR separation on one base inequality at the LP point, with the bounds to
/// substitute given explicitly. Returns the most violated member of the divisor family in
/// the original variables, or nothing.
std::optional<Cut> separate_from_base(const Model& model, const Solution& solution,
                                      const std::vector<double>& col_lower,
                                      const std::vector<double>& col_upper,
                                      const BaseRow& base) {
  const Index n = model.num_cols();
  std::vector<Index> live_columns;  // the base's columns with a nonzero coefficient
  std::vector<double> a;            // coefficient on y_j in the base inequality
  std::vector<bool> is_int;         // y_j integer
  std::vector<bool> complemented;   // y_j = u_j - x_j (else y_j = x_j - l_j)
  std::vector<double> y_star;       // LP value of y_j
  std::vector<double> rounded;
  double b = base.rhs;
  bool any_fractional_integer = false;
  for (std::size_t k = 0; k < base.columns.size(); ++k) {
    const Index j = base.columns[k];
    const auto uj = static_cast<std::size_t>(j);
    const double coef = base.values[k];
    if (std::fabs(coef) <= tol::kZeroDrop) continue;
    const double lo = col_lower[uj];
    const double hi = col_upper[uj];
    const double x = solution.col_value[uj];
    const bool integer_col = model.col_type[uj] == VarType::kInteger;
    const bool has_lo = is_finite_bound(lo);
    const bool has_hi = is_finite_bound(hi);
    if (!has_lo && !has_hi) return std::nullopt;  // a free variable has no substitute
    // Substitute the bound the LP point is nearer to, so y* is small and the cut is
    // measured where the point actually sits.
    const bool use_upper = has_hi && (!has_lo || hi - x < x - lo);
    if (integer_col && !integral(use_upper ? hi : lo)) return std::nullopt;
    live_columns.push_back(j);
    if (use_upper) {
      // x = hi - y:  coef * x = coef * hi - coef * y
      b -= coef * hi;
      a.push_back(-coef);
      complemented.push_back(true);
      y_star.push_back(hi - x);
    } else {
      b -= coef * lo;
      a.push_back(coef);
      complemented.push_back(false);
      y_star.push_back(x - lo);
    }
    is_int.push_back(integer_col);
    if (integer_col && !integral(x)) any_fractional_integer = true;
  }
  if (!any_fractional_integer || !std::isfinite(b)) return std::nullopt;

  // Divisors: 1, and the coefficient magnitudes of the fractional integer variables.
  std::vector<double> divisors{1.0};
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (is_int[k] && !integral(y_star[k]) && std::fabs(a[k]) > tol::kZeroDrop) {
      divisors.push_back(std::fabs(a[k]));
    }
  }
  double best_violation = kMinViolation;
  std::optional<Cut> best;
  for (const double divisor : divisors) {
    double cut_rhs = 0.0;
    if (!mir_inequality(a, is_int, b, divisor, &rounded, &cut_rhs)) continue;
    // Violation at y*: lhs - rhs, relative to max(1, |rhs|).
    double lhs = 0.0;
    double largest = 0.0;
    double smallest = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < rounded.size(); ++k) {
      lhs += rounded[k] * y_star[k];
      const double mag = std::fabs(rounded[k]);
      if (mag > tol::kZeroDrop) {
        largest = std::max(largest, mag);
        smallest = std::min(smallest, mag);
      }
    }
    if (largest == 0.0 || largest / smallest > kMaxDynamism) continue;
    const double violation = (lhs - cut_rhs) / std::max(1.0, std::fabs(cut_rhs));
    if (violation <= best_violation) continue;
    // Back to x: y = x - lo  or  y = hi - x, with the constants moved to the right.
    Cut cut;
    cut.coeff.assign(static_cast<std::size_t>(n), 0.0);
    double rhs_x = cut_rhs;
    for (std::size_t k = 0; k < rounded.size(); ++k) {
      const double c = rounded[k];
      if (std::fabs(c) <= tol::kZeroDrop) continue;
      const auto uj = static_cast<std::size_t>(live_columns[k]);
      if (complemented[k]) {
        cut.coeff[uj] -= c;
        rhs_x -= c * col_upper[uj];
      } else {
        cut.coeff[uj] += c;
        rhs_x += c * col_lower[uj];
      }
    }
    cut.rhs = rhs_x;
    best_violation = violation;
    best = std::move(cut);
  }
  return best;
}

/// The continuous column of `base` that sits furthest inside its bounds at the LP point,
/// or -1 when every continuous column is at a bound (nothing to eliminate).
Index column_to_eliminate(const Model& model, const Solution& solution,
                          const std::vector<double>& col_lower,
                          const std::vector<double>& col_upper, const BaseRow& base) {
  Index chosen = -1;
  double deepest = kInsideTolerance;
  for (std::size_t k = 0; k < base.columns.size(); ++k) {
    if (std::fabs(base.values[k]) <= tol::kZeroDrop) continue;
    const Index j = base.columns[k];
    const auto uj = static_cast<std::size_t>(j);
    if (model.col_type[uj] == VarType::kInteger) continue;
    const double x = solution.col_value[uj];
    double slack = std::numeric_limits<double>::infinity();
    if (is_finite_bound(col_lower[uj])) slack = std::min(slack, x - col_lower[uj]);
    if (is_finite_bound(col_upper[uj])) slack = std::min(slack, col_upper[uj] - x);
    if (std::isinf(slack)) continue;  // a free continuous column: the base is unusable anyway
    if (slack > deepest) {
      deepest = slack;
      chosen = j;
    }
  }
  return chosen;
}

/// Add lambda times row i (as `sum a x <= u` for lambda > 0, `sum a x >= l` for lambda < 0)
/// to `base`, so that column `eliminate` cancels. Returns false when row i has no finite
/// bound on the side the sign needs.
bool aggregate_row(const Model& model, const std::vector<RowEntries>& rows, Index i,
                   Index eliminate, BaseRow* base) {
  const RowEntries& row = rows[static_cast<std::size_t>(i)];
  double coef_in_row = 0.0;
  for (std::size_t k = 0; k < row.columns.size(); ++k) {
    if (row.columns[k] == eliminate) coef_in_row = row.values[k];
  }
  if (std::fabs(coef_in_row) <= tol::kZeroDrop) return false;
  double coef_in_base = 0.0;
  for (std::size_t k = 0; k < base->columns.size(); ++k) {
    if (base->columns[k] == eliminate) coef_in_base = base->values[k];
  }
  const double lambda = -coef_in_base / coef_in_row;
  const auto ui = static_cast<std::size_t>(i);
  const double bound = lambda > 0.0 ? model.row_upper[ui] : model.row_lower[ui];
  if (!is_finite_bound(bound)) return false;
  // Merge: base += lambda * row.
  std::unordered_map<Index, std::size_t> position;
  for (std::size_t k = 0; k < base->columns.size(); ++k) position[base->columns[k]] = k;
  for (std::size_t k = 0; k < row.columns.size(); ++k) {
    const Index j = row.columns[k];
    const double add = lambda * row.values[k];
    auto it = position.find(j);
    if (it == position.end()) {
      base->columns.push_back(j);
      base->values.push_back(add);
    } else {
      base->values[it->second] += add;
    }
  }
  base->rhs += lambda * bound;
  // The eliminated column is exactly cancelled in exact arithmetic; make it so here.
  for (std::size_t k = 0; k < base->columns.size(); ++k) {
    if (base->columns[k] == eliminate) base->values[k] = 0.0;
  }
  return true;
}

}  // namespace

bool mir_inequality(const std::vector<double>& coefficient, const std::vector<bool>& is_integer,
                    double rhs, double divisor, std::vector<double>* cut_coefficient,
                    double* cut_rhs) {
  if (!(divisor > 0.0) || !std::isfinite(divisor)) return false;
  const double b = rhs / divisor;
  const double f0 = fractional_part(b);
  if (f0 < kMinFraction || f0 > 1.0 - kMinFraction) return false;
  const double one_minus_f0 = 1.0 - f0;
  cut_coefficient->assign(coefficient.size(), 0.0);
  for (std::size_t j = 0; j < coefficient.size(); ++j) {
    const double a = coefficient[j] / divisor;
    if (is_integer[j]) {
      const double fj = fractional_part(a);
      (*cut_coefficient)[j] = std::floor(a) + std::max(0.0, fj - f0) / one_minus_f0;
    } else {
      // A continuous term with a negative coefficient is the slack s the inequality allows
      // on the right-hand side, scaled by 1/(1 - f0); one with a positive coefficient only
      // strengthens the base inequality when dropped, so it is dropped (coefficient 0).
      (*cut_coefficient)[j] = a < 0.0 ? a / one_minus_f0 : 0.0;
    }
  }
  *cut_rhs = std::floor(b);
  return true;
}

std::vector<Cut> generate_mir_cuts(const Model& model, const Solution& solution) {
  return generate_mir_cuts(model, solution, model.col_lower, model.col_upper, nullptr);
}

std::vector<Cut> generate_mir_cuts(const Model& model, const Solution& solution,
                                   const std::vector<double>& col_lower,
                                   const std::vector<double>& col_upper) {
  return generate_mir_cuts(model, solution, col_lower, col_upper, nullptr);
}

std::vector<Cut> generate_mir_cuts(const Model& model, const Solution& solution,
                                   const std::vector<double>& col_lower,
                                   const std::vector<double>& col_upper, MirStats* stats) {
  std::vector<Cut> cuts;
  const Index n = model.num_cols();
  if (static_cast<Index>(solution.col_value.size()) != n || n == 0) return cuts;
  if (static_cast<Index>(col_lower.size()) != n || static_cast<Index>(col_upper.size()) != n) {
    return cuts;
  }
  const std::vector<RowEntries> rows = rows_of(model);

  // Which rows contain a fractional integer column: only those can start an aggregation.
  std::vector<char> row_has_fractional(static_cast<std::size_t>(model.num_rows()), 0);
  for (Index j = 0; j < n; ++j) {
    const auto uj = static_cast<std::size_t>(j);
    if (model.col_type[uj] != VarType::kInteger || integral(solution.col_value[uj])) continue;
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      row_has_fractional[static_cast<std::size_t>(column.rows[k])] = 1;
    }
  }

  std::vector<char> used(static_cast<std::size_t>(model.num_rows()), 0);
  for (Index i = 0; i < model.num_rows(); ++i) {
    const RowEntries& row = rows[static_cast<std::size_t>(i)];
    if (row.columns.empty() || !row_has_fractional[static_cast<std::size_t>(i)]) continue;
    const auto ui = static_cast<std::size_t>(i);
    // Both senses of the row are base inequalities: sum a x <= upper, and sum -a x <= -lower.
    for (int side = 0; side < 2; ++side) {
      const double bound = side == 0 ? model.row_upper[ui] : model.row_lower[ui];
      if (!is_finite_bound(bound)) continue;
      const double sign = side == 0 ? 1.0 : -1.0;
      BaseRow base;
      base.columns = row.columns;
      base.values.reserve(row.values.size());
      for (const double v : row.values) base.values.push_back(sign * v);
      base.rhs = sign * bound;

      std::fill(used.begin(), used.end(), 0);
      used[ui] = 1;
      std::optional<Cut> best;
      double best_violation = 0.0;
      int best_depth = 0;
      for (int depth = 0; depth <= kMaxAggregation; ++depth) {
        if (std::optional<Cut> cut =
                separate_from_base(model, solution, col_lower, col_upper, base)) {
          double lhs = 0.0;
          for (Index j = 0; j < n; ++j) {
            const auto uj = static_cast<std::size_t>(j);
            lhs += cut->coeff[uj] * solution.col_value[uj];
          }
          const double violation = (lhs - cut->rhs) / std::max(1.0, std::fabs(cut->rhs));
          if (!best || violation > best_violation) {
            best = std::move(cut);
            best_violation = violation;
            best_depth = depth;
          }
        }
        if (depth == kMaxAggregation) break;
        // Eliminate the continuous column furthest inside its bounds with an unused row that
        // contains it and has a finite bound on the side the multiplier needs.
        const Index eliminate =
            column_to_eliminate(model, solution, col_lower, col_upper, base);
        if (eliminate < 0) break;
        const ColumnView column = model.matrix.column(eliminate);
        bool extended = false;
        for (Index k = 0; k < column.size && !extended; ++k) {
          const Index other = column.rows[k];
          const auto uo = static_cast<std::size_t>(other);
          if (used[uo]) continue;
          BaseRow trial = base;
          if (!aggregate_row(model, rows, other, eliminate, &trial)) continue;
          used[uo] = 1;
          base = std::move(trial);
          extended = true;
        }
        if (!extended) break;
      }
      if (best) {
        if (stats != nullptr) {
          ++stats->cuts;
          if (best_depth > 0) ++stats->aggregated_cuts;
          stats->deepest = std::max(stats->deepest, best_depth);
        }
        cuts.push_back(std::move(*best));
      }
    }
  }
  return cuts;
}

}  // namespace sankhya::mip

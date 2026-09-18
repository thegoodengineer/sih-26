// SPDX-License-Identifier: Apache-2.0
// SANKHYA - mixed-integer rounding cuts (#221). See mir_cuts.hpp for the inequality and
// the citations.

#include "mir_cuts.hpp"

#include <algorithm>
#include <cmath>
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

[[nodiscard]] double fractional_part(double v) {
  return v - std::floor(v);
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
  std::vector<Cut> cuts;
  const Index n = model.num_cols();
  if (static_cast<Index>(solution.col_value.size()) != n || n == 0) return cuts;
  const std::vector<RowEntries> rows = rows_of(model);
  const auto integral = [](double v) {
    return std::fabs(v - std::round(v)) <= tol::kIntegrality;
  };

  // Working arrays over the row's own entries.
  std::vector<double> a;           // coefficient on y_j in the base inequality
  std::vector<bool> is_int;        // y_j integer
  std::vector<bool> complemented;  // y_j = u_j - x_j (else y_j = x_j - l_j)
  std::vector<double> y_star;      // LP value of y_j
  std::vector<double> rounded;
  for (Index i = 0; i < model.num_rows(); ++i) {
    const RowEntries& row = rows[static_cast<std::size_t>(i)];
    if (row.columns.empty()) continue;
    const auto ui = static_cast<std::size_t>(i);
    // Both senses of the row are base inequalities: sum a x <= upper, and sum -a x <= -lower.
    for (int side = 0; side < 2; ++side) {
      const double bound = side == 0 ? model.row_upper[ui] : model.row_lower[ui];
      if (!is_finite_bound(bound)) continue;
      const double sign = side == 0 ? 1.0 : -1.0;
      double b = sign * bound;
      a.clear();
      is_int.clear();
      complemented.clear();
      y_star.clear();
      bool usable = true;
      bool any_fractional_integer = false;
      for (std::size_t k = 0; k < row.columns.size() && usable; ++k) {
        const Index j = row.columns[k];
        const auto uj = static_cast<std::size_t>(j);
        const double coef = sign * row.values[k];
        const double lo = model.col_lower[uj];
        const double hi = model.col_upper[uj];
        const double x = solution.col_value[uj];
        const bool integer_col = model.col_type[uj] == VarType::kInteger;
        const bool has_lo = is_finite_bound(lo);
        const bool has_hi = is_finite_bound(hi);
        if (!has_lo && !has_hi) {
          usable = false;  // a free variable has no non-negative substitute
          break;
        }
        // Substitute the bound the LP point is nearer to, so y* is small and the cut is
        // measured where the point actually sits.
        const bool use_upper = has_hi && (!has_lo || hi - x < x - lo);
        if (integer_col && !integral(use_upper ? hi : lo)) {
          usable = false;  // a non-integral bound on an integer column: y would not be integer
          break;
        }
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
      if (!usable || !any_fractional_integer || !std::isfinite(b)) continue;

      // Divisors: 1, and the coefficient magnitudes of the fractional integer variables.
      std::vector<double> divisors{1.0};
      for (std::size_t k = 0; k < a.size(); ++k) {
        if (is_int[k] && !integral(y_star[k]) && std::fabs(a[k]) > tol::kZeroDrop) {
          divisors.push_back(std::fabs(a[k]));
        }
      }
      double best_violation = kMinViolation;
      Cut best;
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
          const auto uj = static_cast<std::size_t>(row.columns[k]);
          if (complemented[k]) {
            cut.coeff[uj] -= c;
            rhs_x -= c * model.col_upper[uj];
          } else {
            cut.coeff[uj] += c;
            rhs_x += c * model.col_lower[uj];
          }
        }
        cut.rhs = rhs_x;
        best_violation = violation;
        best = std::move(cut);
      }
      if (!best.coeff.empty()) cuts.push_back(std::move(best));
    }
  }
  return cuts;
}

}  // namespace sankhya::mip

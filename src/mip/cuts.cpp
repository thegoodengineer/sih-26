// SPDX-License-Identifier: Apache-2.0
// SANKHYA - cutting planes. See cuts.hpp for the correctness obligations this file carries.

#include "cuts.hpp"

#include <cmath>
#include <vector>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

/// Is `value` an integer, to within the integrality tolerance?
///
/// The tolerance is deliberately the same one the solver uses to decide whether a variable is
/// integral. A coefficient this routine wrongly accepts as an integer makes the row's
/// activity non-integral in truth, and the rounding below then removes feasible points - the
/// exact failure cuts.hpp exists to prevent. Anything the solver would not call an integer,
/// this must not either.
[[nodiscard]] bool is_integral(double value) noexcept {
  return std::fabs(value - std::round(value)) <= tol::kIntegrality;
}

}  // namespace

RowTightening tighten_integral_rows(Model* model, Logger& logger) {
  RowTightening result;
  if (model == nullptr) return result;

  const Index rows = model->num_rows();
  const Index cols = model->num_cols();
  if (rows == 0 || cols == 0 || !model->has_integrality()) return result;

  // Walk the matrix column-wise, since that is how it is stored, and accumulate per row:
  // whether every entry so far belongs to an integer column with an integral coefficient.
  // Rows with no entries stay `true` and are skipped afterwards - an empty row's activity is
  // zero, which is integral, but rounding its bounds changes nothing and presolve has
  // already removed it.
  std::vector<char> eligible(static_cast<std::size_t>(rows), 1);
  std::vector<char> has_entry(static_cast<std::size_t>(rows), 0);

  for (Index j = 0; j < cols; ++j) {
    const ColumnView column = model->matrix.column(j);
    const bool integer_column =
        model->col_type[static_cast<std::size_t>(j)] == VarType::kInteger;
    for (Index k = 0; k < column.size; ++k) {
      const auto row = static_cast<std::size_t>(column.rows[k]);
      has_entry[row] = 1;
      if (!integer_column || !is_integral(column.values[k])) eligible[row] = 0;
    }
  }

  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (!eligible[u] || has_entry[u] == 0) continue;

    double& lower = model->row_lower[u];
    double& upper = model->row_upper[u];
    bool moved = false;

    // Round INWARD on both sides. Outward would relax the row, which is not what this is for
    // and would be a silent correctness change in the other direction.
    //
    // The tolerance is applied before rounding so that a bound already sitting on an integer,
    // give or take floating-point noise, is not pushed a whole unit by its own representation
    // error: ceil(3.0000000001) is 4, and that would cut off the perfectly feasible point
    // where the row is tight at 3.
    if (is_finite_bound(lower)) {
      const double rounded = std::ceil(lower - tol::kIntegrality);
      if (rounded > lower) {
        lower = rounded;
        ++result.bounds_moved;
        moved = true;
      }
    }
    if (is_finite_bound(upper)) {
      const double rounded = std::floor(upper + tol::kIntegrality);
      if (rounded < upper) {
        upper = rounded;
        ++result.bounds_moved;
        moved = true;
      }
    }

    if (moved) ++result.rows_tightened;

    // Rounding can only make an infeasible row visibly infeasible; it cannot create
    // infeasibility that was not already there, because every integer-feasible point
    // satisfies the rounded bounds. Saying so is still worth a line, because "the model
    // became infeasible after cuts" is the first thing anyone will suspect.
    if (is_finite_bound(lower) && is_finite_bound(upper) && lower > upper) {
      logger.verbose(
          "row {} has no integral activity in [{:g}, {:g}]; it was already infeasible for "
          "integer points and the rounding has made that explicit",
          i, model->row_lower[u], model->row_upper[u]);
    }
  }

  if (result.rows_tightened > 0) {
    logger.info("Integer rounding: tightened {} row bound(s) on {} row(s)", result.bounds_moved,
                result.rows_tightened);
  }
  return result;
}

}  // namespace sankhya::mip

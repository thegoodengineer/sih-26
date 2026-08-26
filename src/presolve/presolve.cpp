// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve reductions. See presolve.hpp for the references and the rationale.

#include "presolve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <fmt/format.h>

#include "sankhya/tolerances.hpp"

namespace sankhya::presolve {
namespace {

/// Row activity limits implied by the current column bounds.
///
/// This is the workhorse: a row whose reachable activity already lies inside its own bounds
/// cannot constrain anything and can go, and a row whose bound sits exactly at a reachable
/// extreme forces every variable in it to a bound. Both come straight out of Brearley et al.
struct ActivityBounds {
  double lower = 0.0;
  double upper = 0.0;
  bool lower_finite = true;
  bool upper_finite = true;
};

/// Working state. Rows and columns are marked dead rather than compacted as we go, because
/// a reduction that fires halfway through a pass must not invalidate the indices the rest of
/// the pass is iterating over. Compaction happens once, at the end.
struct Workspace {
  const Model* original = nullptr;
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::vector<bool> col_dead;
  std::vector<bool> row_dead;
  /// Entries of each row, as (column, coefficient). The Model stores columns, and every
  /// reduction here asks row-wise questions, so this is built once up front.
  std::vector<std::vector<std::pair<Index, double>>> rows;
  std::vector<Index> col_count;  ///< live entries per column
  std::vector<Index> row_count;  ///< live entries per row
};

[[nodiscard]] bool finite(double v) {
  return std::fabs(v) < kInfinity;
}

/// Round a derived bound INWARD for an integer column.
///
/// Outward would be the dangerous direction: widening an integer variable's box cannot make
/// the relaxation wrong, but narrowing it by even a hair past a feasible integer removes
/// that point from the problem, and branch and bound then proves the second-best answer
/// optimal without any symptom. floor/ceil with a tolerance is the standard treatment
/// (Achterberg et al.); the tolerance keeps 2.9999999997 from becoming 2.
[[nodiscard]] double round_integer_lower(double value) {
  return std::ceil(value - tol::kIntegrality);
}
[[nodiscard]] double round_integer_upper(double value) {
  return std::floor(value + tol::kIntegrality);
}

[[nodiscard]] ActivityBounds activity_bounds(const Workspace& work, Index row) {
  ActivityBounds bounds;
  for (const auto& [column, coefficient] : work.rows[static_cast<std::size_t>(row)]) {
    const auto u = static_cast<std::size_t>(column);
    if (work.col_dead[u]) continue;
    const double lo = work.col_lower[u];
    const double up = work.col_upper[u];
    // The bound that minimises a * x depends on the sign of a, which is the only subtlety
    // here and the easiest thing to get backwards.
    const double contribution_low = coefficient > 0.0 ? lo : up;
    const double contribution_high = coefficient > 0.0 ? up : lo;
    if (finite(contribution_low)) {
      bounds.lower += coefficient * contribution_low;
    } else {
      bounds.lower_finite = false;
    }
    if (finite(contribution_high)) {
      bounds.upper += coefficient * contribution_high;
    } else {
      bounds.upper_finite = false;
    }
  }
  return bounds;
}

/// Drop a column from every row it appears in, folding its fixed value into the row bounds.
void fold_fixed_column(Workspace* work, Index column, double value) {
  const ColumnView view = work->original->matrix.column(column);
  for (Index k = 0; k < view.size; ++k) {
    const Index row = view.rows[k];
    const auto r = static_cast<std::size_t>(row);
    if (work->row_dead[r]) continue;
    const double shift = view.values[k] * value;
    if (finite(work->row_lower[r])) work->row_lower[r] -= shift;
    if (finite(work->row_upper[r])) work->row_upper[r] -= shift;
    --work->row_count[r];
  }
  work->col_dead[static_cast<std::size_t>(column)] = true;
}

void kill_row(Workspace* work, Index row) {
  const auto r = static_cast<std::size_t>(row);
  if (work->row_dead[r]) return;
  work->row_dead[r] = true;
  for (const auto& [column, coefficient] : work->rows[r]) {
    (void)coefficient;
    const auto u = static_cast<std::size_t>(column);
    if (!work->col_dead[u]) --work->col_count[u];
  }
}

}  // namespace

// ===========================================================================================

Result presolve(const Model& model, const Options& options, Logger& logger) {
  Result result;
  result.original_rows = model.num_rows();
  result.original_cols = model.num_cols();
  result.original_nonzeros = model.num_nonzeros();

  const Index m = model.num_rows();
  const Index n = model.num_cols();
  const double feasibility = options.get_double("primal_feasibility_tolerance");

  Workspace work;
  work.original = &model;
  work.col_lower = model.col_lower;
  work.col_upper = model.col_upper;
  work.row_lower = model.row_lower;
  work.row_upper = model.row_upper;
  work.col_dead.assign(static_cast<std::size_t>(n), false);
  work.row_dead.assign(static_cast<std::size_t>(m), false);
  work.rows.assign(static_cast<std::size_t>(m), {});
  work.col_count.assign(static_cast<std::size_t>(n), 0);
  work.row_count.assign(static_cast<std::size_t>(m), 0);

  for (Index j = 0; j < n; ++j) {
    const ColumnView view = model.matrix.column(j);
    work.col_count[static_cast<std::size_t>(j)] = view.size;
    for (Index k = 0; k < view.size; ++k) {
      work.rows[static_cast<std::size_t>(view.rows[k])].emplace_back(j, view.values[k]);
      ++work.row_count[static_cast<std::size_t>(view.rows[k])];
    }
  }

  const auto infeasible = [&](std::string why) {
    result.proved_infeasible = true;
    result.message = std::move(why);
  };

  // Passes run to a fixed point: fixing a column can empty a row, removing a row can turn a
  // column into a singleton. The cap is a safety net, not an expected limit - each pass must
  // strictly remove something or the loop breaks on its own.
  constexpr int kMaxPasses = 20;
  for (int pass = 0; pass < kMaxPasses && !result.proved_infeasible; ++pass) {
    bool changed = false;

    // --- columns -----------------------------------------------------------------------
    for (Index j = 0; j < n && !result.proved_infeasible; ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (work.col_dead[u]) continue;

      if (work.col_lower[u] > work.col_upper[u] + feasibility) {
        infeasible(fmt::format("column {} has crossed bounds after presolve: [{:.6g}, {:.6g}]",
                               j, work.col_lower[u], work.col_upper[u]));
        break;
      }

      // Fixed column: the value is known, so fold it into the rows and drop it.
      if (work.col_upper[u] - work.col_lower[u] <= feasibility && finite(work.col_lower[u])) {
        const double value = work.col_lower[u];
        Record record;
        record.kind = Record::Kind::kFixedColumn;
        record.index = j;
        record.value = value;
        result.records.push_back(record);
        fold_fixed_column(&work, j, value);
        changed = true;
        continue;
      }

      // Empty column: nothing constrains it, so the cost alone decides. If the cost pushes
      // it toward an infinite bound the problem is unbounded, and we say so rather than
      // parking it somewhere arbitrary and letting the engine discover it later.
      if (work.col_count[u] == 0) {
        const double cost = model.sense_multiplier() * model.col_cost[u];
        double value = 0.0;
        if (cost > 0.0) {
          value = work.col_lower[u];
        } else if (cost < 0.0) {
          value = work.col_upper[u];
        } else {
          value = finite(work.col_lower[u])
                      ? work.col_lower[u]
                      : (finite(work.col_upper[u]) ? work.col_upper[u] : 0.0);
        }
        // PRESOLVE CANNOT CONCLUDE UNBOUNDEDNESS. A column with no entries whose cost
        // pushes it to an infinite bound makes the objective unbounded ONLY IF the feasible
        // region is non-empty, and nothing here has established that. The rational oracle
        // produced the counterexample directly: an instance with an empty column of negative
        // cost AND a row that no assignment can satisfy is infeasible, not unbounded, and
        // reporting the latter is as wrong as reporting the former.
        //
        // So the column is simply left in the model. The simplex settles feasibility in
        // phase 1 before it can report anything about the objective, which is exactly the
        // ordering this reduction cannot reproduce on its own.
        if (!finite(value)) continue;
        Record record;
        record.kind = Record::Kind::kEmptyColumn;
        record.index = j;
        record.value = value;
        result.records.push_back(record);
        work.col_dead[u] = true;
        changed = true;
      }
    }

    // --- rows --------------------------------------------------------------------------
    for (Index i = 0; i < m && !result.proved_infeasible; ++i) {
      const auto r = static_cast<std::size_t>(i);
      if (work.row_dead[r]) continue;

      if (work.row_lower[r] > work.row_upper[r] + feasibility) {
        infeasible(fmt::format("row {} has crossed bounds after presolve: [{:.6g}, {:.6g}]", i,
                               work.row_lower[r], work.row_upper[r]));
        break;
      }

      // Empty row: activity is exactly zero, so the row is either vacuous or a proof of
      // infeasibility all by itself.
      if (work.row_count[r] == 0) {
        const bool violates_lower =
            finite(work.row_lower[r]) && work.row_lower[r] > feasibility;
        const bool violates_upper =
            finite(work.row_upper[r]) && work.row_upper[r] < -feasibility;
        if (violates_lower || violates_upper) {
          infeasible(fmt::format(
              "row {} has no entries but requires activity in [{:.6g}, {:.6g}], which excludes "
              "zero",
              i, work.row_lower[r], work.row_upper[r]));
          break;
        }
        Record record;
        record.kind = Record::Kind::kEmptyRow;
        record.index = i;
        result.records.push_back(record);
        kill_row(&work, i);
        changed = true;
        continue;
      }

      const ActivityBounds bounds = activity_bounds(work, i);

      // Redundant row: whatever the variables do within their bounds, this row is satisfied.
      const bool lower_slack =
          !finite(work.row_lower[r]) ||
          (bounds.lower_finite && bounds.lower >= work.row_lower[r] - feasibility);
      const bool upper_slack =
          !finite(work.row_upper[r]) ||
          (bounds.upper_finite && bounds.upper <= work.row_upper[r] + feasibility);
      if (lower_slack && upper_slack) {
        Record record;
        record.kind = Record::Kind::kRedundantRow;
        record.index = i;
        result.records.push_back(record);
        kill_row(&work, i);
        changed = true;
        continue;
      }

      // Infeasible by activity: the row demands more than the bounds can ever supply.
      if (bounds.upper_finite && finite(work.row_lower[r]) &&
          bounds.upper < work.row_lower[r] - feasibility) {
        infeasible(fmt::format(
            "row {} needs activity of at least {:.6g} but the column bounds cap it at {:.6g}",
            i, work.row_lower[r], bounds.upper));
        break;
      }
      if (bounds.lower_finite && finite(work.row_upper[r]) &&
          bounds.lower > work.row_upper[r] + feasibility) {
        infeasible(fmt::format(
            "row {} allows activity of at most {:.6g} but the column bounds force at least "
            "{:.6g}",
            i, work.row_upper[r], bounds.lower));
        break;
      }

      // Singleton row: a * x within [lo, up] is a bound on x, not a constraint. Tighten and
      // drop the row. The division flips the sense when a is negative, which is the other
      // easy thing to get backwards here.
      if (work.row_count[r] == 1) {
        Index column = -1;
        double coefficient = 0.0;
        for (const auto& entry : work.rows[r]) {
          if (!work.col_dead[static_cast<std::size_t>(entry.first)]) {
            column = entry.first;
            coefficient = entry.second;
            break;
          }
        }
        if (column < 0 || std::fabs(coefficient) < tol::kZeroDrop) continue;

        const auto c = static_cast<std::size_t>(column);
        double implied_lower = -kInfinity;
        double implied_upper = kInfinity;
        if (coefficient > 0.0) {
          if (finite(work.row_lower[r])) implied_lower = work.row_lower[r] / coefficient;
          if (finite(work.row_upper[r])) implied_upper = work.row_upper[r] / coefficient;
        } else {
          if (finite(work.row_upper[r])) implied_lower = work.row_upper[r] / coefficient;
          if (finite(work.row_lower[r])) implied_upper = work.row_lower[r] / coefficient;
        }
        if (model.col_type[c] == VarType::kInteger) {
          if (finite(implied_lower)) implied_lower = round_integer_lower(implied_lower);
          if (finite(implied_upper)) implied_upper = round_integer_upper(implied_upper);
        }

        Record record;
        record.kind = Record::Kind::kSingletonRow;
        record.index = i;
        record.column = column;
        record.coefficient = coefficient;
        record.row_lower = work.row_lower[r];
        record.row_upper = work.row_upper[r];
        result.records.push_back(record);

        if (finite(implied_lower))
          work.col_lower[c] = std::max(work.col_lower[c], implied_lower);
        if (finite(implied_upper))
          work.col_upper[c] = std::min(work.col_upper[c], implied_upper);
        kill_row(&work, i);
        changed = true;
      }
    }

    if (!changed) break;
  }

  if (result.proved_infeasible) {
    logger.info("Presolve proved infeasibility: {}", result.message);
    return result;
  }

  // --- compaction ----------------------------------------------------------------------
  std::vector<Index> new_col_index(static_cast<std::size_t>(n), -1);
  std::vector<Index> new_row_index(static_cast<std::size_t>(m), -1);
  for (Index j = 0; j < n; ++j) {
    if (work.col_dead[static_cast<std::size_t>(j)]) continue;
    new_col_index[static_cast<std::size_t>(j)] =
        static_cast<Index>(result.col_to_original.size());
    result.col_to_original.push_back(j);
  }
  for (Index i = 0; i < m; ++i) {
    if (work.row_dead[static_cast<std::size_t>(i)]) continue;
    new_row_index[static_cast<std::size_t>(i)] =
        static_cast<Index>(result.row_to_original.size());
    result.row_to_original.push_back(i);
  }

  Model& reduced = result.model;
  reduced.name = model.name;
  reduced.source_path = model.source_path;
  reduced.sense = model.sense;
  reduced.objective_offset = model.objective_offset;

  const auto reduced_cols = static_cast<Index>(result.col_to_original.size());
  const auto reduced_rows = static_cast<Index>(result.row_to_original.size());

  for (const Index j : result.col_to_original) {
    const auto u = static_cast<std::size_t>(j);
    reduced.col_cost.push_back(model.col_cost[u]);
    reduced.col_lower.push_back(work.col_lower[u]);
    reduced.col_upper.push_back(work.col_upper[u]);
    reduced.col_type.push_back(model.col_type[u]);
    if (u < model.col_names.size()) reduced.col_names.push_back(model.col_names[u]);
  }
  for (const Index i : result.row_to_original) {
    const auto r = static_cast<std::size_t>(i);
    reduced.row_lower.push_back(work.row_lower[r]);
    reduced.row_upper.push_back(work.row_upper[r]);
    if (r < model.row_names.size()) reduced.row_names.push_back(model.row_names[r]);
  }

  // The objective CONSTANT contributed by fixed columns is folded in here. Forgetting this
  // is the classic presolve bug: every reported objective comes back short by exactly the
  // cost of the variables that were removed, on every instance, and it looks like a solver
  // accuracy problem rather than a bookkeeping one.
  for (const Record& record : result.records) {
    if (record.kind == Record::Kind::kFixedColumn ||
        record.kind == Record::Kind::kEmptyColumn) {
      reduced.objective_offset +=
          model.col_cost[static_cast<std::size_t>(record.index)] * record.value;
    }
  }

  reduced.matrix.reset(reduced_rows, reduced_cols);
  for (Index j = 0; j < reduced_cols; ++j) {
    const Index original_col = result.col_to_original[static_cast<std::size_t>(j)];
    const ColumnView view = model.matrix.column(original_col);
    for (Index k = 0; k < view.size; ++k) {
      const Index mapped = new_row_index[static_cast<std::size_t>(view.rows[k])];
      if (mapped >= 0) reduced.matrix.add_entry(mapped, j, view.values[k]);
    }
  }
  reduced.matrix.finalize();
  reduced.hessian.reset(reduced_cols, reduced_cols);
  reduced.hessian.finalize();

  logger.info("Presolve: {} rows -> {}, {} columns -> {}, {} nonzeros -> {}",
              result.original_rows, reduced_rows, result.original_cols, reduced_cols,
              result.original_nonzeros, reduced.num_nonzeros());
  return result;
}

// ===========================================================================================

Solution postsolve(const Result& result, const Model& original, const Solution& reduced) {
  Solution solution;
  solution.allocate_for(original);
  solution.status = reduced.status;
  solution.algorithm = reduced.algorithm;
  solution.message = reduced.message;
  solution.iterations = reduced.iterations;
  solution.nodes = reduced.nodes;
  solution.solve_seconds = reduced.solve_seconds;

  // Start from the reduced point, scattered back into original positions.
  for (std::size_t j = 0; j < result.col_to_original.size(); ++j) {
    const auto target = static_cast<std::size_t>(result.col_to_original[j]);
    if (j < reduced.col_value.size()) solution.col_value[target] = reduced.col_value[j];
    if (j < reduced.col_dual.size()) solution.col_dual[target] = reduced.col_dual[j];
    if (j < reduced.col_status.size()) solution.col_status[target] = reduced.col_status[j];
  }
  for (std::size_t i = 0; i < result.row_to_original.size(); ++i) {
    const auto target = static_cast<std::size_t>(result.row_to_original[i]);
    if (i < reduced.row_dual.size()) solution.row_dual[target] = reduced.row_dual[i];
    if (i < reduced.row_status.size()) solution.row_status[target] = reduced.row_status[i];
  }

  // Replay in REVERSE for the PRIMAL values. A column fixed in pass 3 may sit in a row
  // removed in pass 1, so undoing them in application order would price a row against a
  // point that does not exist yet.
  for (auto it = result.records.rbegin(); it != result.records.rend(); ++it) {
    const Record& record = *it;
    switch (record.kind) {
      case Record::Kind::kFixedColumn:
      case Record::Kind::kEmptyColumn:
        solution.col_value[static_cast<std::size_t>(record.index)] = record.value;
        solution.col_status[static_cast<std::size_t>(record.index)] = BasisStatus::kFixed;
        break;
      case Record::Kind::kEmptyRow:
      case Record::Kind::kRedundantRow:
      case Record::Kind::kForcingRow:
        // A row removed because it cannot bind is, by construction, inactive at the optimum,
        // so its dual is zero. That is what "redundant" means; any other value would violate
        // complementary slackness.
        solution.row_dual[static_cast<std::size_t>(record.index)] = 0.0;
        solution.row_status[static_cast<std::size_t>(record.index)] = BasisStatus::kBasic;
        break;
      case Record::Kind::kSingletonRow:
        solution.row_dual[static_cast<std::size_t>(record.index)] = 0.0;
        solution.row_status[static_cast<std::size_t>(record.index)] = BasisStatus::kBasic;
        break;
    }
  }

  // THE DUALS ARE A SEPARATE PASS, and getting this wrong is the whole difficulty of
  // postsolve. A singleton row became a BOUND on one column, so at the optimum the price
  // that would have sat on that row is hiding in the column's reduced cost - or, when the
  // bound it implied was tight enough to fix the column outright, nowhere at all.
  //
  // The first version of this computed the row's dual from whatever col_dual happened to
  // hold, having just zeroed it two branches earlier. On adlittle that left column 95 sitting
  // at its lower bound with a reduced cost of -857: an engine claiming optimality while
  // pointing at a direction that improves the objective. The status guard caught it and
  // downgraded the answer to `feasible`, which is exactly what that guard is for, but the
  // answer was still wrong.
  //
  // So each singleton row is priced from the ORIGINAL matrix, using the duals known at that
  // point, and asked a direct question: given every other row's price, does this column's
  // reduced cost violate the sign its own ORIGINAL bounds demand? If it does, this row is
  // what pays for it.
  const double sense = original.sense_multiplier();
  const auto reduced_cost_of = [&](Index column) {
    double d = original.col_cost[static_cast<std::size_t>(column)];
    const ColumnView view = original.matrix.column(column);
    for (Index k = 0; k < view.size; ++k) {
      d -= view.values[k] * solution.row_dual[static_cast<std::size_t>(view.rows[k])];
    }
    return d;
  };

  for (auto it = result.records.rbegin(); it != result.records.rend(); ++it) {
    if (it->kind != Record::Kind::kSingletonRow) continue;
    const auto c = static_cast<std::size_t>(it->column);
    if (std::fabs(it->coefficient) <= tol::kZeroDrop) continue;

    const double x = solution.col_value[c];
    const double lo = original.col_lower[c];
    const double hi = original.col_upper[c];
    const bool at_lower = finite(lo) && std::fabs(x - lo) <= tol::kPrimalFeasibility;
    const bool at_upper = finite(hi) && std::fabs(x - hi) <= tol::kPrimalFeasibility;

    const double d = reduced_cost_of(it->column);
    const double signed_d = sense * d;
    // Free to sit where it is, so the reduced cost must be zero; at a bound, only the wrong
    // sign needs paying for. A column already at an ORIGINAL bound with an admissible sign
    // needs no price on this row at all, and inventing one would break complementary
    // slackness on a row that is not active.
    const bool needs_price = (!at_lower && !at_upper) ||
                             (at_lower && !at_upper && signed_d < -tol::kDualFeasibility) ||
                             (at_upper && !at_lower && signed_d > tol::kDualFeasibility);
    if (!needs_price) continue;

    // THE ROW MUST ACTUALLY BE ACTIVE. Complementary slackness forbids a price on a
    // constraint that is not binding, and a column can be constrained by SEVERAL singleton
    // rows - a lower bound from one and an upper from another. Without this test the first
    // record reached wins, and on `-3*x1 >= -9` together with `2*x1 >= 2` at x1 = 3 the
    // price landed on the second, which is slack by 4. The oracle reported 59 degenerate
    // instances where the optimum existed and the solver returned merely `feasible`; this
    // was all of them.
    const double activity = it->coefficient * x;
    const bool row_at_lower =
        finite(it->row_lower) && std::fabs(activity - it->row_lower) <= tol::kPrimalFeasibility;
    const bool row_at_upper =
        finite(it->row_upper) && std::fabs(activity - it->row_upper) <= tol::kPrimalFeasibility;
    if (!row_at_lower && !row_at_upper) continue;

    // AND THE PRICE MUST HAVE AN ADMISSIBLE SIGN. A column can be pinned between two
    // opposing singleton rows - `6*x2 >= 24` and `-3*x2 >= -12` both hold with equality at
    // x2 = 4 - and only one of them can legitimately carry the price. In minimise space a
    // row active at its lower bound needs a non-negative dual and one active at its upper
    // bound a non-positive one; the other row is active but its multiplier would have the
    // wrong sign, which is dual infeasibility however tidy the arithmetic looks.
    //
    // Taking whichever record came first put -2 on the `-3*x2 >= -12` row and left the
    // instance reported `feasible` with a correct objective. Skipping the inadmissible one
    // lets the next record price it correctly, because d is recomputed from the duals each
    // time round.
    const double candidate = d / it->coefficient;
    const double signed_candidate = sense * candidate;
    if (row_at_lower && !row_at_upper && signed_candidate < -tol::kDualFeasibility) continue;
    if (row_at_upper && !row_at_lower && signed_candidate > tol::kDualFeasibility) continue;

    solution.row_dual[static_cast<std::size_t>(it->index)] = candidate;
    solution.row_status[static_cast<std::size_t>(it->index)] =
        row_at_lower ? BasisStatus::kAtLower : BasisStatus::kAtUpper;
    solution.col_status[c] = BasisStatus::kBasic;
    // The price was chosen precisely to cancel this column's reduced cost, so set it to
    // exactly zero rather than leaving a rounded residue for the verifier to trip over.
    solution.col_dual[c] = 0.0;
  }

  // Reduced costs for the REMOVED columns only. The surviving columns keep what the engine
  // reported, and that is deliberate.
  //
  // Recomputing them all looks tidier and is subtly worse. A removed row contributes nothing
  // (its dual is zero), and a priced singleton row has exactly one entry, so it can only
  // touch its own column - which means a survivor's reduced cost is already correct and
  // recomputing merely re-derives it through different floating-point operations. On capri
  // that turned an exact 0.0 on a FREE column into -1.1e-16, and the verifier's
  // complementary-slackness product |d| * slack, with slack infinite on a free column,
  // evaluated to inf. A correct answer, rejected, because we had rounded a zero.
  for (const Record& record : result.records) {
    if (record.kind != Record::Kind::kFixedColumn &&
        record.kind != Record::Kind::kEmptyColumn) {
      continue;
    }
    solution.col_dual[static_cast<std::size_t>(record.index)] = reduced_cost_of(record.index);
  }

  // Activities, the objective and every quality measure are recomputed against the ORIGINAL
  // model rather than carried across. That is the whole safety net: if a reduction or its
  // postsolve is wrong, it shows up here as a feasibility violation on a model the engine
  // never saw, and the dispatcher's status guard then refuses to call it optimal.
  // dual_bound BEFORE recompute_quality, which derives absolute_gap and relative_gap from
  // it. Setting it afterwards left every presolved solve reporting a gap of |objective - 0|
  // - on the simplex's own gap test that came out as an absolute gap of 36 on a solved LP.
  // The reduced model carries the folded objective offset, so its bound is already in the
  // original problem's units.
  solution.dual_bound = reduced.dual_bound;
  solution.recompute_quality(original);
  return solution;
}

}  // namespace sankhya::presolve

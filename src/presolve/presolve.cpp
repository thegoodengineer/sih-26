// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve reductions. See presolve.hpp for the references and the rationale.

#include "presolve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
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
  /// Column costs, mutated by kFreeColumnSingleton and kDoubletonEquation as they fold an
  /// eliminated column's cost into whichever columns still share its row(s). Always in the
  /// ORIGINAL (not sense-adjusted) sense, exactly like Model::col_cost.
  std::vector<double> col_cost;
  /// Constant shifted into the objective by the same two reductions, added to
  /// Model::objective_offset at the end alongside the existing fixed/empty-column fold.
  double objective_offset_delta = 0.0;
  /// Set once a column has played EITHER role - eliminated or kept - in a doubleton-equation
  /// fold. Postsolve's per-doubleton special case prices that fold from one row's dual alone;
  /// if the same column entered a SECOND, independently-folded doubleton in either role, its
  /// true reduced cost would depend on both row duals simultaneously - a coupled system the
  /// per-row postsolve formula cannot recover. Declining the second fold (see presolve())
  /// trades a rarer reduction opportunity for a postsolve that is provably correct in every
  /// case it fires.
  std::vector<bool> doubleton_touched;
  /// Every row whose coefficient on a column doubleton fill-in touched, indexed by column:
  /// the accumulated DELTA for (column, row), whether that fill-in adjusted an entry already
  /// in `original` or created a brand-new one. `original->matrix.column()` alone - what
  /// fold_fixed_column used to rely on exclusively - only ever sees the ORIGINAL coefficient,
  /// stale the moment fill-in adjusts it, and blind entirely to a brand-new one.
  std::vector<std::unordered_map<Index, double>> extra_row_delta;
  /// The subset of extra_row_delta's rows that are BRAND NEW - absent from `original`
  /// entirely, so a walk of `original->matrix.column()` would not visit them at all even
  /// with the delta available. fold_fixed_column needs this list to know which EXTRA rows
  /// to fold into, on top of patching the ones `original` already finds.
  std::vector<std::vector<Index>> extra_new_rows;
  /// Set once a column has been the SUBJECT of a kSingletonRow reduction (its bound was
  /// tightened, or it was fully explained, by that row). A doubleton declines to eliminate OR
  /// keep such a column afterwards (see doubleton_touched's guard, extended below) - if the
  /// column is genuinely interior, postsolve's kSingletonRow pass may have already fixed a
  /// row price that assumed this doubleton's own row contributes nothing to the column's
  /// stationarity, and the doubleton has no way to revise that fixed price if its own
  /// admissibility check later disagrees. Two rows contending over one column's price is
  /// exactly the kind of coupled system the per-row postsolve formulas cannot solve; declining
  /// here keeps every doubleton that DOES fire provably correct, at the cost of a reduction.
  std::vector<bool> singleton_row_touched;
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
///
/// Walks `original`'s view, PATCHED by `extra_row_delta[column]`, plus `extra_new_rows
/// [column]` for rows fill-in put the column into that `original` has no trace of at all.
/// Using `original`'s coefficient unpatched is just as wrong as skipping a fill-in row
/// outright: a doubleton's fill-in can have adjusted an entry `original` already had, and
/// folding the STALE, pre-fill-in coefficient shifts that row's bound by the wrong amount -
/// leaving a contribution the column no longer makes (or makes at the wrong rate) baked into
/// the row's bound, which the row's later activity-bounds check then measures against the
/// TRUE, patched coefficients of everything else still in it. A fuzzed instance where the
/// fixed column was ALSO a doubleton's `keep`, patched in a row it already originally
/// touched, is what caught this - `extra_new_rows` alone (brand-new rows only) had already
/// fixed the same failure mode for a row the column was not originally in at all.
void fold_fixed_column(Workspace* work, Index column, double value) {
  const auto uc = static_cast<std::size_t>(column);
  const auto& deltas = work->extra_row_delta[uc];
  const ColumnView view = work->original->matrix.column(column);
  for (Index k = 0; k < view.size; ++k) {
    const Index row = view.rows[k];
    const auto r = static_cast<std::size_t>(row);
    if (work->row_dead[r]) continue;
    double coefficient = view.values[k];
    const auto found = deltas.find(row);
    if (found != deltas.end()) coefficient += found->second;
    const double shift = coefficient * value;
    if (finite(work->row_lower[r])) work->row_lower[r] -= shift;
    if (finite(work->row_upper[r])) work->row_upper[r] -= shift;
    --work->row_count[r];
  }
  for (const Index row : work->extra_new_rows[uc]) {
    const auto r = static_cast<std::size_t>(row);
    if (work->row_dead[r]) continue;
    const auto found = deltas.find(row);
    if (found == deltas.end()) continue;
    const double shift = found->second * value;
    if (finite(work->row_lower[r])) work->row_lower[r] -= shift;
    if (finite(work->row_upper[r])) work->row_upper[r] -= shift;
    --work->row_count[r];
  }
  work->col_dead[uc] = true;
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
  work.col_cost = model.col_cost;
  work.doubleton_touched.assign(static_cast<std::size_t>(n), false);
  work.singleton_row_touched.assign(static_cast<std::size_t>(n), false);
  work.extra_row_delta.assign(static_cast<std::size_t>(n), {});
  work.extra_new_rows.assign(static_cast<std::size_t>(n), {});

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
        const double cost = model.sense_multiplier() * work.col_cost[u];
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
        continue;
      }

      // Free column singleton (issue #92; Andersen & Andersen 1995). A column with no bound
      // on either side, appearing in exactly one row, can always be moved to make that row's
      // activity land wherever it needs to - it constrains nothing ELSE, so both column and
      // row can go. What is left behind is a choice: for fixed values of the other variables
      // in the row, x_j ranges freely over an interval as the row activity sweeps [lo, hi],
      // and since x_j's cost is linear, the objective is minimised at whichever END of that
      // interval the sign of cost/coefficient prefers.
      //
      // If that favourable end is an INFINITE row bound, the column can be pushed that way
      // without limit - which does not yet prove the whole LP unbounded, for the identical
      // reason kEmptyColumn above does not: some other, as yet unexamined, row might make the
      // feasible region empty. So this reduction simply declines rather than guess, exactly
      // like kEmptyColumn's own unboundedness case just above it.
      if (work.col_count[u] == 1 && !finite(work.col_lower[u]) && !finite(work.col_upper[u])) {
        // The live row's coefficient is read from `original` and then patched by
        // extra_row_delta[j] - a doubleton's fill-in can have adjusted it already, and using
        // the ORIGINAL, stale value here computes a substitution formula for the WRONG row
        // equation, one that has not existed since the fill-in changed it. extra_new_rows[j]
        // covers the other half: a row fill-in put this column into that it did not
        // originally touch at all, which a plain walk of `original`'s view would miss even
        // with the delta available - the only way this column ends up a singleton by way of
        // a row it was never originally in.
        const ColumnView view = model.matrix.column(j);
        Index row = -1;
        double a = 0.0;
        for (Index k = 0; k < view.size; ++k) {
          const auto rr = static_cast<std::size_t>(view.rows[k]);
          if (!work.row_dead[rr]) {
            row = view.rows[k];
            a = view.values[k];
            const auto found = work.extra_row_delta[u].find(row);
            if (found != work.extra_row_delta[u].end()) a += found->second;
            break;
          }
        }
        if (row < 0) {
          for (const Index candidate : work.extra_new_rows[u]) {
            const auto rr = static_cast<std::size_t>(candidate);
            if (work.row_dead[rr]) continue;
            row = candidate;
            a = work.extra_row_delta[u].at(candidate);
            break;
          }
        }
        if (row < 0 || std::fabs(a) < tol::kZeroDrop) continue;
        const auto r = static_cast<std::size_t>(row);

        const double cj = work.col_cost[u];
        // Rate of change of the (sense-adjusted) objective per unit of row activity: moving
        // the row's activity by ds moves x_j by ds/a, and the objective by cj*ds/a.
        const double rate = model.sense_multiplier() * cj / a;
        double target = 0.0;
        if (rate > tol::kZeroDrop) {
          if (!finite(work.row_lower[r])) continue;
          target = work.row_lower[r];
        } else if (rate < -tol::kZeroDrop) {
          if (!finite(work.row_upper[r])) continue;
          target = work.row_upper[r];
        } else {
          // The row's own activity does not affect the objective at all: any point in its
          // range does equally well, so any finite one will do, and 0 if the row is entirely
          // unconstrained (nothing else uses this row - it would not have survived the
          // empty/redundant-row checks above otherwise, if it had any OTHER live column).
          target = finite(work.row_lower[r])
                       ? work.row_lower[r]
                       : (finite(work.row_upper[r]) ? work.row_upper[r] : 0.0);
        }

        Record record;
        record.kind = Record::Kind::kFreeColumnSingleton;
        record.index = row;
        record.column = j;
        record.coefficient = a;
        record.substituted_rhs = target;
        record.eliminated_cost = cj;
        result.records.push_back(record);

        // Fold x_j's cost into every other live column sharing the row, and shift the
        // objective by the constant term - the same elimination arithmetic doubleton
        // equation uses below, just without any fill-in because a singleton column has no
        // OTHER row to fill in.
        for (auto& entry : work.rows[r]) {
          const Index other = entry.first;
          if (other == j) continue;
          const auto ou = static_cast<std::size_t>(other);
          if (work.col_dead[ou]) continue;
          work.col_cost[ou] -= cj * (entry.second / a);
        }
        work.objective_offset_delta += cj * target / a;

        work.col_dead[u] = true;
        kill_row(&work, row);
        changed = true;
        continue;
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

        // A column already touched by a doubleton fold (as `elim` or `keep`) cannot also
        // absorb a singleton row's price: the doubleton's own dual reconstruction folds the
        // ELIMINATED column's cost into this one using a STATIC baseline (its cost at the
        // moment presolve() eliminated it), which is only exact when that eliminated column
        // has no other live row of its own to also pull its true price away from that
        // baseline - true for a plain free-column-singleton, false in general for a
        // doubleton's `elim`. When it has other rows, this column's TRUE reduced cost needs
        // the doubleton's row dual and THIS row's dual solved jointly, which the per-row
        // postsolve formulas cannot do (the same class of coupling doubleton_touched already
        // declines between two doubletons - see its field comment). `bandm`'s ORROLC/RDAS2Q
        // pair is exactly this: RDAS2Q (elim) touches 8 other rows, so its true row-47 price
        // is not eliminated_cost/a, and pricing ORROLC from row 111 alone - as this reduction
        // would - ignored that, failing the independent verifier's reduced-cost and strong-
        // duality checks even though the primal objective was exact. Leaving this row alone
        // keeps it in the reduced model, where the simplex itself solves the coupling
        // correctly instead of postsolve trying to reconstruct it by hand.
        if (work.doubleton_touched[c]) continue;

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
        work.singleton_row_touched[c] = true;

        if (finite(implied_lower))
          work.col_lower[c] = std::max(work.col_lower[c], implied_lower);
        if (finite(implied_upper))
          work.col_upper[c] = std::min(work.col_upper[c], implied_upper);
        kill_row(&work, i);
        changed = true;
        continue;
      }

      // Doubleton equation (issue #92; Andersen & Andersen 1995). An equality with exactly
      // two live entries lets one column be written EXACTLY as an affine function of the
      // other, for every value the other can take - not an approximation, a genuine
      // elimination. Restricted to eliminating a CONTINUOUS column: an integer column's
      // eliminated role would need its own integrality re-derived on the survivor from the
      // coefficient ratio, which is a different reduction (Achterberg et al. cover it
      // separately) and out of scope here.
      if (work.row_count[r] == 2 && work.row_upper[r] - work.row_lower[r] <= feasibility &&
          finite(work.row_lower[r])) {
        Index first = -1, second = -1;
        double first_coeff = 0.0, second_coeff = 0.0;
        for (const auto& entry : work.rows[r]) {
          if (work.col_dead[static_cast<std::size_t>(entry.first)]) continue;
          if (first < 0) {
            first = entry.first;
            first_coeff = entry.second;
          } else {
            second = entry.first;
            second_coeff = entry.second;
          }
        }
        if (first < 0 || second < 0) continue;
        // Eliminate whichever of the two is continuous; if both are, either choice is exact
        // and `first` is taken. If NEITHER is continuous, this reduction does not apply.
        Index elim = first, keep = second;
        double a = first_coeff, b = second_coeff;
        if (model.col_type[static_cast<std::size_t>(first)] == VarType::kInteger) {
          if (model.col_type[static_cast<std::size_t>(second)] == VarType::kInteger) continue;
          elim = second;
          keep = first;
          a = second_coeff;
          b = first_coeff;
        }
        if (std::fabs(a) < tol::kZeroDrop || std::fabs(b) < tol::kZeroDrop) continue;

        const auto ue = static_cast<std::size_t>(elim);
        const auto uk = static_cast<std::size_t>(keep);

        // A column already touched by an earlier doubleton fold - whether as that fold's
        // survivor (`keep`) or as the column it eliminated (`elim`, whose own
        // "eliminated_cost" already carries that fold's effect) - cannot enter a SECOND
        // doubleton at all, in either role: postsolve's per-doubleton special case treats one
        // row's dual as the only unknown in that column's stationarity, and a column touched
        // twice needs two duals solved simultaneously, which it cannot do (see the field
        // comment on doubleton_touched above). Leave this row as an ordinary two-entry
        // equality instead.
        if (work.doubleton_touched[ue] || work.doubleton_touched[uk]) continue;

        // A column ALREADY the subject of a kSingletonRow reduction (its bound tightened or
        // fully explained by that row) cannot enter a doubleton either, in either role - see
        // singleton_row_touched's field comment for why: two rows would then be contending
        // over the same column's stationarity, and postsolve's dual passes decide each row
        // from its own record with one unknown at a time; they run to a fixed point (#157),
        // which propagates a price between records but cannot solve two rows' prices for one
        // column jointly.
        if (work.singleton_row_touched[ue] || work.singleton_row_touched[uk]) continue;

        const double rhs = work.row_lower[r];  // == work.row_upper[r], within `feasibility`

        // The range of a*x_elim given x_elim in [lo_elim, hi_elim] - same contribution
        // pattern activity_bounds() uses above, specialised to one term.
        const double lo_e = work.col_lower[ue];
        const double hi_e = work.col_upper[ue];
        double term_lo = 0.0, term_hi = 0.0;
        bool term_lo_finite = false, term_hi_finite = false;
        if (a > 0.0) {
          if (finite(lo_e)) {
            term_lo = a * lo_e;
            term_lo_finite = true;
          }
          if (finite(hi_e)) {
            term_hi = a * hi_e;
            term_hi_finite = true;
          }
        } else {
          if (finite(hi_e)) {
            term_lo = a * hi_e;
            term_lo_finite = true;
          }
          if (finite(lo_e)) {
            term_hi = a * lo_e;
            term_hi_finite = true;
          }
        }
        // b*x_keep = rhs - a*x_elim, so its range is [rhs - term_hi, rhs - term_lo].
        const double bxk_lo = term_hi_finite ? rhs - term_hi : -kInfinity;
        const double bxk_hi = term_lo_finite ? rhs - term_lo : kInfinity;
        const bool bxk_lo_finite = term_hi_finite;
        const bool bxk_hi_finite = term_lo_finite;
        double implied_k_lower = -kInfinity;
        double implied_k_upper = kInfinity;
        if (b > 0.0) {
          if (bxk_lo_finite) implied_k_lower = bxk_lo / b;
          if (bxk_hi_finite) implied_k_upper = bxk_hi / b;
        } else {
          if (bxk_hi_finite) implied_k_lower = bxk_hi / b;
          if (bxk_lo_finite) implied_k_upper = bxk_lo / b;
        }
        if (model.col_type[uk] == VarType::kInteger) {
          if (finite(implied_k_lower)) implied_k_lower = round_integer_lower(implied_k_lower);
          if (finite(implied_k_upper)) implied_k_upper = round_integer_upper(implied_k_upper);
        }

        Record record;
        record.kind = Record::Kind::kDoubletonEquation;
        record.index = i;
        record.column = elim;
        record.coefficient = a;
        record.partner_column = keep;
        record.partner_coefficient = b;
        record.substituted_rhs = rhs;
        record.eliminated_cost = work.col_cost[ue];
        result.records.push_back(record);
        work.doubleton_touched[ue] = true;
        work.doubleton_touched[uk] = true;

        if (finite(implied_k_lower))
          work.col_lower[uk] = std::max(work.col_lower[uk], implied_k_lower);
        if (finite(implied_k_upper))
          work.col_upper[uk] = std::min(work.col_upper[uk], implied_k_upper);

        // Fold the eliminated column's cost into the row it is leaving, exactly like the
        // free-column-singleton fold above.
        const double ce = work.col_cost[ue];
        work.col_cost[uk] -= ce * (b / a);
        work.objective_offset_delta += ce * rhs / a;

        // FILL-IN. The eliminated column may appear in OTHER rows too - unlike a free
        // singleton, a doubleton column is not required to be a singleton itself. Every such
        // row loses the eliminated column and gains (or adjusts) a term on the kept column,
        // exactly as substituting x_elim = (rhs - b*x_keep)/a into a_i'e*x_elim would produce
        // by hand.
        const ColumnView elim_view = model.matrix.column(elim);
        for (Index t = 0; t < elim_view.size; ++t) {
          const Index other_row = elim_view.rows[t];
          if (other_row == i) continue;
          const auto orr = static_cast<std::size_t>(other_row);
          if (work.row_dead[orr]) continue;
          const double a_other = elim_view.values[t];
          if (std::fabs(a_other) < tol::kZeroDrop) continue;

          const double factor = a_other / a;  // a_i'e / a
          if (finite(work.row_lower[orr])) work.row_lower[orr] -= factor * rhs;
          if (finite(work.row_upper[orr])) work.row_upper[orr] -= factor * rhs;

          const double delta_keep = -factor * b;
          bool found = false;
          for (auto& entry : work.rows[orr]) {
            if (entry.first == keep) {
              entry.second += delta_keep;
              found = true;
              break;
            }
          }
          work.extra_row_delta[uk][other_row] += delta_keep;
          if (!found) {
            work.rows[orr].emplace_back(keep, delta_keep);
            work.extra_new_rows[uk].push_back(other_row);
            ++work.row_count[orr];
            ++work.col_count[uk];
          }
          // The eliminated column's own entry in this row is left in place but is now
          // meaningless - every reader of work.rows[] already skips dead columns, and elim
          // is marked dead below, before this row is examined again.
          --work.row_count[orr];
        }

        work.col_dead[ue] = true;
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
    reduced.col_cost.push_back(work.col_cost[u]);
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

  // The objective CONSTANT contributed by fixed and empty columns is folded in here.
  // Forgetting this is the classic presolve bug: every reported objective comes back short
  // by exactly the cost of the variables that were removed, on every instance, and it looks
  // like a solver accuracy problem rather than a bookkeeping one.
  //
  // work.col_cost, not model.col_cost: a column fixed or emptied in a LATER pass may already
  // have had its cost changed by an EARLIER free-column-singleton or doubleton-equation fold
  // (both mutate work.col_cost for whichever columns keep sharing their row), and it is that
  // post-fold cost the removed value actually paid.
  for (const Record& record : result.records) {
    if (record.kind == Record::Kind::kFixedColumn ||
        record.kind == Record::Kind::kEmptyColumn) {
      reduced.objective_offset +=
          work.col_cost[static_cast<std::size_t>(record.index)] * record.value;
    }
  }
  reduced.objective_offset += work.objective_offset_delta;

  // Built from work.rows, NOT model.matrix directly: doubleton-equation fill-in mutates
  // work.rows (a survivor's row picks up the eliminated column's OTHER row memberships), and
  // that is the structure that must reach the engine, not the original untouched matrix.
  reduced.matrix.reset(reduced_rows, reduced_cols);
  for (Index i = 0; i < m; ++i) {
    const auto r = static_cast<std::size_t>(i);
    if (work.row_dead[r]) continue;
    const Index mapped_row = new_row_index[r];
    for (const auto& entry : work.rows[r]) {
      const auto col = static_cast<std::size_t>(entry.first);
      if (work.col_dead[col]) continue;
      const Index mapped_col = new_col_index[col];
      if (mapped_col >= 0) reduced.matrix.add_entry(mapped_row, mapped_col, entry.second);
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

  // Which position (index into result.records) removed each column, for columns that were
  // removed at all - needed below to reconstruct a free-column-singleton's row correctly.
  // fold_fixed_column() (presolve()) already subtracted an EARLIER-removed column's known
  // contribution out of the row bounds at the time it was folded, via substituted_rhs; a
  // free-column-singleton row can carry several other columns, and summing an
  // already-folded one's value back in here would double-count it. A column removed LATER
  // (by position) is the opposite case the file already handles: its value is not yet known
  // at fold time either way, but reverse order guarantees it is finalised before we need it.
  std::vector<std::size_t> column_removed_at(static_cast<std::size_t>(original.num_cols()),
                                             result.records.size());
  for (std::size_t p = 0; p < result.records.size(); ++p) {
    const Record& rec = result.records[p];
    Index removed = -1;
    switch (rec.kind) {
      case Record::Kind::kFixedColumn:
      case Record::Kind::kEmptyColumn: removed = rec.index; break;
      case Record::Kind::kFreeColumnSingleton:
      case Record::Kind::kDoubletonEquation: removed = rec.column; break;
      default: break;
    }
    if (removed >= 0) column_removed_at[static_cast<std::size_t>(removed)] = p;
  }

  // Every kind that removes a row entirely, by the position that removed it - not just the
  // two new kinds row_is_folded already tracks. A doubleton's fill-in walks the ELIMINATED
  // column's ORIGINAL matrix view to find every OTHER row it touches, with no idea that one
  // of those rows might have been killed by an EARLIER record (a free-column-singleton, a
  // plain singleton row, anything) before this doubleton ever fired in presolve() - presolve
  // itself skips a dead row there (`work.row_dead[orr]`), and postsolve's reconstruction of
  // that same fill-in needs the identical guard, or it invents a relationship between a
  // column and a row that had already stopped existing.
  std::vector<std::size_t> row_removed_at(static_cast<std::size_t>(original.num_rows()),
                                          result.records.size());
  for (std::size_t p = 0; p < result.records.size(); ++p) {
    const Record& rec = result.records[p];
    switch (rec.kind) {
      case Record::Kind::kEmptyRow:
      case Record::Kind::kRedundantRow:
      case Record::Kind::kForcingRow:
      case Record::Kind::kSingletonRow:
      case Record::Kind::kFreeColumnSingleton:
      case Record::Kind::kDoubletonEquation:
        row_removed_at[static_cast<std::size_t>(rec.index)] = p;
        break;
      default: break;
    }
  }

  // Row-wise view of the ORIGINAL matrix, built lazily: only a free-column-singleton's row
  // can have more than the one partner a doubleton already carries in its own record.
  CsrView original_rows;
  bool have_original_rows = false;
  const auto ensure_original_rows = [&]() {
    if (!have_original_rows) {
      original_rows.build(original.matrix);
      have_original_rows = true;
    }
  };

  // ADJUSTED COST AND FOLDED ROWS, replayed in FORWARD order - the same order presolve()
  // itself applied them in.
  //
  // A free-column-singleton or doubleton-equation row does not just disappear: it is
  // algebraically ELIMINATED, and its entire effect on every OTHER column it touched is
  // folded into that column's cost (work.col_cost, in presolve()). Once that fold has
  // happened, the row's own dual has NOTHING further to contribute to pricing any OTHER
  // column - using it again there would double an effect the fold already applied.
  //
  // A later reduction can still end up pricing one of those OTHER columns (here: a doubleton
  // folds into a column that a later pass turns into a plain singleton row). reduced_cost_of
  // below computed that price from Model::col_cost and every ORIGINAL row unconditionally,
  // which is exactly the state before the fold, not after it - on a 2-row, 2-column model
  // with one doubleton feeding a singleton row, that put a factor-of-two error in one price,
  // caught immediately by Solve.DispatchesAnLpToTheSimplex once the two new reductions had
  // any interaction to exercise at all. adjusted_cost and row_is_folded are the fix: use the
  // POST-fold cost, and never re-read a folded row's dual as if it still applied directly.
  //
  // DOUBLETON FILL-IN ALSO CHANGES COEFFICIENTS, not just costs: every OTHER row the
  // eliminated column touched picks up an adjusted coefficient on the partner column (see
  // presolve()'s own fill-in loop). original.matrix still holds the PRE-fill-in coefficient
  // for that (row, partner) pair, so anything pricing the partner through one of those rows
  // needs the adjusted value - row_overrides records the accumulated delta per (row,
  // column), and effective_coefficient() below folds it in, whether the partner already had
  // an entry there or picked one up for the first time.
  std::vector<double> adjusted_cost = original.col_cost;
  std::vector<bool> row_is_folded(static_cast<std::size_t>(original.num_rows()), false);
  // Which columns actually had THIS row's fold baked into their adjusted_cost above - a
  // doubleton's partner, or whichever of a free-column-singleton's row-mates were still LIVE
  // when it fired. A column that ALSO has an original entry in a folded row but was NOT one
  // of those recipients - because something else (a fixed-column reduction, say) had already
  // removed it from that row BEFORE the fold ever ran - never had this row's effect folded
  // into it anywhere, and skipping the row for such a column below would simply drop its
  // true contribution rather than avoid double-counting it. Netlib's `bandm`, column ORROLC:
  // fixed by an EARLIER singleton-row equality, so already dead in row 49 by the time that
  // row's doubleton fired - yet ORROLC's ORIGINAL coefficient there is very much real, and
  // the independent verifier's reduced-cost check is computed against the ORIGINAL matrix,
  // which has no idea presolve ever removed it.
  std::vector<std::vector<Index>> folded_row_recipients(
      static_cast<std::size_t>(original.num_rows()));
  std::vector<std::unordered_map<Index, double>> row_overrides(
      static_cast<std::size_t>(original.num_rows()));
  // Rows a column picked up a fill-in entry in that it did NOT originally appear in - the
  // only case iterating original.matrix.column(column) below would miss entirely.
  std::vector<std::vector<Index>> extra_rows_for_column(
      static_cast<std::size_t>(original.num_cols()));
  // The mirror image, indexed by row: columns THAT row picked up via fill-in and did not
  // originally contain. A free-column-singleton's primal reconstruction walks its row to sum
  // every other column's contribution, and original_rows (built from the ORIGINAL matrix)
  // has no idea a doubleton's fill-in put a brand-new column there - missing it silently
  // drops a term from the row-activity equation, which is exactly what let a chain of THREE
  // free-column-singleton folds, one of them into a row a doubleton had just filled in,
  // reconstruct a primal point that violated the very row it claimed to satisfy.
  std::vector<std::vector<Index>> extra_columns_for_row(
      static_cast<std::size_t>(original.num_rows()));
  for (std::size_t p = 0; p < result.records.size(); ++p) {
    const Record& rec = result.records[p];
    if (rec.kind != Record::Kind::kFreeColumnSingleton &&
        rec.kind != Record::Kind::kDoubletonEquation) {
      continue;
    }
    row_is_folded[static_cast<std::size_t>(rec.index)] = true;
    if (rec.kind == Record::Kind::kDoubletonEquation) {
      // Exactly one partner, no liveness question to ask.
      adjusted_cost[static_cast<std::size_t>(rec.partner_column)] -=
          rec.eliminated_cost * (rec.partner_coefficient / rec.coefficient);
      folded_row_recipients[static_cast<std::size_t>(rec.index)].push_back(rec.partner_column);

      const ColumnView elim_view = original.matrix.column(rec.column);
      for (Index t = 0; t < elim_view.size; ++t) {
        const Index other_row = elim_view.rows[t];
        if (other_row == rec.index) continue;
        // A row already gone by the time THIS doubleton fired in presolve() (any kind - an
        // earlier free-column-singleton, a plain singleton row, anything) has nothing left
        // to fill in - presolve()'s own live fill-in loop skips it via `work.row_dead`, and
        // row_removed_at[other_row] < p is that same fact seen from postsolve's side.
        if (row_removed_at[static_cast<std::size_t>(other_row)] < p) continue;
        const double a_other = elim_view.values[t];
        if (std::fabs(a_other) < tol::kZeroDrop) continue;
        const double delta = -(a_other / rec.coefficient) * rec.partner_coefficient;
        auto& overrides_here = row_overrides[static_cast<std::size_t>(other_row)];
        const bool is_new_key = overrides_here.find(rec.partner_column) == overrides_here.end();
        overrides_here[rec.partner_column] += delta;
        if (is_new_key && original.matrix.at(other_row, rec.partner_column) == 0.0) {
          extra_rows_for_column[static_cast<std::size_t>(rec.partner_column)].push_back(
              other_row);
          extra_columns_for_row[static_cast<std::size_t>(other_row)].push_back(
              rec.partner_column);
        }
      }
    } else {
      // A free-column-singleton's row can have several other columns; fold into whichever
      // ones were still live in that row WHEN THIS RECORD FIRED - the same guard the primal
      // reconstruction above needs, and for the same reason: one removed earlier already
      // has this fold baked into `substituted_rhs` from the row-bound side, so folding its
      // cost again here too would be the same double-count from the other direction.
      ensure_original_rows();
      const ColumnView row_view = original_rows.row(rec.index);
      const auto& overrides_here = row_overrides[static_cast<std::size_t>(rec.index)];
      for (Index t = 0; t < row_view.size; ++t) {
        const Index col = row_view.rows[t];
        if (col == rec.column) continue;
        const auto uc = static_cast<std::size_t>(col);
        if (column_removed_at[uc] < p) continue;  // already gone when this record fired
        double coefficient = row_view.values[t];
        const auto found = overrides_here.find(col);
        if (found != overrides_here.end()) coefficient += found->second;
        adjusted_cost[uc] -= rec.eliminated_cost * (coefficient / rec.coefficient);
        folded_row_recipients[static_cast<std::size_t>(rec.index)].push_back(col);
      }
      // A column an EARLIER doubleton's fill-in added to this row - not in the ORIGINAL row
      // at all, so the loop above never sees it - still shares this row and must receive the
      // same cost fold.
      for (const Index col : extra_columns_for_row[static_cast<std::size_t>(rec.index)]) {
        if (col == rec.column) continue;
        const auto uc = static_cast<std::size_t>(col);
        if (column_removed_at[uc] < p) continue;
        const auto found = overrides_here.find(col);
        if (found == overrides_here.end()) continue;
        adjusted_cost[uc] -= rec.eliminated_cost * (found->second / rec.coefficient);
        folded_row_recipients[static_cast<std::size_t>(rec.index)].push_back(col);
      }
    }
  }

  // Replay in REVERSE for the PRIMAL values. A column fixed in pass 3 may sit in a row
  // removed in pass 1, so undoing them in application order would price a row against a
  // point that does not exist yet.
  for (std::size_t idx = result.records.size(); idx-- > 0;) {
    const Record& record = result.records[idx];
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
      case Record::Kind::kFreeColumnSingleton: {
        // x_j = (target - sum of every OTHER live-at-elimination-time column's contribution)
        // / a. "Live at elimination time" is the guard: a column removed EARLIER (lower
        // position) had its contribution already folded into `target` when this row's bound
        // was adjusted, and including it again here would double it.
        //
        // The row's coefficients are read from `original_rows` and then patched by
        // `row_overrides[record.index]` - a doubleton's fill-in can have adjusted an entry
        // already there, or added a column that was not in the ORIGINAL row at all
        // (`extra_columns_for_row` is exactly the columns original_rows would otherwise miss).
        ensure_original_rows();
        const ColumnView row_view = original_rows.row(record.index);
        const auto& overrides_here = row_overrides[static_cast<std::size_t>(record.index)];
        double sum = 0.0;
        for (Index t = 0; t < row_view.size; ++t) {
          const Index col = row_view.rows[t];  // CsrView reuses ColumnView: rows[] holds
                                               // COLUMN indices for a row view.
          if (col == record.column) continue;
          const auto uc = static_cast<std::size_t>(col);
          if (column_removed_at[uc] < idx) continue;  // already folded into `target`
          double coefficient = row_view.values[t];
          const auto found = overrides_here.find(col);
          if (found != overrides_here.end()) coefficient += found->second;
          sum += coefficient * solution.col_value[uc];
        }
        for (const Index col : extra_columns_for_row[static_cast<std::size_t>(record.index)]) {
          if (col == record.column) continue;
          const auto uc = static_cast<std::size_t>(col);
          if (column_removed_at[uc] < idx) continue;
          const auto found = overrides_here.find(col);
          if (found == overrides_here.end()) continue;
          sum += found->second * solution.col_value[uc];
        }
        const double xj = (record.substituted_rhs - sum) / record.coefficient;
        solution.col_value[static_cast<std::size_t>(record.column)] = xj;
        solution.col_status[static_cast<std::size_t>(record.column)] = BasisStatus::kBasic;
        // Placeholder; refined below, in the duals pass, exactly like kSingletonRow.
        solution.row_dual[static_cast<std::size_t>(record.index)] = 0.0;
        solution.row_status[static_cast<std::size_t>(record.index)] = BasisStatus::kBasic;
        break;
      }
      case Record::Kind::kDoubletonEquation: {
        // x_elim = (rhs - b*x_keep) / a. `keep` needs no liveness guard: the doubleton
        // trigger itself required both columns live at that moment, and `keep` survives
        // into the reduced model (or is eliminated LATER, by position, so reverse order has
        // already finalised it either way).
        const double xk = solution.col_value[static_cast<std::size_t>(record.partner_column)];
        const double xe =
            (record.substituted_rhs - record.partner_coefficient * xk) / record.coefficient;
        solution.col_value[static_cast<std::size_t>(record.column)] = xe;
        solution.col_status[static_cast<std::size_t>(record.column)] = BasisStatus::kBasic;
        solution.row_dual[static_cast<std::size_t>(record.index)] = 0.0;
        solution.row_status[static_cast<std::size_t>(record.index)] = BasisStatus::kBasic;
        break;
      }
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
  // The coefficient of (row, column) AFTER doubleton fill-in, which original.matrix does not
  // reflect: it patches the partner column's coefficient in every row the eliminated column
  // used to touch. Falls back to the plain original value everywhere fill-in never reached.
  const auto effective_coefficient = [&](Index row, Index column, double original_value) {
    const auto& overrides_here = row_overrides[static_cast<std::size_t>(row)];
    const auto found = overrides_here.find(column);
    return found == overrides_here.end() ? original_value : original_value + found->second;
  };
  const auto reduced_cost_of = [&](Index column) {
    // adjusted_cost, not original.col_cost: a column whose row was folded away by
    // kFreeColumnSingleton or kDoubletonEquation carries that fold's cost adjustment here,
    // and row_is_folded skips re-reading that SAME row's dual below - together they price
    // this column exactly as if the fold had never introduced an extra row to account for.
    double d = adjusted_cost[static_cast<std::size_t>(column)];
    const ColumnView view = original.matrix.column(column);
    for (Index k = 0; k < view.size; ++k) {
      const Index row = view.rows[k];
      // A folded row's effect is already baked into adjusted_cost, but ONLY for the columns
      // that were actually LIVE in it when the fold fired (folded_row_recipients). A column
      // that also has an original entry there but was already gone by fold time - fixed by
      // an earlier, unrelated reduction, say - never received that bake-in, so its true
      // contribution from this row must still be counted normally, from whatever dual this
      // row ends up with (see folded_row_recipients' comment above).
      if (row_is_folded[static_cast<std::size_t>(row)]) {
        const auto& recipients = folded_row_recipients[static_cast<std::size_t>(row)];
        const bool is_recipient =
            std::find(recipients.begin(), recipients.end(), column) != recipients.end();
        if (is_recipient) continue;
      }
      const double coefficient = effective_coefficient(row, column, view.values[k]);
      d -= coefficient * solution.row_dual[static_cast<std::size_t>(row)];
    }
    // Rows fill-in added this column to for the first time - absent from `view` entirely.
    for (const Index row : extra_rows_for_column[static_cast<std::size_t>(column)]) {
      if (row_is_folded[static_cast<std::size_t>(row)]) continue;
      const double coefficient = effective_coefficient(row, column, 0.0);
      d -= coefficient * solution.row_dual[static_cast<std::size_t>(row)];
    }
    return d;
  };

  // Columns this loop, and the two-kind loop below it, finalise col_dual for directly - the
  // generic "reduced costs for the REMOVED columns only" pass further down must NOT
  // recompute these, because it does so via reduced_cost_of() alone, which has no way to
  // fold in a price this row was JUST given. Declared here, ahead of BOTH loops that can set
  // it, so a doubleton's `keep` that was ALSO priced by one of its OTHER rows turning into a
  // kSingletonRow (fill-in can do that) is recognised as already spoken for by the time the
  // doubleton loop reaches it - see the comment there.
  std::vector<bool> dual_finalized(static_cast<std::size_t>(original.num_cols()), false);

  // A column that PARTICIPATES in some doubleton fold - as its survivor OR as the column it
  // eliminates, even one whose own row has not been priced yet by the time this loop reaches
  // it - must not be finalised by a kSingletonRow decision that merely declines to price ITS
  // row. Declining only means "reduced_cost_of, as computed with this row's OWN contribution
  // held at zero, is already admissible or this row is not active" - it says nothing about
  // whether the STILL-UNDECIDED doubleton row is also owed a contribution. kSingletonRow's
  // own loop runs unconditionally before the doubleton loop below regardless of which record
  // was pushed later, so without this a doubleton whose `keep` also picked up an unrelated,
  // ultimately-inactive kSingletonRow (fill-in can cause this indirectly, without keep even
  // being that row's OWN column) would have its price wrongly locked in before the doubleton
  // ever gets a chance to supply it - and the ELIMINATED column is exactly as exposed: it
  // keeps its OWN real bounds and can independently need pricing from a kSingletonRow that
  // ALSO happens to constrain it (an implied-bound-tightening singleton row on a column a
  // LATER doubleton then eliminates, say), and that row's own admissibility decision cannot
  // see the doubleton's not-yet-computed price any better than it could for a partner.
  std::vector<bool> is_doubleton_participant(static_cast<std::size_t>(original.num_cols()),
                                             false);
  for (const Record& rec : result.records) {
    if (rec.kind == Record::Kind::kDoubletonEquation) {
      is_doubleton_participant[static_cast<std::size_t>(rec.column)] = true;
      is_doubleton_participant[static_cast<std::size_t>(rec.partner_column)] = true;
    }
  }

  // A column can have an ORIGINAL entry in a folded row without that row's fold ever having
  // touched its cost - `folded_row_recipients` above is exactly the set that DID get baked
  // in, and anything outside it (bandm's ORROLC in row 49: fixed by an earlier, unrelated
  // singleton-row equality, so already gone from that row by the time its doubleton fired)
  // still needs the row's TRUE dual, not the zero placeholder this loop runs with. Since that
  // dual is only known once the free-column-singleton/doubleton loop below has run, a
  // kSingletonRow record for such a column cannot be safely decided here - it is deferred to
  // a second pass, after that loop, using the identical logic below.
  const auto depends_on_unresolved_fold = [&](Index column) {
    const ColumnView view = original.matrix.column(column);
    for (Index k = 0; k < view.size; ++k) {
      const Index row = view.rows[k];
      if (!row_is_folded[static_cast<std::size_t>(row)]) continue;
      const auto& recipients = folded_row_recipients[static_cast<std::size_t>(row)];
      if (std::find(recipients.begin(), recipients.end(), column) == recipients.end()) {
        return true;
      }
    }
    return false;
  };

  const auto process_singleton_row = [&](const Record& it) {
    const auto c = static_cast<std::size_t>(it.column);
    // This row's own placeholder is restored first. The dual passes below run to a fixed
    // point (#157), and on a repeat pass reduced_cost_of(it.column) would otherwise include
    // the price this very record set last time, pricing the row against itself. On the
    // first pass both are already the placeholder and this changes nothing.
    solution.row_dual[static_cast<std::size_t>(it.index)] = 0.0;
    solution.row_status[static_cast<std::size_t>(it.index)] = BasisStatus::kBasic;
    const double x = solution.col_value[c];
    const double lo = original.col_lower[c];
    const double hi = original.col_upper[c];
    const bool at_lower = finite(lo) && std::fabs(x - lo) <= tol::kPrimalFeasibility;
    const bool at_upper = finite(hi) && std::fabs(x - hi) <= tol::kPrimalFeasibility;

    const double d = reduced_cost_of(it.column);
    const double signed_d = sense * d;
    // Free to sit where it is, so the reduced cost must be zero; at a bound, only the wrong
    // sign needs paying for. A column already at an ORIGINAL bound with an admissible sign
    // needs no price on this row at all, and inventing one would break complementary
    // slackness on a row that is not active.
    const bool needs_price = (!at_lower && !at_upper) ||
                             (at_lower && !at_upper && signed_d < -tol::kDualFeasibility) ||
                             (at_upper && !at_lower && signed_d > tol::kDualFeasibility);
    // Every bail-out below leaves THIS row unpriced, which means d - computed with this
    // row's own contribution held at the zero placeholder - already IS the column's correct
    // final reduced cost UNLESS some OTHER kSingletonRow record for the same column already
    // priced it (the opposing-rows case just below finalises exactly one of two records this
    // way, and it may run before or after this one). Recording d here, guarded on not already
    // finalised, is what stops a LATER pass - the free-column-singleton/doubleton loop below
    // this one - from treating the column as still unpriced and re-deriving a DIFFERENT value
    // from its own row instead. A doubleton whose fill-in turned the survivor's other row
    // into exactly this kind of declined singleton row is what caught the gap: the doubleton
    // loop had no way to know this row had already settled the question.
    if (!needs_price) {
      if (!dual_finalized[c] && !is_doubleton_participant[c]) {
        solution.col_dual[c] = d;
        dual_finalized[c] = true;
      }
      return;
    }

    // THE ROW MUST ACTUALLY BE ACTIVE. Complementary slackness forbids a price on a
    // constraint that is not binding, and a column can be constrained by SEVERAL singleton
    // rows - a lower bound from one and an upper from another. Without this test the first
    // record reached wins, and on `-3*x1 >= -9` together with `2*x1 >= 2` at x1 = 3 the
    // price landed on the second, which is slack by 4. The oracle reported 59 degenerate
    // instances where the optimum existed and the solver returned merely `feasible`; this
    // was all of them.
    const double activity = it.coefficient * x;
    const bool row_at_lower =
        finite(it.row_lower) && std::fabs(activity - it.row_lower) <= tol::kPrimalFeasibility;
    const bool row_at_upper =
        finite(it.row_upper) && std::fabs(activity - it.row_upper) <= tol::kPrimalFeasibility;
    if (!row_at_lower && !row_at_upper) {
      if (!dual_finalized[c] && !is_doubleton_participant[c]) {
        solution.col_dual[c] = d;
        dual_finalized[c] = true;
      }
      return;
    }

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
    // time round - and if there IS no next record to correct it, this row genuinely was not
    // the one that needed to act, so d stands as the finalised answer, same as above.
    const double candidate = d / it.coefficient;
    const double signed_candidate = sense * candidate;
    if (row_at_lower && !row_at_upper && signed_candidate < -tol::kDualFeasibility) {
      if (!dual_finalized[c] && !is_doubleton_participant[c]) {
        solution.col_dual[c] = d;
        dual_finalized[c] = true;
      }
      return;
    }
    if (row_at_upper && !row_at_lower && signed_candidate > tol::kDualFeasibility) {
      if (!dual_finalized[c] && !is_doubleton_participant[c]) {
        solution.col_dual[c] = d;
        dual_finalized[c] = true;
      }
      return;
    }

    solution.row_dual[static_cast<std::size_t>(it.index)] = candidate;
    solution.row_status[static_cast<std::size_t>(it.index)] =
        row_at_lower ? BasisStatus::kAtLower : BasisStatus::kAtUpper;
    solution.col_status[c] = BasisStatus::kBasic;
    dual_finalized[c] = true;
    // The price was chosen precisely to cancel this column's reduced cost, so set it to
    // exactly zero rather than leaving a rounded residue for the verifier to trip over.
    solution.col_dual[c] = 0.0;
    // This row absorbed c's stationarity outright. A doubleton's fill-in can turn c's OTHER
    // row into exactly this kind of singleton row, and if the doubleton loop below then
    // re-derived c's dual too - from ITS OWN row instead - the two would disagree about
    // which row actually did the pricing. Claiming c here is what stops that.
    dual_finalized[c] = true;
  };

  // THE DUAL PASSES RUN TO A FIXED POINT (#157). Three stages follow: the singleton rows
  // in reverse record order, then the free-column-singleton and doubleton folds in reverse
  // record order, then the singleton rows whose column touches a folded row it never
  // received the fold of (deferred, see depends_on_unresolved_fold). Every stage prices a
  // row from the duals known at that moment, and a row priced in a LATER stage can change
  // the reduced cost of a column priced in an earlier one.
  //
  // ganges: CONT4701 (= 160) pins X4701, which also has a -1 in CONT4601. CONT4601 became
  // a singleton row only after X4701 and X4801 were fixed, and its own column sits in the
  // doubleton row CONT4501, so its record is deferred past the folds. CONT4701's record
  // is not - neither of X4701's rows is folded - and is priced in the first stage with
  // CONT4601 still at the placeholder 0: d = 0, declined. CONT4601 is then priced at
  // -0.714 in the deferred stage, and X4701, interior, ends with d = -0.714. Twelve
  // columns of that shape on ganges, and the same story on perold, pilot, greenbea and
  // greenbeb: the point right, the certificate rejected.
  //
  // Records depend only on records replayed before them in reverse order - a row removed
  // BEFORE a column was fixed either had that column as its only entry (then it is the
  // fixing row itself), or folded its cost into the column (then reduced_cost_of skips
  // it), or was redundant (dual 0) - so the dependency graph is acyclic, and repeating the
  // three stages is a Gauss-Seidel iteration that reaches the exact fixed point in as
  // many passes as the graph is deep, with identical arithmetic on every pass after that
  // (so the change test below hits exactly zero; it is not a tolerance race). Each pass
  // re-decides from scratch: dual_finalized is cleared, and every priced row restores its
  // own placeholder before pricing itself again, so no row is priced against its own
  // previous price. Pass 1 is the old behaviour exactly; the passes after it are what
  // fixes the five instances above. Bounded, because a bug in the acyclicity argument
  // must show up as a wrong certificate and not as a hang.
  constexpr int kMaxDualPasses = 8;
  std::vector<const Record*> deferred_singleton_rows;
  for (int pass = 0; pass < kMaxDualPasses; ++pass) {
    const std::vector<double> row_dual_before = solution.row_dual;
    std::fill(dual_finalized.begin(), dual_finalized.end(), false);
    deferred_singleton_rows.clear();

    for (auto it = result.records.rbegin(); it != result.records.rend(); ++it) {
      if (it->kind != Record::Kind::kSingletonRow) continue;
      if (std::fabs(it->coefficient) <= tol::kZeroDrop) continue;
      if (depends_on_unresolved_fold(it->column)) {
        deferred_singleton_rows.push_back(&*it);
        continue;
      }
      process_singleton_row(*it);
    }

    // kFreeColumnSingleton and kDoubletonEquation duals.
    //
    // A free-column-singleton's column is UNBOUNDED on both sides by definition, so it is
    // always interior at the optimum and its reduced cost must be EXACTLY zero - that alone
    // pins the row's dual from stationarity. A doubleton's eliminated column carries its own
    // real bounds, though, and can legitimately end up sitting AT one of them (the very first
    // fuzzed instance to combine a doubleton with a resulting singleton row did exactly this),
    // in which case zero is not what its reduced cost has to be - only SIGN-admissible, exactly
    // the question kSingletonRow above already answers for its own column. So both kinds share
    // that same logic here rather than assuming the zero case unconditionally: price the row
    // only if the column's natural reduced cost (row's own dual still the placeholder 0, via
    // row_is_folded) is not already admissible for whichever bound it is at - and because this
    // row is always an EQUALITY, the price it would need has no sign restriction to fail: it is
    // adopted outright once we have decided a price is needed at all.
    for (std::size_t idx = result.records.size(); idx-- > 0;) {
      const Record& record = result.records[idx];
      if (record.kind != Record::Kind::kFreeColumnSingleton &&
          record.kind != Record::Kind::kDoubletonEquation) {
        continue;
      }
      if (std::fabs(record.coefficient) <= tol::kZeroDrop) continue;
      // Restore this row's placeholder before pricing it again. reduced_cost_of() excludes a
      // folded row only for the columns that received its cost fold, and the eliminated
      // column is not one of them - so on a repeat pass its reduced cost would include the
      // price this record itself set last time, and the row would be priced against its
      // own price. Zeroing it first makes each pass recompute the row from the OTHER rows'
      // current duals, which is the Gauss-Seidel step this loop is repeated for. On the
      // first pass it is already zero and this changes nothing.
      solution.row_dual[static_cast<std::size_t>(record.index)] = 0.0;
      const auto c = static_cast<std::size_t>(record.column);

      // Doubleton, unless `keep` was already priced by something else entirely (a kSingletonRow
      // from fill-in turning keep's OTHER row into one - see dual_finalized's declaration).
      //
      // THE IDENTITY THIS RELIES ON. reduced_cost_of(elim) and reduced_cost_of(keep) both fold
      // in every OTHER live row each column touches - an untouched original row, or one fill-in
      // gave it, effective_coefficient and extra_rows_for_column make no difference between the
      // two. That is a pure change of variables (substituting elim = (rhs - b*keep)/a into
      // every row it used to touch), so it does NOT introduce a second unknown: it can be shown
      // algebraically that reduced_cost_of(keep), computed this way, is EXACTLY d_keep
      // evaluated at Y = y_from_elim = reduced_cost_of(elim)/a - the same Y that would zero
      // elim's own reduced cost, regardless of whether elim actually NEEDS to be zero there.
      // That gives the general formula
      //     d_elim(Y) = reduced_cost_of(elim) - a*Y
      //     d_keep(Y) = reduced_cost_of(keep) - b*(Y - y_from_elim)
      // valid whether or not keep has any other row at all (when it has none, reduced_cost_of
      // (keep) is just adjusted_cost[keep] and y_from_elim collapses to the old eliminated_cost
      // / a "baseline" - the same formula, just written generally).
      //
      // A version of this that anchored keep's formula on eliminated_cost/a UNCONDITIONALLY,
      // and skipped touching keep's dual whenever it had another row on the theory that
      // reduced_cost_of(keep) was already final, passed every hand-written test but failed a
      // fuzzed instance where `elim` sat AT its own bound (not interior) while `keep` - reached
      // only through fill-in - was genuinely interior: the row's price was decided from elim
      // alone, correctly admissible for elim, while leaving keep's already-computed (but
      // WRONGLY anchored) reduced cost nonzero. y_from_elim, not eliminated_cost/a, is the
      // correct anchor whenever elim itself has other rows of its own - and using it costs
      // nothing in the simpler case, since the two coincide there.
      //
      // UNLIKE A FREE-COLUMN-SINGLETON, THE ELIMINATED COLUMN IS NOT ASSUMED FREE: it keeps its
      // own real bounds, so it - not just the partner - can independently force this row's
      // dual. `min x + 2y s.t. x + y = 4, x,y >= 0` is the minimal case: y sits at its own
      // bound with an admissible sign no matter what the row's dual is, so a check that only
      // asked about y concluded no price was needed - but x is INTERIOR (x = 4), which requires
      // its reduced cost to be EXACTLY zero regardless of y, and that pins the row's dual to a
      // specific value y could never have asked for on its own.
      //
      // An interior column pins Y outright (its d must be exactly 0); at a bound it only
      // constrains Y's SIGN. Interior beats at-a-bound as the thing that decides Y, because
      // "exactly zero" leaves no freedom to also satisfy a sign constraint from Y = 0 the way
      // an already-admissible at-bound column does.
      if (record.kind == Record::Kind::kDoubletonEquation &&
          !dual_finalized[static_cast<std::size_t>(record.partner_column)]) {
        const auto pc = static_cast<std::size_t>(record.partner_column);
        const double a = record.coefficient;
        const double b = record.partner_coefficient;

        const auto status_of = [&](Index column) {
          const auto u = static_cast<std::size_t>(column);
          const double v = solution.col_value[u];
          const double clo = original.col_lower[u];
          const double chi = original.col_upper[u];
          const bool at_lo = finite(clo) && std::fabs(v - clo) <= tol::kPrimalFeasibility;
          const bool at_hi = finite(chi) && std::fabs(v - chi) <= tol::kPrimalFeasibility;
          return std::pair<bool, bool>(at_lo, at_hi);
        };
        const auto [elim_at_lo, elim_at_hi] = status_of(record.column);
        const auto [keep_at_lo, keep_at_hi] = status_of(record.partner_column);
        const bool elim_interior = !elim_at_lo && !elim_at_hi;
        const bool keep_interior = !keep_at_lo && !keep_at_hi;

        const double rco_elim = reduced_cost_of(record.column);
        const double rco_keep = reduced_cost_of(record.partner_column);
        // Y that forces d_elim(Y) = 0, and Y that forces d_keep(Y) = 0 respectively. rco_keep
        // is anchored at Y = y_from_elim (see the derivation above), not at eliminated_cost/a.
        const double y_from_elim = rco_elim / a;
        const double y_from_keep = y_from_elim + rco_keep / b;

        // Admissibility of a candidate Y for whichever column is only AT a bound (never
        // called when that column is interior - interior always pins Y directly instead).
        const auto admissible_at_bound = [&](double d, bool at_lo, bool at_hi) {
          const double signed_d = sense * d;
          if (at_lo && !at_hi) return signed_d >= -tol::kDualFeasibility;
          if (at_hi && !at_lo) return signed_d <= tol::kDualFeasibility;
          return true;  // fixed (both) or free (neither, but that is the interior case)
        };

        double y = 0.0;
        if (elim_interior) {
          // x_elim pins Y outright; keep's admissibility at this Y follows from strong
          // duality at a genuinely optimal point; it is not an internal DOF this row still
          // has - checking is nonetheless the truthful move.
          y = y_from_elim;
        } else if (keep_interior) {
          y = y_from_keep;
        } else {
          // BOTH COLUMNS AT A BOUND: Y HAS TO SATISFY BOTH SIGN CONDITIONS AT ONCE (#157).
          //
          // The previous logic fixed one column's sign exactly and never re-checked the
          // other. On Netlib recipe, row BN44..BE eliminates JN43MXBE against JN43TGBE with
          // both ending at their lower bounds; the Y that zeroed one column's reduced cost
          // pushed the other's to -4.000e-03, an improving direction the simplex was never
          // shown. The point was optimal - HiGHS agrees to 1e-10 - and the reported duals
          // were not, by exactly 4e-3 times the column's range of 20: a strong-duality gap of
          // 8e-2 that only the verifier could see.
          //
          // Each at-bound column gives a HALF-LINE of admissible Y, because its reduced cost
          // is affine in Y:  d_elim(Y) = rco_elim - a*Y  and  d_keep(Y) = rco_keep - b*(Y -
          // y_from_elim). The intersection is an interval; any point in it is a valid price
          // for this row. Zero is preferred when it lies inside - complementary slackness
          // says an equality row that needs no price should carry none - and otherwise the
          // nearest endpoint, which is the smallest price that makes both signs admissible.
          // A fixed column (both bounds) constrains nothing. An empty interval means the
          // point is not dual feasible for any Y - which cannot happen at a true optimum -
          // and the old choice is kept so the answer is at least no worse than before.
          double y_min = -kInfinity;
          double y_max = kInfinity;
          // Constrain Y by the requirement on sense * d(Y), where d(Y) = r - g * (Y - anchor).
          const auto constrain = [&](double r, double g, double anchor, bool at_lo,
                                     bool at_hi) {
            if (at_lo == at_hi) return;   // fixed (both) or free (neither): no sign condition
            const double k = -sense * g;  // coefficient of Y in sense * d(Y)
            const double c0 = sense * (r + g * anchor);  // constant term
            if (std::fabs(k) <= tol::kZeroDrop) return;
            if (at_lo) {  // need k*Y + c0 >= 0, exactly: the tolerance is for judging, not for
                          // choosing, and choosing at the tolerance edge parks the reduced cost
                          // exactly where rounding tips it over
              const double bound = -c0 / k;
              if (k > 0.0)
                y_min = std::max(y_min, bound);
              else
                y_max = std::min(y_max, bound);
            } else {  // at_hi: need k*Y + c0 <= 0, exactly
              const double bound = -c0 / k;
              if (k > 0.0)
                y_max = std::min(y_max, bound);
              else
                y_min = std::max(y_min, bound);
            }
          };
          constrain(rco_elim, a, 0.0, elim_at_lo, elim_at_hi);
          constrain(rco_keep, b, y_from_elim, keep_at_lo, keep_at_hi);
          if (y_min <= y_max + tol::kDualFeasibility) {
            y = std::min(std::max(0.0, y_min), y_max);
          } else if (!admissible_at_bound(rco_elim, elim_at_lo, elim_at_hi)) {
            y = y_from_elim;
          } else {
            y = y_from_keep;
          }
        }

        solution.row_dual[static_cast<std::size_t>(record.index)] = y;
        solution.row_status[static_cast<std::size_t>(record.index)] = BasisStatus::kBasic;
        solution.col_dual[pc] = rco_keep - b * (y - y_from_elim);
        solution.col_status[pc] = BasisStatus::kBasic;
        dual_finalized[pc] = true;
        solution.col_dual[c] = rco_elim - a * y;
        dual_finalized[c] = true;
        continue;
      }

      const double x = solution.col_value[c];
      const double lo = original.col_lower[c];
      const double hi = original.col_upper[c];
      const bool at_lower = finite(lo) && std::fabs(x - lo) <= tol::kPrimalFeasibility;
      const bool at_upper = finite(hi) && std::fabs(x - hi) <= tol::kPrimalFeasibility;

      const double d = reduced_cost_of(record.column);
      const double signed_d = sense * d;
      const bool needs_price = (!at_lower && !at_upper) ||
                               (at_lower && !at_upper && signed_d < -tol::kDualFeasibility) ||
                               (at_upper && !at_lower && signed_d > tol::kDualFeasibility);
      if (!needs_price) {
        // Already admissible without this row's help - it really is redundant, exactly like a
        // kRedundantRow, and complementary slackness forbids inventing a price for it anyway.
        solution.row_dual[static_cast<std::size_t>(record.index)] = 0.0;
        solution.row_status[static_cast<std::size_t>(record.index)] = BasisStatus::kBasic;
        solution.col_dual[c] = d;
        dual_finalized[c] = true;
        continue;
      }

      solution.row_dual[static_cast<std::size_t>(record.index)] = d / record.coefficient;
      solution.row_status[static_cast<std::size_t>(record.index)] = BasisStatus::kBasic;
      solution.col_status[c] = BasisStatus::kBasic;
      // The price was chosen precisely to cancel this column's reduced cost, so set it to
      // exactly zero rather than leaving a rounded residue for the verifier to trip over.
      solution.col_dual[c] = 0.0;
      dual_finalized[c] = true;
    }

    // The kSingletonRow records deferred above, now that every folded row above has a REAL
    // dual instead of the zero placeholder - process_singleton_row's reduced_cost_of call
    // reads exactly the same solution.row_dual it always did, but the entries that matter for
    // these deferred columns are no longer placeholders.

    for (const Record* rec : deferred_singleton_rows) {
      process_singleton_row(*rec);
    }

    double largest_change = 0.0;
    for (std::size_t r = 0; r < solution.row_dual.size(); ++r) {
      largest_change =
          std::max(largest_change, std::fabs(solution.row_dual[r] - row_dual_before[r]));
    }
    if (largest_change == 0.0) break;
  }

  // THE REDUCED COSTS ARE RECOMPUTED FROM THE FINAL ROW DUALS, ALL OF THEM (#157), AND
  // BEFORE THE QUALITY IS MEASURED. An earlier version ran this after recompute_quality():
  // the dispatcher's status guard then judged reduced costs this pass had already replaced,
  // downgraded perold, greenbea and greenbeb to `feasible` for a dual violation of 0.71,
  // and wrote a .sol file the independent verifier passed. The solver was refusing to
  // certify an answer over numbers that no longer existed.
  //
  // This pass replaced one that recomputed the REMOVED columns only, on the grounds that
  // a survivor's reduced cost was already right and recomputing it through different
  // floating-point operations turned an exact 0.0 on a free column of capri into -1.1e-16,
  // which the verifier's |d| * slack product, with slack infinite, evaluated to inf. The
  // verifier now judges an infinite slack by |d| alone, so that reason is gone.
  //
  // Everything above sets col_dual as records are replayed, in reverse order, and several
  // of those replays price a column with reduced_cost_of() at a moment when some row it
  // touches still carries the placeholder dual of 0 - the free-column-singleton and
  // doubleton rows are only priced in the passes that follow. A column whose record ran
  // before those passes keeps a reduced cost computed against a row dual that later
  // changed. On ganges the reported d disagrees with c - A^T y by 0.71; on perold by 0.71,
  // greenbeb by 6.4. The point is right on every one - HiGHS agrees to 1e-10 - and the
  // certificate handed out with it is not, which is the exact failure class postsolve is
  // dangerous for.
  //
  // d = c - A^T y is not one property of the answer among several; it is the definition of
  // d. So once every row dual is final, every column's reduced cost is set from it. A
  // column the passes above priced correctly is unchanged by this; a column they priced
  // against a stale row dual is corrected; and if a ROW dual is itself wrong, that now
  // shows up as a sign violation on the columns it prices - which the status check and
  // the verifier both test - instead of hiding behind a reduced cost that agreed with
  // nothing.
  for (Index j = 0; j < original.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    double d = original.col_cost[u];
    const ColumnView view = original.matrix.column(j);
    for (Index k = 0; k < view.size; ++k) {
      d -= view.values[k] * solution.row_dual[static_cast<std::size_t>(view.rows[k])];
    }
    solution.col_dual[u] = d;
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

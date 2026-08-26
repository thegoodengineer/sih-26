// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve and postsolve tests.
//
// THE GATE HERE IS THE ROUND TRIP, not the reduction counts. A presolve that removes nothing
// is merely useless; a presolve whose postsolve is wrong returns a confident, feasible-
// looking answer to a DIFFERENT problem, with no crash and no stack trace. CLAUDE.md names
// that class of failure as the worst available outcome, alongside reporting a MILP's
// fractional relaxation as optimal.
//
// So every test below solves the SAME model twice, with presolve on and off, and requires
// the two to agree on the objective AND on dual feasibility. The second half matters as much
// as the first: three separate postsolve bugs found while writing this produced the right
// objective and wrong duals, and every one of them was caught by the status guard
// downgrading `optimal` to `feasible` rather than by a wrong number.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Options with_presolve(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", on);
  return options;
}

/// Build an LP from dense rows.
Model make_lp(const std::vector<std::vector<double>>& rows,
              const std::vector<double>& row_lower, const std::vector<double>& row_upper,
              const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

/// The whole point of this file: presolve must not change the answer.
void expect_agrees_with_unpresolved(const Model& model) {
  const Solution off = solve(model, with_presolve(false));
  const Solution on = solve(model, with_presolve(true));

  ASSERT_EQ(on.status, off.status) << "on: " << on.message << " / off: " << off.message;
  if (off.status != SolveStatus::kOptimal) return;

  EXPECT_NEAR(on.objective, off.objective, 1e-9 * std::max(1.0, std::fabs(off.objective)));
  // The recovered point must satisfy the ORIGINAL model, which is what recompute_quality
  // measured it against inside postsolve.
  EXPECT_LE(on.primal_infeasibility, tol::kPrimalFeasibility) << on.message;
  // And the recovered DUALS must be feasible too. Dropping this check would let every
  // postsolve bug found while writing this file through: each produced a correct objective.
  EXPECT_LE(on.dual_infeasibility, tol::kDualFeasibility) << on.message;
  EXPECT_NEAR(on.absolute_gap, 0.0, 1e-7) << "a solved LP has no gap";
}

// =========================================================================================

TEST(Presolve, FixedColumnIsFoldedIntoTheObjectiveConstant) {
  // x1 is pinned to 3 by its own bounds. Removing it must carry 2 * 3 into the objective;
  // forgetting that is the classic presolve bug, and it makes every objective short by
  // exactly the cost of what was removed - which reads as a solver accuracy problem.
  //   min 4*x0 + 2*x1  s.t.  x0 + x1 >= 5,  x1 in [3,3],  x0 >= 0
  //   x0 = 2, x1 = 3  ->  8 + 6 = 14
  const Model model =
      make_lp({{1.0, 1.0}}, {5.0}, {kInfinity}, {4.0, 2.0}, {0.0, 3.0}, {kInfinity, 3.0});
  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 14.0, 1e-9);
  EXPECT_NEAR(on.col_value[1], 3.0, 1e-9);
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, SingletonRowBecomesABoundAndKeepsItsDual) {
  // `2*x0 >= 6` is a bound on x0, not a constraint. Presolve turns it into one and drops the
  // row - but at the optimum that row IS active, so its dual has to come back or the answer
  // is dual infeasible on the original model.
  //   min 5*x0  s.t.  2*x0 >= 6  ->  x0 = 3, objective 15, row dual 2.5
  const Model model = make_lp({{2.0}}, {6.0}, {kInfinity}, {5.0}, {0.0}, {kInfinity});
  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 15.0, 1e-9);
  EXPECT_NEAR(on.row_dual[0], 2.5, 1e-9) << "the removed row must be priced";
  EXPECT_LE(on.dual_infeasibility, tol::kDualFeasibility);
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, TwoOpposingSingletonRowsPriceTheAdmissibleOne) {
  // A column pinned between `6*x1 >= 24` and `-3*x1 >= -12` - both hold with equality at
  // x1 = 4. Only one can carry the price: in minimise space a >= row active at its lower
  // bound needs a NON-NEGATIVE dual, and the second would need -2.
  //
  // Taking whichever record came first put the price on the wrong row. The objective was
  // still right, so only the dual check catches it.
  const Model model = make_lp({{0.0, 6.0}, {0.0, -3.0}, {-3.0, -6.0}}, {24.0, -12.0, -24.0},
                              {kInfinity, kInfinity, kInfinity}, {-5.0, 6.0}, {0.0, 0.0},
                              {kInfinity, kInfinity});
  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 24.0, 1e-9);
  EXPECT_LE(on.dual_infeasibility, tol::kDualFeasibility) << on.message;
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, RedundantRowIsRemovedAndPricedAtZero) {
  // Every coefficient positive, every variable non-negative, so `x0 + x1 >= -5` cannot bind.
  // A row removed because it cannot bind must come back with a dual of zero - that is what
  // "redundant" means, and complementary slackness forbids anything else.
  const Model model = make_lp({{1.0, 1.0}, {1.0, 0.0}}, {-5.0, 2.0}, {kInfinity, kInfinity},
                              {1.0, 1.0}, {0.0, 0.0}, {kInfinity, kInfinity});
  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.row_dual[0], 0.0, 1e-12) << "a redundant row cannot carry a price";
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, EmptyRowExcludingZeroIsInfeasible) {
  // A row with no entries has activity exactly zero. If its bounds exclude zero the model is
  // infeasible on that evidence alone, and saying so is both faster and more honest than
  // handing the simplex a model whose answer is already known.
  Model model = make_lp({{0.0}}, {2.0}, {kInfinity}, {1.0}, {0.0}, {kInfinity});
  const Solution on = solve(model, with_presolve(true));
  EXPECT_EQ(on.status, SolveStatus::kInfeasible) << on.message;
}

TEST(Presolve, AnUnboundedObjectiveIsNotReportedAsInfeasible) {
  // PRESOLVE CANNOT CONCLUDE UNBOUNDEDNESS. An empty column whose cost drives it to an
  // infinite bound makes the objective unbounded only if the feasible region is non-empty,
  // and presolve has established no such thing. Both directions are tested here because the
  // first version got each of them wrong in turn: it reported unbounded models as infeasible,
  // and then infeasible models as unbounded.
  //   min -x0, x0 free above, appearing in no row.
  const Model unbounded_model = make_lp({{0.0, 1.0}}, {1.0}, {kInfinity}, {-1.0, 1.0},
                                        {0.0, 0.0}, {kInfinity, kInfinity});
  const Solution unbounded = solve(unbounded_model, with_presolve(true));
  EXPECT_EQ(unbounded.status, SolveStatus::kUnbounded) << unbounded.message;

  // The same empty column of negative cost, but with a row nothing can satisfy. Infeasible
  // beats unbounded: there is no point to be unbounded over.
  const Model infeasible_model =
      make_lp({{0.0, -1.0}}, {2.0}, {kInfinity}, {-1.0, 1.0}, {0.0, 0.0}, {kInfinity, 0.0});
  const Solution infeasible = solve(infeasible_model, with_presolve(true));
  EXPECT_EQ(infeasible.status, SolveStatus::kInfeasible) << infeasible.message;
}

TEST(Presolve, IntegerBoundsAreRoundedInwardNeverOutward) {
  // `2*x0 <= 7` implies x0 <= 3.5, and for an integer column that is x0 <= 3. Rounding the
  // other way would leave 3.5 in the box, which is harmless; rounding a LOWER bound outward
  // would not be, and neither would rounding an upper bound up to 4. A bound that excludes a
  // feasible integer removes the optimum with no symptom at all.
  Model model = make_lp({{2.0}}, {-kInfinity}, {7.0}, {-1.0}, {0.0}, {10.0});
  model.col_type = {VarType::kInteger};
  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.col_value[0], 3.0, 1e-9);
  EXPECT_NEAR(on.objective, -3.0, 1e-9);
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, ChainedReductionsStillRoundTrip) {
  // Fixing a column empties a row, removing that row makes another column a singleton, and
  // so on. The passes run to a fixed point, so this exercises the ORDER postsolve has to
  // undo - which is the part a single-reduction test cannot reach.
  const Model model = make_lp({{1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}},
                              {4.0, 2.0, 3.0}, {kInfinity, 2.0, kInfinity}, {1.0, 3.0, 2.0},
                              {0.0, 0.0, 0.0}, {kInfinity, kInfinity, kInfinity});
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, LeavesAModelWithNothingToRemoveAlone) {
  // Nothing here is empty, fixed, singleton or redundant, so presolve must be a no-op. A
  // reduction that fires when it should not is how a correct model becomes a wrong answer.
  const Model model = make_lp({{1.0, 2.0}, {3.0, 1.0}}, {4.0, 5.0}, {kInfinity, kInfinity},
                              {1.0, 1.0}, {0.0, 0.0}, {kInfinity, kInfinity});
  expect_agrees_with_unpresolved(model);
}

}  // namespace
}  // namespace sankhya

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
#include <iostream>
#include <random>
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

TEST(Presolve, FreeColumnSingletonIsSubstitutedAndRemoved) {
  // x1 is free and appears in exactly one row. It constrains nothing else, so the row and
  // the column both go, and x1's cost is folded into x0, the only other column sharing the
  // row.
  //   min 5*x0 + 2*x1  s.t.  x0 + x1 >= 2,  x0 in [0,10], x1 free
  // x1's cost pushes the row to its (only, finite) bound: x1 = 2 - x0, and since x1's own
  // cost coefficient is positive, minimising 2*x1 pushes x0 to its own lower bound, 0 - so
  // x0 = 0, x1 = 2, objective = 0 + 4 = 4.
  const Model model = make_lp({{1.0, 1.0}}, {2.0}, {kInfinity}, {5.0, 2.0}, {0.0, -kInfinity},
                              {10.0, kInfinity});
  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 4.0, 1e-9);
  EXPECT_NEAR(on.col_value[0], 0.0, 1e-9);
  EXPECT_NEAR(on.col_value[1], 2.0, 1e-9);
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, DoubletonEquationEliminatesAColumn) {
  // An equality with exactly two entries lets x0 be written exactly as 2*x1 and eliminated;
  // x1's bounds are then implied by x0's own [0,10] via that relationship.
  //   min x0 + 2*x1  s.t.  x0 - 2*x1 = 0,  x0 in [0,10], x1 in [-5,5]
  // x0 = 2*x1 pins x1 to [0,5] (x0's range divided by 2), the folded cost on x1 becomes
  // 2 - 1*(-2/1) = 4, and with no rows left the reduced problem just parks x1 at its new
  // lower bound, 0. x0 = 0, objective = 0.
  const Model model =
      make_lp({{1.0, -2.0}}, {0.0}, {0.0}, {1.0, 2.0}, {0.0, -5.0}, {10.0, 5.0});
  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 0.0, 1e-9);
  EXPECT_NEAR(on.col_value[0], 0.0, 1e-9);
  EXPECT_NEAR(on.col_value[1], 0.0, 1e-9);
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, DoubletonEquationCascadesIntoASingletonRow) {
  // The doubleton eliminates x0, and x0 also appears in the OTHER row - fill-in adjusts
  // THAT row's coefficient on x1 (from 1 to 3, here), turning it into a singleton row on a
  // column the doubleton's own fold already changed the cost of. This is the exact
  // interaction a fuzz run first found wrong: postsolve priced the singleton row using x1's
  // pre-fold cost and the pre-fill-in coefficient, giving a dual infeasibility of 2 on a
  // problem this small.
  //   min x0 + 2*x1  s.t.  1 <= x0 + x1 <= 4,  x0 - 2*x1 = 0,  x0 in [0,10], x1 free
  // x0 = 2*x1 makes the first row 3*x1 in [1,4], i.e. x1 in [1/3, 4/3]; folded cost on x1 is
  // 4, so the reduced problem parks x1 at 1/3. x0 = 2/3, objective = 2/3 + 2/3 = 4/3.
  const Model model = make_lp({{1.0, 1.0}, {1.0, -2.0}}, {1.0, 0.0}, {4.0, 0.0}, {1.0, 2.0},
                              {0.0, -kInfinity}, {10.0, kInfinity});
  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 4.0 / 3.0, 1e-9);
  EXPECT_NEAR(on.col_value[0], 2.0 / 3.0, 1e-9);
  EXPECT_NEAR(on.col_value[1], 1.0 / 3.0, 1e-9);
  EXPECT_LE(on.dual_infeasibility, tol::kDualFeasibility) << on.message;
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, FixedColumnSharingAFoldedDoubletonRowStillPricesCorrectly) {
  // x0 is fixed OUTRIGHT by its own bounds, not by any row - fold_fixed_column() removes it
  // from the equality row below before the doubleton reduction ever looks at that row, so
  // the doubleton only ever sees x1 and x2 there and has no idea x0 used to share it.
  //   min x0 + 2*x1 + 3*x2  s.t.  2*x0 + 3*x1 + 4*x2 = 20,  x0 in [5,5],  x1,x2 in [0,10]
  // x0 = 5 folds the row to 3*x1 + 4*x2 = 10, a genuine doubleton: x1 = (10 - 4*x2)/3,
  // folding x1's cost (2) into x2's gives 3 - 2*(4/3) = 1/3, and x1 in [0,10] implies
  // x2 in [0, 2.5]. Minimising a positive cost on x2 parks it at 0, so x1 = 10/3.
  // objective = 5 + 20/3 + 0 = 35/3.
  //
  // x0's ORIGINAL coefficient in that row (2) is real and must be priced against the row's
  // TRUE dual once the doubleton derives it - postsolve found x0 dead in that row already
  // (fixed columns are folded out before doubletons are even detected) and, before this was
  // fixed, treated the row as fully "explained" for every column touching it, silently
  // dropping x0's own term and reporting a self-inconsistent reduced cost. Found on Netlib's
  // `bandm` (column ORROLC, fixed by an unrelated earlier equality, sharing a row with a
  // later doubleton) - this is the same structure at unit-test scale.
  const Model model =
      make_lp({{2.0, 3.0, 4.0}}, {20.0}, {20.0}, {1.0, 2.0, 3.0}, {5.0, 0.0, 0.0}, {5.0, 10.0, 10.0});
  const Solution on = solve(model, with_presolve(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, 35.0 / 3.0, 1e-9);
  EXPECT_NEAR(on.col_value[0], 5.0, 1e-9);
  EXPECT_NEAR(on.col_value[1], 10.0 / 3.0, 1e-9);
  EXPECT_NEAR(on.col_value[2], 0.0, 1e-9);
  EXPECT_LE(on.dual_infeasibility, tol::kDualFeasibility) << on.message;
  expect_agrees_with_unpresolved(model);
}

TEST(Presolve, LeavesAModelWithNothingToRemoveAlone) {
  // Nothing here is empty, fixed, singleton or redundant, so presolve must be a no-op. A
  // reduction that fires when it should not is how a correct model becomes a wrong answer.
  const Model model = make_lp({{1.0, 2.0}, {3.0, 1.0}}, {4.0, 5.0}, {kInfinity, kInfinity},
                              {1.0, 1.0}, {0.0, 0.0}, {kInfinity, kInfinity});
  expect_agrees_with_unpresolved(model);
}

// =========================================================================================
// Fuzz: free-column-singleton and doubleton-equation, deliberately at scale
//
// The hand-written cases above pin specific, once-found bugs down for good, but the random
// generic fuzz gate in tests/oracles/ rarely happens to produce an EXACT equality row with
// EXACTLY two nonzero entries, or a genuinely free column appearing in exactly one row -
// these two reductions need that specific structure to fire at all. Three real postsolve
// bugs were found writing this file BY HAND (a cost fold that ignored an earlier fold's
// cascade, a fill-in coefficient reduced_cost_of did not know about, and a doubleton's
// eliminated column independently forcing the row's dual even though its partner did not
// need the help) - all three were interaction bugs between reductions, exactly the kind
// generic random generation is least likely to construct on its own. So this generator
// builds the structure deliberately, at scale, rather than hoping for it.
// =========================================================================================

namespace {

/// A model built around a known feasible point x0, with SOME rows forced to be exactly the
/// two-entry equalities doubleton-equation looks for, and some columns forced free so they
/// are eligible for free-column-singleton whenever they end up alone in their row.
Model random_doubleton_prone_lp(std::mt19937& rng, int trial) {
  std::uniform_real_distribution<double> coefficient(-4.0, 4.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  const auto n = static_cast<Index>(3 + trial % 5);  // 3..7 columns
  const auto m = static_cast<Index>(2 + trial % 4);  // 2..5 rows
  const auto un = static_cast<std::size_t>(n);
  const auto um = static_cast<std::size_t>(m);

  std::vector<double> cost(un);
  for (double& c : cost) c = coefficient(rng);

  std::vector<double> col_lower(un);
  std::vector<double> col_upper(un);
  std::vector<double> x0(un);
  for (std::size_t j = 0; j < un; ++j) {
    if (unit(rng) < 0.35) {
      // Free - a candidate for free-column-singleton if it ends up alone in its row.
      col_lower[j] = -kInfinity;
      col_upper[j] = kInfinity;
      x0[j] = coefficient(rng);
    } else {
      const double centre = coefficient(rng);
      const double half_width = 1.0 + 4.0 * unit(rng);
      col_lower[j] = centre - half_width;
      col_upper[j] = centre + half_width;
      x0[j] = col_lower[j] + unit(rng) * (col_upper[j] - col_lower[j]);
    }
  }

  std::vector<std::vector<double>> rows(um, std::vector<double>(un, 0.0));
  std::vector<double> row_lower(um);
  std::vector<double> row_upper(um);
  for (std::size_t i = 0; i < um; ++i) {
    if (unit(rng) < 0.5 && n >= 2) {
      // Force a genuine doubleton: exactly two nonzero entries, equality, tight at x0.
      const auto j1 = static_cast<std::size_t>(unit(rng) * static_cast<double>(n)) % un;
      std::size_t j2 = static_cast<std::size_t>(unit(rng) * static_cast<double>(n)) % un;
      if (j2 == j1) j2 = (j1 + 1) % un;
      double a1 = coefficient(rng);
      double a2 = coefficient(rng);
      if (std::fabs(a1) < 0.5) a1 = a1 < 0.0 ? -0.5 : 0.5;
      if (std::fabs(a2) < 0.5) a2 = a2 < 0.0 ? -0.5 : 0.5;
      rows[i][j1] = a1;
      rows[i][j2] = a2;
      const double activity = a1 * x0[j1] + a2 * x0[j2];
      row_lower[i] = activity;
      row_upper[i] = activity;
    } else {
      // A general row, exactly like the KKT-certificate fuzz generator elsewhere: most
      // columns included with some probability, bounds placed around x0's activity.
      bool any = false;
      for (std::size_t j = 0; j < un; ++j) {
        if (unit(rng) < 0.6) {
          rows[i][j] = coefficient(rng);
          any = true;
        }
      }
      if (!any) rows[i][0] = 1.0;
      double activity = 0.0;
      for (std::size_t j = 0; j < un; ++j) activity += rows[i][j] * x0[j];
      row_lower[i] = activity - (0.5 + 3.0 * unit(rng));
      row_upper[i] = activity + (0.5 + 3.0 * unit(rng));
    }
  }

  return make_lp(rows, row_lower, row_upper, cost, col_lower, col_upper);
}

}  // namespace

TEST(Presolve, FuzzFreeColumnSingletonAndDoubletonEquation) {
  std::mt19937 rng(923004);
  int compared = 0;
  int skipped_invalid = 0;

  for (int trial = 0; trial < 400; ++trial) {
    const Model model = random_doubleton_prone_lp(rng, trial);
    if (!model.validate().empty()) {
      ++skipped_invalid;
      continue;
    }
    const Solution off = solve(model, with_presolve(false));
    const Solution on = solve(model, with_presolve(true));
    ASSERT_EQ(on.status, off.status)
        << "trial " << trial << " - on: " << on.message << " / off: " << off.message;
    if (off.status != SolveStatus::kOptimal) continue;
    ++compared;

    EXPECT_NEAR(on.objective, off.objective, 1e-6 * std::max(1.0, std::fabs(off.objective)))
        << "trial " << trial;
    EXPECT_LE(on.primal_infeasibility, tol::kPrimalFeasibility) << "trial " << trial;
    EXPECT_LE(on.dual_infeasibility, tol::kDualFeasibility)
        << "trial " << trial << ": " << on.message;
  }

  // Printed so a generator that drifts towards trivial instances is visible rather than
  // quietly making the whole run mean less than its trial count suggests.
  std::cout << "doubleton/free-singleton fuzz: " << compared << " compared, " << skipped_invalid
            << " invalid, out of 400\n";
  EXPECT_GT(compared, 250) << "too few instances reached optimal under both paths for this "
                              "comparison to mean anything";
}

}  // namespace
}  // namespace sankhya

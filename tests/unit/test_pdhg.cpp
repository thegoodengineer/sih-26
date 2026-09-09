// SPDX-License-Identifier: Apache-2.0
// SANKHYA - restarted PDHG tests.
//
// The controlling test in this file is AgreesWithTheSimplexOnGeneratedInstances. Two engines
// with nothing in common beyond the Model - one pivoting on exact ratios, one taking
// projected gradient steps - landing on the same objective is much stronger evidence than
// either matching a hand-written expectation. It is also the only cheap way to test PDHG at
// all: a first-order method has no basis to inspect and no pivot sequence to reason about.
//
// Note what is NOT asserted here: that PDHG is fast. On instances this small it is
// enormously slower than the simplex, by design and by construction. Its value is at a scale
// where a dense factorization cannot go, and on hardware this suite does not run on.

#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "core/status_guard.hpp"

#include "support/temp_file.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

Options pdhg_options(double tolerance) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "pdhg");
  options.set_double("pdhg_tolerance", tolerance);
  options.set_int("iteration_limit", 200000);
  return options;
}

Options simplex_options() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "simplex");
  return options;
}

/// A tiny LP built directly, so the expected optimum is arithmetic in the comment.
Model make_lp(const std::vector<std::vector<double>>& rows, const std::vector<double>& lower,
              const std::vector<double>& upper, const std::vector<double>& cost,
              const std::vector<double>& col_upper = {}) {
  Model model;
  const auto cols = static_cast<Index>(cost.size());
  const auto num_rows = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower.assign(static_cast<std::size_t>(cols), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(cols), kInfinity);
  if (!col_upper.empty()) model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(cols), VarType::kContinuous);
  model.row_lower = lower;
  model.row_upper = upper;
  model.matrix.reset(num_rows, cols);
  for (Index i = 0; i < num_rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  return model;
}

// =========================================================================================

TEST(Pdhg, MinimisesASum) {
  // min x + y  s.t.  x + y >= 2,  x, y >= 0.  Optimum 2.
  const Model model = make_lp({{1.0, 1.0}}, {2.0}, {kInfinity}, {1.0, 1.0});
  const Solution s = solve(model, pdhg_options(1e-8));
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, 2.0, 1e-6);
}

TEST(Pdhg, HonoursAnEqualityRow) {
  // min x + 2y  s.t.  x + y = 4,  x, y >= 0.  Optimum 4 at (4, 0).
  const Model model = make_lp({{1.0, 1.0}}, {4.0}, {4.0}, {1.0, 2.0});
  const Solution s = solve(model, pdhg_options(1e-8));
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, 4.0, 1e-6);
}

TEST(Pdhg, HonoursARangeRow) {
  // 2 <= x <= 6 written as a range row; minimising x gives 2. The Moreau projection has to
  // handle a two-sided row without the caller splitting it into two inequalities.
  const Model model = make_lp({{1.0}}, {2.0}, {6.0}, {1.0});
  const Solution s = solve(model, pdhg_options(1e-8));
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, 2.0, 1e-6);
}

TEST(Pdhg, HonoursAColumnUpperBound) {
  // max x  (as min -x)  with x <= 7 by bound and a slack row. Optimum -7.
  const Model model = make_lp({{1.0}}, {-kInfinity}, {100.0}, {-1.0}, {7.0});
  const Solution s = solve(model, pdhg_options(1e-8));
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, -7.0, 1e-6);
}

TEST(Pdhg, ReportsTheObjectiveInTheOriginalSense) {
  // max 3x + 5y  s.t. x <= 4, 2y <= 12, 3x + 2y <= 18.  Optimum 36 at (2, 6).
  Model model = make_lp({{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}},
                        {-kInfinity, -kInfinity, -kInfinity}, {4.0, 12.0, 18.0}, {3.0, 5.0});
  model.sense = ObjSense::kMaximize;
  const Solution s = solve(model, pdhg_options(1e-8));
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, 36.0, 1e-5);
  EXPECT_NEAR(s.col_value[0], 2.0, 1e-4);
  EXPECT_NEAR(s.col_value[1], 6.0, 1e-4);
}

TEST(Pdhg, CarriesTheObjectiveOffset) {
  Model model = make_lp({{1.0}}, {3.0}, {kInfinity}, {1.0});
  model.objective_offset = 10.0;
  const Solution s = solve(model, pdhg_options(1e-8));
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, 13.0, 1e-6);
}

TEST(Pdhg, StopsAtTheIterationLimitWithoutClaimingOptimality) {
  Model model = make_lp({{1.0, 1.0}}, {2.0}, {kInfinity}, {1.0, 1.0});
  Options options = pdhg_options(1e-12);
  options.set_int("iteration_limit", 20);
  const Solution s = solve(model, options);
  EXPECT_NE(s.status, SolveStatus::kOptimal);
  // An unfinished first-order run must not present its incumbent as a proven bound.
  EXPECT_TRUE(std::isinf(s.dual_bound));
  EXPECT_NE(s.message, "");
}

TEST(Pdhg, IsSelectedOnlyWhenAskedFor) {
  const Model model = make_lp({{1.0, 1.0}}, {2.0}, {kInfinity}, {1.0, 1.0});
  const Solution automatic = solve(model, simplex_options());
  EXPECT_EQ(automatic.algorithm, "simplex-primal");
  const Solution requested = solve(model, pdhg_options(1e-8));
  EXPECT_EQ(requested.algorithm, "pdhg-cpu");
}

TEST(Pdhg, GpuFlagFallsBackToCpuWithoutCrashing) {
  // CLAUDE.md: the CPU build must work with zero CUDA installed, and --gpu must degrade
  // silently rather than fail.
  const Model model = make_lp({{1.0, 1.0}}, {2.0}, {kInfinity}, {1.0, 1.0});
  Options options = pdhg_options(1e-8);
  options.set_bool("gpu", true);
  const Solution s = solve(model, options);
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, 2.0, 1e-6);
}

TEST(Pdhg, AgreesWithTheSimplexOnGeneratedInstances) {
  // Two engines sharing nothing but the Model. Instances come from the Phase 3 generator so
  // that neither engine's author chose them.
  std::mt19937_64 rng(20260904);
  oracle::GeneratorConfig config;
  config.max_rows = 6;
  config.max_cols = 6;

  int compared = 0;
  int disagreed = 0;
  double worst = 0.0;
  std::vector<std::string> failures;

  for (int trial = 0; trial < 120; ++trial) {
    const oracle::KktInstance instance = oracle::kkt_lp(rng, config);
    const Model model = oracle::to_model(instance.lp);

    const Solution simplex = solve(model, simplex_options());
    if (simplex.status != SolveStatus::kOptimal) continue;

    const Solution first_order = solve(model, pdhg_options(1e-9));
    if (first_order.status != SolveStatus::kOptimal) continue;  // counted below, not here

    ++compared;
    const double scale = std::max(1.0, std::fabs(simplex.objective));
    const double relative = std::fabs(first_order.objective - simplex.objective) / scale;
    worst = std::max(worst, relative);
    if (relative > 1e-6) {
      ++disagreed;
      if (failures.size() < 3) {
        failures.push_back("simplex " + std::to_string(simplex.objective) + " vs pdhg " +
                           std::to_string(first_order.objective) + "\n" +
                           instance.lp.to_text());
      }
    }
  }

  std::cout << "PDHG vs simplex: " << compared << " compared, worst relative difference "
            << worst << "\n";
  for (const std::string& failure : failures) {
    std::cout << "\n--- disagreement ---\n" << failure << "\n";
  }
  EXPECT_EQ(disagreed, 0);
  EXPECT_GT(compared, 60) << "too few instances converged for this to mean anything";
}

TEST(Pdhg, OptimalIsNeverClaimedOnAPointThatWouldFailVerification) {
  // The bug this pins: PDHG converges on RELATIVE residuals, dividing by (1 + ||bounds||).
  // On a model whose right-hand sides are large, a relative 1e-8 leaves an absolute
  // violation orders of magnitude bigger - and the engine used to stamp "optimal" on it,
  // which tools/verify_solution.py then rejected. kOptimal now means, and must keep meaning,
  // "this point would survive verification".
  //
  // Right-hand side 1e5 makes the two measures diverge by five orders of magnitude.
  const Model model = make_lp({{1.0, 1.0}}, {1.0e5}, {kInfinity}, {1.0, 1.0});

  for (const double requested : {1e-4, 1e-6, 1e-8}) {
    const Solution s = solve(model, pdhg_options(requested));
    if (s.status != SolveStatus::kOptimal) continue;
    EXPECT_LE(s.primal_infeasibility, tol::kPrimalFeasibility)
        << "claimed optimal at requested tolerance " << requested
        << " with absolute primal infeasibility " << s.primal_infeasibility;
    EXPECT_LE(s.integrality_violation, tol::kIntegrality);
  }
}

// =========================================================================================
// The two changes extracted from the CUDA branch, each pinned by the behaviour it fixes
// =========================================================================================

TEST(Pdhg, TheStepSizeSurvivesItsOwnFirstIteration) {
  // The adaptive rule sets shrink = 1 - pow(iteration + 1, -0.3), so at iteration 0 the
  // exponent is 1 and shrink is exactly zero: the proposal min(0 * limit, grow * eta) is
  // zero, and eta is clamped from 1/||A||_2 down to its 1e-12 floor on the first admissible
  // step. Recovery is capped at the grow factor, so the solve spends tens of iterations
  // climbing back to the step size it started with.
  //
  // A ceiling on iterations is the way to test that without asserting a particular step size:
  // the collapse costs iterations and nothing else observable. 500 is far above what this
  // two-variable model needs after the fix and far below the several thousand the collapse
  // costs; it is a guard against the bug returning, not a performance target.
  const Model model = make_lp({{1.0, 1.0}}, {2.0}, {kInfinity}, {1.0, 1.0});
  const Solution s = solve(model, pdhg_options(1e-8));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_LT(s.iterations, 500) << "the first-iteration step-size collapse is back";
}

TEST(Pdhg, StopsOnlyOnAPointThatMeetsTheProjectStandard) {
  // The loop's stopping test and the report's verdict used to be different tests: the loop
  // stopped when the RELATIVE residuals met the requested tolerance, and the report then
  // downgraded that point to `feasible` when its ABSOLUTE residuals did not meet the
  // project's standard. So the solver stopped early and handed back the weaker answer while
  // it was still converging.
  //
  // The model is the one OptimalIsNeverClaimedOnAPointThatWouldFailVerification uses, whose
  // right-hand side of 1e5 makes the two measures diverge by five orders of magnitude - it is
  // exactly the case that used to stop at `feasible`.
  const Model model = make_lp({{1.0, 1.0}}, {1.0e5}, {kInfinity}, {1.0, 1.0});
  const Solution s = solve(model, pdhg_options(1e-4));
  EXPECT_NE(s.status, SolveStatus::kFeasible)
      << "the loop stopped on a point it then had to downgrade: " << s.message;
  if (s.status == SolveStatus::kOptimal) {
    EXPECT_LE(s.primal_infeasibility, tol::kPrimalFeasibility);
  }
}

TEST(Pdhg, AFeasibleStatusStillMeansTheePointIsActuallyFeasible) {
  // kFeasible is a weaker claim than kOptimal but it is still a claim: sankhya::Solution
  // documents it as "a feasible point exists and is reported". Stopping on a relative
  // residual alone would let this engine assert feasibility for a point that misses the
  // project's own primal tolerance.
  const Model model = make_lp({{1.0, 1.0}}, {1.0e5}, {kInfinity}, {1.0, 1.0});
  const Solution s = solve(model, pdhg_options(1e-6));
  if (s.status == SolveStatus::kOptimal || s.status == SolveStatus::kFeasible) {
    EXPECT_LE(s.primal_infeasibility, tol::kPrimalFeasibility);
  }
}

// =========================================================================================
// The status must agree with the measured quality of the point
//
// These pin down a defect that reached main and that every existing test was blind to.
// PDHG terminates on a RELATIVE KKT criterion at pdhg_tolerance, and that was being
// translated straight into kOptimal. A relative KKT residual of 1e-4 is not the same claim
// as "primal feasible to 1e-7": on all eight fetched Netlib instances the engine reported
// `optimal` and tools/verify_solution.py rejected every one, with row violations up to
// 4.5e-2. sc50b, published optimum exactly -70, came back as -70.0139 and was labelled
// optimal - a value BETTER than the optimum, reachable only from outside the feasible set.
//
// Solution::recompute_quality() had measured and printed the violation the whole time.
// Nothing connected that measurement to the status. solve() now reconciles the two.
// =========================================================================================

/// The blending model from demo/crude_blend.mps, built in memory. It is used here because
/// its conditioning makes PDHG stop well short of the solver's feasibility tolerance at a
/// loose setting, which is exactly the situation being pinned down.
Model make_blend_lp() {
  Model model;
  model.name = "BLENDGUARD";
  model.sense = ObjSense::kMaximize;
  model.col_cost = {2.40, 1.60, 1.64};
  model.col_lower = {10.0, 0.0, 0.0};
  model.col_upper = {kInfinity, 45.0, 60.0};
  model.col_type.assign(3, VarType::kContinuous);
  model.row_lower = {90.0, 40.0, -kInfinity};
  model.row_upper = {120.0, 40.0, 0.0};

  model.matrix.reset(3, 3);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(0, 2, 1.0);
  model.matrix.add_entry(1, 0, 0.30);
  model.matrix.add_entry(1, 1, 0.45);
  model.matrix.add_entry(1, 2, 0.38);
  model.matrix.add_entry(2, 0, 0.80);
  model.matrix.add_entry(2, 1, -0.86);
  model.matrix.add_entry(2, 2, -0.22);
  model.matrix.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

TEST(Pdhg, StopAtRequestGivesTheCheapAnswerAndNeverCallsItOptimal) {
  // #180: the default runs on to the project's absolute standard whatever the request, so a
  // caller who wants the cheap approximate answer a first-order method exists to give has to
  // say so. No hand-built model in this file separates the two measures - on every one the
  // relative request and the absolute standard flip on the same 40-iteration check - so this
  // uses a committed Netlib instance, the way test_ipm.cpp does: adlittle at a 1e-4 request
  // meets it at 132,520 iterations and the standard at 192,080.
  //
  // Three things are pinned. The switch stops the loop earlier than the default on the same
  // model. The point it stops on is reported `feasible`, never `optimal`, unless it happens
  // to meet the standard anyway - the switch cannot manufacture a claim. And the honest run
  // on the same instance does reach `optimal`, so the two are measuring the same thing.
  Model model;
  const io::ReadResult read = io::read_model("data/netlib/adlittle.mps", &model);
  if (!read.ok) {
    std::cout << "pdhg #180: data/netlib/adlittle.mps not readable here (" << read.error
              << "); skipped\n";
    return;
  }
  const Solution honest = solve(model, pdhg_options(1e-4));
  Options cheap_options = pdhg_options(1e-4);
  cheap_options.set_bool("pdhg_stop_at_request", true);
  const Solution cheap = solve(model, cheap_options);

  ASSERT_EQ(honest.status, SolveStatus::kOptimal) << honest.message;
  ASSERT_TRUE(cheap.status == SolveStatus::kFeasible || cheap.status == SolveStatus::kOptimal)
      << cheap.message;
  EXPECT_LT(cheap.iterations, honest.iterations)
      << "stopping on the request should cost fewer iterations than running to the standard";
  if (cheap.status == SolveStatus::kOptimal) {
    EXPECT_LE(cheap.primal_infeasibility, tol::kPrimalFeasibility) << cheap.message;
    EXPECT_LE(cheap.dual_infeasibility, tol::kDualFeasibility) << cheap.message;
  } else {
    EXPECT_TRUE(cheap.has_primal_values());
  }
}

TEST(SolveStatusGuard, AnInfeasiblePointIsNeverReportedAsOptimal) {
  // Exercised DIRECTLY rather than through an engine. The original version drove PDHG at a
  // loose tolerance until it returned an infeasible point labelled optimal, and carried a
  // guard message saying to re-tune it if that stopped happening. It has stopped happening:
  // PDHG now refuses to claim optimal or feasible unless the point meets the absolute
  // tolerance, so the trigger path through that engine no longer exists.
  //
  // Which is the point. A guard tested only through a misbehaving engine loses its subject
  // the moment that engine is fixed, and quietly stops testing anything. This calls the
  // reconciliation with a Solution built to contradict itself, so it keeps its subject
  // whatever the engines do - and it is the shape the Phase 8 IPM and QP engines will hit.
  const Model model = make_blend_lp();

  Solution solution;
  solution.allocate_for(model);
  solution.status = SolveStatus::kOptimal;
  solution.algorithm = "fabricated";
  // A point well outside the feasible region, claimed as a proven optimum.
  solution.col_value.assign(static_cast<std::size_t>(model.num_cols()), 1.0e4);
  solution.recompute_quality(model);

  Options options;
  options.set_bool("log_to_console", false);
  ASSERT_GT(solution.primal_infeasibility, options.get_double("primal_feasibility_tolerance"))
      << "the fabricated point is supposed to be infeasible";

  Logger silent(nullptr);
  reconcile_status_with_measurement(&solution, options, silent, /*check_dual=*/true);

  EXPECT_NE(solution.status, SolveStatus::kOptimal);
  EXPECT_NE(solution.status, SolveStatus::kFeasible)
      << "a point that violates its own constraints is not feasible either";
  EXPECT_EQ(solution.status, SolveStatus::kNumericalError);
  EXPECT_NE(solution.message.find("primal feasibility"), std::string::npos) << solution.message;
}

TEST(SolveStatusGuard, AFeasibleOptimalPointIsLeftAlone) {
  // The guard must not fire on a good answer. Without this, tightening it later could start
  // rejecting correct solutions and every other test would still pass.
  const Model model = make_blend_lp();
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "simplex");

  Solution solution = solve(model, options);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal);
  const double objective = solution.objective;

  Logger silent(nullptr);
  reconcile_status_with_measurement(&solution, options, silent, /*check_dual=*/true);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal);
  EXPECT_DOUBLE_EQ(solution.objective, objective);
}

TEST(SolveStatusGuard, PrimalFeasibleButDualInfeasibleIsFeasibleNotOptimal) {
  // The other half of the rule. The point satisfies every constraint, so it is usable and
  // kFeasible is honest - but the reduced costs do not support a claim of optimality, and
  // kOptimal is a claim of proof.
  // THIS TEST USED TO MANUFACTURE ITS POINT BY ASKING PDHG FOR A LOOSE TOLERANCE (0.1) and
  // taking what the loop stopped on. That route is gone: the loop now stops only on a point
  // that also meets the project's absolute standard, so a converged PDHG solve no longer
  // hands back a primal-feasible point with unusable duals. The rule under test is the status
  // guard's, not PDHG's, so the point is now built directly - solve exactly, then invalidate
  // the reduced costs - which tests the same rule without depending on an engine stopping
  // somewhere it no longer stops.
  const Model model = make_blend_lp();
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "simplex");

  Solution solution = solve(model, options);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal);
  ASSERT_LE(solution.primal_infeasibility, options.get_double("primal_feasibility_tolerance"));

  solution.col_dual.assign(static_cast<std::size_t>(model.num_cols()), 1.0);
  solution.recompute_quality(model);

  ASSERT_LE(solution.primal_infeasibility, options.get_double("primal_feasibility_tolerance"))
      << "expected a primal-feasible point";
  ASSERT_GT(solution.dual_infeasibility, options.get_double("dual_feasibility_tolerance"))
      << "expected the duals to be short of tolerance after invalidating the multipliers";

  Logger silent(nullptr);
  reconcile_status_with_measurement(&solution, options, silent, /*check_dual=*/true);

  EXPECT_EQ(solution.status, SolveStatus::kFeasible);
  EXPECT_TRUE(solution.has_primal_values());
  EXPECT_NE(solution.message.find("dual feasibility"), std::string::npos) << solution.message;
}

TEST(SolveStatusGuard, AConvergedPdhgSolveStillReportsOptimal) {
  // The guard must not simply forbid PDHG from ever succeeding. Given a tolerance it can
  // actually meet, the optimality claim stands.
  const Model model = make_blend_lp();
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "pdhg");
  options.set_double("pdhg_tolerance", 1e-12);

  const Solution solution = solve(model, options);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_LE(solution.primal_infeasibility, options.get_double("primal_feasibility_tolerance"));
  EXPECT_LE(solution.dual_infeasibility, options.get_double("dual_feasibility_tolerance"));
  EXPECT_NEAR(solution.objective, 214.14594594594595, 1e-4);
}

TEST(SolveStatusGuard, TheSimplexIsUnaffected) {
  // The simplex terminates at a vertex with an exact basis, so it must pass the guard
  // untouched. A false positive here would be as damaging as the bug being fixed.
  const Model model = make_blend_lp();
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "simplex");

  const Solution solution = solve(model, options);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  // Within tolerance, NOT exactly zero. The guard's contract is that a measured violation
  // stays under the documented tolerance; the simplex never promised a bit-exact zero and
  // does not deliver one. Recomputing row activities from a model whose coefficients span
  // 0.22 to 20 leaves accumulation at machine-epsilon scale - this assertion first read
  // EXPECT_DOUBLE_EQ(..., 0.0), which passed on Windows/Release and failed on Linux/Debug
  // at 3.55e-15. That is twelve orders of magnitude inside the tolerance being tested, so
  // the assertion was wrong rather than the code.
  EXPECT_LE(solution.primal_infeasibility, options.get_double("primal_feasibility_tolerance"));
  EXPECT_LE(solution.dual_infeasibility, options.get_double("dual_feasibility_tolerance"));
  EXPECT_NEAR(solution.objective, 214.14594594594595, 1e-9);
}

TEST(SolveStatusGuard, TheMilpPathIsReconciledWithoutFalsePositives) {
  // Branch and bound goes through the same gate, but with the dual test DISABLED. An
  // incumbent comes from a node LP whose bounds were tightened by branching, so its reduced
  // costs are dual feasible for that node and generally are not for the original model.
  // Optimality of a MILP is proved by the bound closing against the incumbent, not by the
  // last LP's reduced costs, so applying the dual test here would reject correct answers.
  //
  // Integrality replaces it: an engine reporting optimal while holding a fractional integer
  // column has reported the relaxation, which is the failure CLAUDE.md singles out.
  Model model = make_blend_lp();
  model.col_type[0] = VarType::kInteger;

  Options options;
  options.set_bool("log_to_console", false);
  const Solution solution = solve(model, options);

  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_LE(solution.integrality_violation, options.get_double("integrality_tolerance"));
  EXPECT_LE(solution.primal_infeasibility, options.get_double("primal_feasibility_tolerance"));
  // The integer column really is integral, so the claim is not vacuous.
  EXPECT_NEAR(solution.col_value[0], std::round(solution.col_value[0]), 1e-6);
}

TEST(SolveStatusGuard, ANonClaimingStatusIsLeftAlone) {
  // kIterationLimit makes no assertion about optimality, so the guard has no business
  // rewriting it - the caller needs to know the run was cut short, not that it was
  // numerically unsound.
  const Model model = make_blend_lp();
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "pdhg");
  options.set_int("iteration_limit", 5);

  const Solution solution = solve(model, options);
  EXPECT_EQ(solution.status, SolveStatus::kIterationLimit) << solution.message;
}

// issue #122: the live progress stream reported the objective of whatever model the ENGINE
// was handed, and presolve hands it a reduced model whose objective_offset absorbs every
// column presolve eliminated. The final Solution added that offset back; the stream did not.
// A solve watched through `tail -f` therefore converged to a number the reported result
// never reached - off by exactly the offset, 365 on the demo's 5000x5000 instance, where
// the two appeared one under the other in section 2.5.
//
// PrimalSimplex::minimization_objective() had this right all along and says why in a
// comment: the iteration table has to report the same quantity as the final line and the
// .sol file. This pins that invariant for PDHG too, which is where it was broken.
TEST(Pdhg, ProgressStreamReportsTheSameObjectiveAsTheResultUnderPresolve) {
  // x is FIXED at 5 with a cost of 3, so presolve eliminates it and folds 15 into the
  // reduced model's objective_offset (presolve.cpp:397). Without that offset the stream
  // would converge to 5 while the result reported 20 - a difference no reader could
  // attribute to anything, since both numbers are individually plausible.
  //
  // Three more columns and two more rows than the offset alone needs, so that presolve
  // eliminating x still leaves a genuine LP for the engine to iterate on. A model presolve
  // can finish by itself would let this test keep passing while exercising nothing, since
  // the stream would then carry a single trivial line.
  //
  //   min 3x + y + 2z + w
  //   s.t.  x + y + z      >= 10
  //             y      + w >=  4
  //                 z  + w <=  8
  //         x == 5,  y, z, w >= 0
  //
  // presolve fixes x at 5 and folds 3 * 5 = 15 into the reduced objective_offset.
  Model model =
      make_lp({{1.0, 1.0, 1.0, 0.0}, {0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}},
              {10.0, 4.0, -kInfinity}, {kInfinity, kInfinity, 8.0}, {3.0, 1.0, 2.0, 1.0});
  model.col_lower[0] = 5.0;
  model.col_upper[0] = 5.0;
  ASSERT_EQ(model.validate(), "");

  for (const bool presolve : {true, false}) {
    const testing::TempFile file("", ".jsonl");
    Options options = pdhg_options(1e-10);
    options.set_bool("presolve", presolve);
    options.set_string("progress_out", file.path());

    const Solution solution = solve(model, options);
    ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;

    // The engine has to have actually run. If presolve ever grows strong enough to finish
    // this model on its own, that is fine for the solver and fatal for this test, and it
    // should say so rather than pass silently on an empty stream.
    EXPECT_GT(solution.iterations, 1)
        << "presolve=" << presolve << ": the engine did no work, so nothing was exercised";

    // The LAST line of the stream is the iterate the solver stopped on, so it is the one
    // that has to agree with what the result reports.
    std::ifstream in(file.path());
    std::string line;
    double last_objective = 0.0;
    int lines = 0;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) continue;
      const nlohmann::json parsed = nlohmann::json::parse(line);
      ASSERT_TRUE(parsed.contains("best_bound"));
      if (parsed["best_bound"].is_number()) {
        last_objective = parsed["best_bound"].get<double>();
        ++lines;
      }
    }
    ASSERT_GT(lines, 0) << "no progress lines written with presolve=" << presolve;

    // Tolerance is the solver's, not one chosen to make this pass: the stream reports an
    // iterate and the result reports the accepted point, so they agree to convergence
    // tolerance rather than exactly. The bug this guards against was off by 15, not 1e-9.
    EXPECT_NEAR(last_objective, solution.objective, 1e-4)
        << "presolve=" << presolve << ": the progress stream and the reported objective "
        << "disagree by " << std::fabs(last_objective - solution.objective);
  }
}

}  // namespace
}  // namespace sankhya

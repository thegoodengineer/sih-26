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
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

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

TEST(SolveStatusGuard, AnInfeasiblePointIsNeverReportedAsOptimal) {
  const Model model = make_blend_lp();
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "pdhg");
  options.set_double("pdhg_tolerance", 0.01);

  const Solution solution = solve(model, options);

  // The engine's own measurement is what convicts it, so assert on that first: if this
  // stops holding the test has lost its subject and must be re-tuned, not deleted.
  ASSERT_GT(solution.primal_infeasibility, options.get_double("primal_feasibility_tolerance"))
      << "this tolerance no longer produces an infeasible point; pick a looser one";

  EXPECT_NE(solution.status, SolveStatus::kOptimal);
  EXPECT_NE(solution.status, SolveStatus::kFeasible)
      << "a point that violates its own constraints is not feasible either";
  EXPECT_EQ(solution.status, SolveStatus::kNumericalError);
  EXPECT_NE(solution.message.find("primal feasibility"), std::string::npos) << solution.message;
}

TEST(SolveStatusGuard, PrimalFeasibleButDualInfeasibleIsFeasibleNotOptimal) {
  // The other half of the rule. The point satisfies every constraint, so it is usable and
  // kFeasible is honest - but the reduced costs do not support a claim of optimality, and
  // kOptimal is a claim of proof.
  const Model model = make_blend_lp();
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "pdhg");
  options.set_double("pdhg_tolerance", 0.1);

  const Solution solution = solve(model, options);

  ASSERT_LE(solution.primal_infeasibility, options.get_double("primal_feasibility_tolerance"))
      << "expected a primal-feasible point at this tolerance";
  ASSERT_GT(solution.dual_infeasibility, options.get_double("dual_feasibility_tolerance"))
      << "expected the duals to be short of tolerance at this setting";

  EXPECT_EQ(solution.status, SolveStatus::kFeasible);
  EXPECT_TRUE(solution.has_primal_values());
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
  EXPECT_DOUBLE_EQ(solution.primal_infeasibility, 0.0);
  EXPECT_DOUBLE_EQ(solution.dual_infeasibility, 0.0);
  EXPECT_NEAR(solution.objective, 214.14594594594595, 1e-9);
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

}  // namespace
}  // namespace sankhya

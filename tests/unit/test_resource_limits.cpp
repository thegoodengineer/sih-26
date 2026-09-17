// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the unified resource-limit manager (#289).
//
// Two halves. The first tests ResourceLimits directly, with elapsed times passed in as
// numbers: the issue asks for limit tests that do not sleep, and a class that holds no clock
// is what makes that possible. The second half is the part that actually broke - what each
// ENGINE does with the same option - because the defects this file exists for were not in any
// one engine's arithmetic but in four engines answering one question four ways.
//
// Measured on main before the fix, on bandm.mps and lot_sizing.mps:
//
//   time_limit=0        the interior point returned OPTIMAL after 38 iterations; the simplex,
//                       the dual simplex and PDHG stopped at once
//   iteration_limit=0   the simplex and the dual simplex ran ONE iteration; the others none
//   node_limit=0        the MILP returned NUMERICAL_ERROR, and a dual bound of 0 that nothing
//                       had proved
//   iteration_limit=5   the MILP returned NUMERICAL_ERROR, the node LP having run out of
//                       iterations
//
// The last two are the ones that matter most: a resource limit reported as a numerical
// failure tells a user the solver broke when what happened is that it did as it was told.

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/resource_limits.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

ResourceLimits read(const Options& options) {
  Logger sink(nullptr);
  return ResourceLimits(options, sink);
}

/// A model that takes real iterations: transport-shaped, 25 columns, 25 rows.
Model lp(int size) {
  Model model;
  const auto n = static_cast<Index>(size);
  model.col_cost.assign(static_cast<std::size_t>(n), -1.0);
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 10.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(n), -kInfinity);
  model.row_upper.assign(static_cast<std::size_t>(n), 0.0);
  model.matrix.reset(n, n);
  for (Index i = 0; i < n; ++i) {
    model.row_upper[static_cast<std::size_t>(i)] = 40.0 + static_cast<double>(i % 5);
    for (Index j = 0; j < n; ++j) {
      model.matrix.add_entry(i, j, 1.0 + static_cast<double>((i * 13 + j * 7) % 8));
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

/// A MILP whose optimum is NEGATIVE, so that a dual bound of zero is a false claim rather
/// than a weak true one.
Model milp_with_a_negative_optimum() {
  Model model;
  const Index n = 14;
  model.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  model.matrix.reset(1, n);
  double total = 0.0;
  for (Index j = 0; j < n; ++j) {
    const double weight = 17.0 + static_cast<double>((j * 29) % 41);
    model.col_cost[static_cast<std::size_t>(j)] = -(weight + 3.0);  // minimise: go negative
    model.matrix.add_entry(0, j, weight);
    total += weight;
  }
  model.matrix.finalize();
  model.row_lower = {-kInfinity};
  model.row_upper = {std::floor(total / 2.0)};
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

// =========================================================================================
// The manager itself, with no clock and no solve
// =========================================================================================

TEST(ResourceLimits, TheDefaultsAreNoLimitAtAll) {
  const ResourceLimits limits = read(quiet());
  EXPECT_FALSE(limits.has_time_limit());
  EXPECT_EQ(limits.iteration_limit(), -1);
  EXPECT_EQ(limits.node_limit(), -1);
  EXPECT_FALSE(limits.time_exhausted(1e30));
  EXPECT_FALSE(limits.iterations_exhausted(1000000));
  EXPECT_FALSE(limits.nodes_exhausted(1000000));
  EXPECT_EQ(limits.exhausted(1e30, 1000000, 1000000), LimitReason::kNone);
}

TEST(ResourceLimits, ZeroSecondsIsABudgetOfZeroAndNotAnAbsentLimit) {
  // The ambiguity the issue asks to be resolved. Zero used to mean "no limit" in the
  // interior point and "stop now" everywhere else.
  Options options = quiet();
  options.set_double("time_limit", 0.0);
  const ResourceLimits limits = read(options);
  EXPECT_TRUE(limits.has_time_limit());
  EXPECT_TRUE(limits.time_exhausted(0.0));
  EXPECT_TRUE(limits.time_exhausted(1e-12));
  EXPECT_EQ(limits.remaining_seconds(0.0), 0.0);
}

TEST(ResourceLimits, ATimeLimitIsExhaustedOnlyOnceItIsPast) {
  Options options = quiet();
  options.set_double("time_limit", 10.0);
  const ResourceLimits limits = read(options);
  EXPECT_FALSE(limits.time_exhausted(0.0));
  EXPECT_FALSE(limits.time_exhausted(9.999));
  EXPECT_FALSE(limits.time_exhausted(10.0)) << "at the limit is not past it";
  EXPECT_TRUE(limits.time_exhausted(10.001));
  EXPECT_DOUBLE_EQ(limits.remaining_seconds(3.0), 7.0);
  EXPECT_DOUBLE_EQ(limits.remaining_seconds(30.0), 0.0) << "never negative";
}

TEST(ResourceLimits, ACountOfNMeansAtMostNOfThem) {
  Options options = quiet();
  options.set_int("iteration_limit", 3);
  options.set_int("node_limit", 5);
  const ResourceLimits limits = read(options);
  EXPECT_FALSE(limits.iterations_exhausted(0));
  EXPECT_FALSE(limits.iterations_exhausted(2));
  EXPECT_TRUE(limits.iterations_exhausted(3));
  EXPECT_FALSE(limits.nodes_exhausted(4));
  EXPECT_TRUE(limits.nodes_exhausted(5));

  options.set_int("iteration_limit", 0);
  options.set_int("node_limit", 0);
  const ResourceLimits none = read(options);
  EXPECT_TRUE(none.iterations_exhausted(0)) << "a budget of zero is exhausted before it starts";
  EXPECT_TRUE(none.nodes_exhausted(0));
}

TEST(ResourceLimits, TheDocumentedPrecedenceIsWhatTheManagerReports) {
  Options options = quiet();
  options.set_double("time_limit", 1.0);
  options.set_int("iteration_limit", 10);
  options.set_int("node_limit", 10);
  const ResourceLimits limits = read(options);

  EXPECT_EQ(limits.exhausted(2.0, 10, 10), LimitReason::kTime) << "time outranks the counters";
  EXPECT_EQ(limits.exhausted(0.5, 10, 10), LimitReason::kIterations)
      << "iterations outrank nodes";
  EXPECT_EQ(limits.exhausted(0.5, 3, 10), LimitReason::kNodes);
  EXPECT_EQ(limits.exhausted(0.5, 3, 3), LimitReason::kNone);
}

TEST(ResourceLimits, AnImpossibleConfigurationIsRefusedRatherThanObeyed) {
  // The registry's range check catches these on the way in from a string; set_double and
  // set_int do not validate, so a program that calls them reaches the manager directly.
  Options options = quiet();
  options.set_double("time_limit", -5.0);
  options.set_int("iteration_limit", -99);
  options.set_int("node_limit", -99);
  const ResourceLimits limits = read(options);
  EXPECT_FALSE(limits.has_time_limit()) << "a negative duration is not a limit of zero";
  EXPECT_EQ(limits.iteration_limit(), -1);
  EXPECT_EQ(limits.node_limit(), -1);
  EXPECT_EQ(limits.exhausted(1e9, 1000000, 1000000), LimitReason::kNone);

  Options nan_seconds = quiet();
  nan_seconds.set_double("time_limit", std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE(read(nan_seconds).has_time_limit());
}

TEST(ResourceLimits, EveryReasonHasAStatusAndAName) {
  EXPECT_EQ(status_for(LimitReason::kTime), SolveStatus::kTimeLimit);
  EXPECT_EQ(status_for(LimitReason::kIterations), SolveStatus::kIterationLimit);
  EXPECT_EQ(status_for(LimitReason::kNodes), SolveStatus::kNodeLimit);
  EXPECT_EQ(status_for(LimitReason::kInterrupt), SolveStatus::kInterrupted);
  EXPECT_STREQ(to_string(LimitReason::kNone), "none");
  EXPECT_STREQ(to_string(LimitReason::kTime), "time_limit");
  EXPECT_STREQ(to_string(LimitReason::kNodes), "node_limit");

  // None of the resource statuses may collide with a mathematical verdict.
  for (const LimitReason reason : {LimitReason::kInterrupt, LimitReason::kTime,
                                   LimitReason::kIterations, LimitReason::kNodes}) {
    const SolveStatus status = status_for(reason);
    EXPECT_NE(status, SolveStatus::kOptimal);
    EXPECT_NE(status, SolveStatus::kInfeasible);
    EXPECT_NE(status, SolveStatus::kUnbounded);
    EXPECT_NE(status, SolveStatus::kNumericalError);
  }
}

// =========================================================================================
// What the ENGINES do with the same option
// =========================================================================================

TEST(ResourceLimits, EveryLpEngineReadsAnIterationLimitTheSameWay) {
  const Model model = lp(25);
  for (const char* engine : {"simplex", "dual-simplex", "ipm", "pdhg"}) {
    Options options = quiet();
    options.set_string("algorithm", engine);
    options.set_bool("presolve", false);  // presolve would settle this model outright
    // The polish is a SECOND engine with a budget of its own (#229), and the test below
    // covers what the two-phase route does with a limit; here each engine answers for its
    // own iterations.
    options.set_bool("pdhg_polish", false);

    options.set_int("iteration_limit", 0);
    const Solution none = solve(model, options);
    EXPECT_EQ(none.status, SolveStatus::kIterationLimit) << engine << ": " << none.message;
    EXPECT_EQ(none.iterations, 0) << engine << ": a budget of zero bought " << none.iterations;
    EXPECT_EQ(none.stopped_by, LimitReason::kIterations) << engine;

    options.set_int("iteration_limit", 3);
    const Solution three = solve(model, options);
    EXPECT_EQ(three.status, SolveStatus::kIterationLimit) << engine << ": " << three.message;
    EXPECT_EQ(three.iterations, 3) << engine;
  }
}

TEST(ResourceLimits, TheTwoPhaseRouteSpendsTwoBudgetsAndSaysSo) {
  // The one documented place where the iterations reported exceed iteration_limit: PDHG
  // stops at its limit and the interior-point polish finishes the point on
  // polish_iteration_limit of its own (#229). Changing that quietly under the banner of
  // consistent limits would undo a measured feature, so it is asserted instead.
  const Model model = lp(25);
  Options options = quiet();
  options.set_string("algorithm", "pdhg");
  options.set_bool("presolve", false);
  options.set_bool("pdhg_polish", true);
  options.set_int("iteration_limit", 5);
  const Solution polished = solve(model, options);
  ASSERT_GT(polished.polish_iterations, 0) << polished.message;
  EXPECT_EQ(polished.iterations, 5 + polished.polish_iterations)
      << "the count is the sum of both phases: " << polished.message;
  EXPECT_NE(polished.algorithm.find("+ipm"), std::string::npos) << polished.algorithm;
}

TEST(ResourceLimits, EveryLpEngineReadsAZeroTimeLimitTheSameWay) {
  const Model model = lp(25);
  for (const char* engine : {"simplex", "dual-simplex", "ipm", "pdhg"}) {
    Options options = quiet();
    options.set_string("algorithm", engine);
    options.set_bool("presolve", false);
    options.set_double("time_limit", 0.0);
    const Solution stopped = solve(model, options);
    EXPECT_EQ(stopped.status, SolveStatus::kTimeLimit) << engine << ": " << stopped.message;
    EXPECT_EQ(stopped.stopped_by, LimitReason::kTime) << engine;
  }
}

TEST(ResourceLimits, TimeOutranksTheCountersWhenBothAreExhausted) {
  const Model model = lp(25);
  for (const char* engine : {"simplex", "dual-simplex", "ipm", "pdhg"}) {
    Options options = quiet();
    options.set_string("algorithm", engine);
    options.set_bool("presolve", false);
    options.set_double("time_limit", 0.0);
    options.set_int("iteration_limit", 0);
    const Solution stopped = solve(model, options);
    EXPECT_EQ(stopped.status, SolveStatus::kTimeLimit) << engine << ": " << stopped.message;
  }
}

TEST(ResourceLimits, AGenerousLimitChangesNothingAboutTheAnswer) {
  // The regression the issue asks for: limits that cannot bind must leave the mathematics
  // alone, on every engine.
  const Model model = lp(25);
  for (const char* engine : {"simplex", "dual-simplex", "ipm"}) {
    Options plain = quiet();
    plain.set_string("algorithm", engine);
    const Solution unrestricted = solve(model, plain);
    ASSERT_EQ(unrestricted.status, SolveStatus::kOptimal) << engine << unrestricted.message;

    Options generous = plain;
    generous.set_double("time_limit", 3600.0);
    generous.set_int("iteration_limit", 1000000);
    generous.set_int("node_limit", 1000000);
    const Solution bounded = solve(model, generous);
    EXPECT_EQ(bounded.status, SolveStatus::kOptimal) << engine << ": " << bounded.message;
    EXPECT_NEAR(bounded.objective, unrestricted.objective, 1e-9) << engine;
    EXPECT_EQ(bounded.stopped_by, LimitReason::kNone) << engine;
  }
}

TEST(ResourceLimits, ANodeLimitOfZeroIsALimitAndNotANumericalFailure) {
  const Model model = milp_with_a_negative_optimum();
  Options options = quiet();
  options.set_int("node_limit", 0);
  const Solution stopped = solve(model, options);

  EXPECT_EQ(stopped.status, SolveStatus::kNodeLimit) << stopped.message;
  EXPECT_NE(stopped.status, SolveStatus::kNumericalError);
  EXPECT_EQ(stopped.stopped_by, LimitReason::kNodes);
  EXPECT_EQ(stopped.nodes, 0);
  EXPECT_TRUE(stopped.col_value.empty())
      << "no point was found, so none may be reported: " << stopped.message;
}

TEST(ResourceLimits, ASearchThatProvedNothingReportsNoBound) {
  // The root node's bound field starts at 0.0, meaning "inherited from the parent", and the
  // root has no parent. A tree stopped before it evaluated the root used to report that 0 as
  // a proven dual bound. On this model the optimum is NEGATIVE, so a bound of 0 is not a
  // weak true statement, it is a false one.
  const Model model = milp_with_a_negative_optimum();
  Options unrestricted = quiet();
  const Solution solved = solve(model, unrestricted);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  ASSERT_LT(solved.objective, 0.0) << "the model was meant to have a negative optimum";

  Options options = quiet();
  options.set_int("node_limit", 0);
  const Solution stopped = solve(model, options);
  EXPECT_FALSE(stopped.dual_bound > solved.objective)
      << "a bound of " << stopped.dual_bound << " claims the optimum is no better than that, "
      << "and the optimum is " << solved.objective;
  EXPECT_TRUE(std::isinf(stopped.dual_bound)) << stopped.dual_bound;
  EXPECT_TRUE(std::isinf(stopped.relative_gap)) << "an unknown gap is not a closed one";
}

TEST(ResourceLimits, ANodeLimitKeepsTheIncumbentAndTheBoundAndTheGap) {
  const Model model = milp_with_a_negative_optimum();
  // Nine nodes is where this model's dive reaches its first integer point; under that it
  // reports the fractional relaxation with the message saying so (#223), which the test
  // below this one covers.
  Options options = quiet();
  options.set_int("node_limit", 10);
  const Solution stopped = solve(model, options);

  ASSERT_EQ(stopped.status, SolveStatus::kFeasible) << stopped.message;
  EXPECT_EQ(stopped.stopped_by, LimitReason::kNodes)
      << "kFeasible cannot say which limit stopped it; the field must";
  EXPECT_LE(stopped.nodes, 10);
  EXPECT_FALSE(stopped.col_value.empty()) << "the incumbent is preserved";
  EXPECT_TRUE(std::isfinite(stopped.objective));
  EXPECT_TRUE(std::isfinite(stopped.dual_bound));
  EXPECT_TRUE(std::isfinite(stopped.relative_gap));
}

TEST(ResourceLimits, ALimitedSearchWithNoIncumbentSaysWhatThePointIs) {
  // Under nine nodes this model has no integer point yet. The convention (#223) is to report
  // the last LP relaxation and say in the message that it is fractional - a limit is not a
  // reason to return nothing, and it is not a reason to call a fractional point a solution.
  const Model model = milp_with_a_negative_optimum();
  Options options = quiet();
  options.set_int("node_limit", 4);
  const Solution stopped = solve(model, options);
  ASSERT_EQ(stopped.status, SolveStatus::kNodeLimit) << stopped.message;
  EXPECT_EQ(stopped.stopped_by, LimitReason::kNodes);
  EXPECT_NE(stopped.message.find("fractional"), std::string::npos) << stopped.message;
  EXPECT_GT(stopped.integrality_violation, 0.0);
}

TEST(ResourceLimits, ANodeLpOutOfIterationsIsALimitAndNotANumericalFailure) {
  // The iteration limit reaches the tree through the node engine. A node LP that stops on it
  // leaves a node whose bound is unknown, so the search cannot go on - but a search that was
  // told to stop is not a solver that broke.
  const Model model = milp_with_a_negative_optimum();
  Options options = quiet();
  options.set_int("iteration_limit", 2);
  const Solution stopped = solve(model, options);
  EXPECT_NE(stopped.status, SolveStatus::kNumericalError) << stopped.message;
  EXPECT_EQ(stopped.status, SolveStatus::kIterationLimit) << stopped.message;
  EXPECT_EQ(stopped.stopped_by, LimitReason::kIterations);
}

TEST(ResourceLimits, TheReasonIsNoneWhenNothingStoppedTheSolve) {
  const Model model = lp(20);
  const Solution solved = solve(model, quiet());
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_EQ(solved.stopped_by, LimitReason::kNone);

  const Model integer_model = milp_with_a_negative_optimum();
  const Solution closed = solve(integer_model, quiet());
  ASSERT_EQ(closed.status, SolveStatus::kOptimal) << closed.message;
  EXPECT_EQ(closed.stopped_by, LimitReason::kNone);
}

TEST(ResourceLimits, TheBudgetIsSharedBetweenPresolveAndTheEngine) {
  // The clock starts when the solve does. An engine given the full time_limit after presolve
  // had already spent part of it would run for longer than the caller allowed.
  Options options = quiet();
  options.set_bool("presolve", true);
  options.set_double("time_limit", 0.0);
  const Solution stopped = solve(lp(25), options);
  EXPECT_EQ(stopped.status, SolveStatus::kTimeLimit) << stopped.message;
  EXPECT_EQ(stopped.stopped_by, LimitReason::kTime);
}

}  // namespace
}  // namespace sankhya

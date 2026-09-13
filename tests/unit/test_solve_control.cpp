// SPDX-License-Identifier: Apache-2.0
// SANKHYA - progress callback and cooperative interruption (#223).
//
// Two things are tested at the unit level, directly against SolveControl, rather than by
// timing a real solve: the callback's own rate limit (by iteration/node count, and by wall
// time) is arithmetic SolveControl owns outright, and testing it through an actual solve
// would make the test's pass/fail depend on how fast the machine running it happens to be -
// exactly the kind of nondeterminism CLAUDE.md's evidence rules warn about elsewhere in this
// project. The rest of the file is integration coverage: a real simplex solve honouring an
// interrupt, and a real branch-and-bound search stopped by a callback partway through a tree
// that would otherwise take thousands of nodes to close.

#include <atomic>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Model make_lp(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
             const std::vector<double>& row_upper, const std::vector<double>& cost,
             const std::vector<double>& col_lower, const std::vector<double>& col_upper) {
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
  return model;
}

/// A strongly-correlated 0/1 knapsack (Pisinger's term): value_i = weight_i + a constant
/// surplus. Every item has nearly the same value/weight ratio, so the LP relaxation's bound
/// discriminates poorly between items and branch-and-bound is forced to explore genuinely
/// many nodes rather than pruning most of the tree on the bound alone - measured below at
/// tens of thousands of nodes before the gap target closes it, for the sizes used here.
Model make_hard_knapsack(int n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> weight_dist(1, 1000);
  std::vector<double> weights(static_cast<std::size_t>(n));
  std::vector<double> values(static_cast<std::size_t>(n));
  double capacity = 0.0;
  for (int j = 0; j < n; ++j) {
    const double w = static_cast<double>(weight_dist(rng));
    weights[static_cast<std::size_t>(j)] = w;
    values[static_cast<std::size_t>(j)] = w + 500.0;  // strongly correlated
    capacity += w;
  }
  capacity *= 0.5;

  Model model;
  model.sense = ObjSense::kMaximize;
  model.col_cost = values;
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  model.row_lower = {-kInfinity};
  model.row_upper = {capacity};
  model.matrix.reset(1, n);
  for (Index j = 0; j < n; ++j) {
    model.matrix.add_entry(0, j, weights[static_cast<std::size_t>(j)]);
  }
  model.matrix.finalize();
  return model;
}

Options quiet_options() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

// =========================================================================================
// SolveControl's own throttle - unit tests, no solve involved.
// =========================================================================================

/// #223 acceptance criterion: "a test that the callback rate is bounded (a callback that
/// counts, on an LP of 10,000 iterations, is called fewer than N times)". Simulated
/// directly against poll() rather than a real 10,000-iteration LP so the count is
/// arithmetic, not a race against the machine's speed.
TEST(SolveControl, CallbackRateIsBoundedByIterationCount) {
  SolveControl control;
  int calls = 0;
  control.set_callback([&](const Progress&) {
    ++calls;
    return false;
  });
  // min_interval_seconds = 0 isolates the COUNT gate: with it at its default (0.1s) every
  // call here (elapsed_seconds pinned at 0.0, as a real 10,000-iteration LP that finishes
  // in under 100ms would also see) would be suppressed after the first, which would pass
  // trivially without exercising "every k iterations" at all.
  control.set_throttle(/*every_iterations=*/200, /*every_nodes=*/1, /*min_interval_seconds=*/0.0);

  Progress progress;
  progress.phase = SolvePhase::kLp;
  for (Count i = 0; i < 10000; ++i) {
    progress.iterations = i;
    EXPECT_FALSE(control.poll(progress));
  }
  // Exactly every 200th iteration (0, 200, 400, ..., 9800): 10000 / 200 = 50.
  EXPECT_EQ(calls, 50);
  EXPECT_LT(calls, 10000) << "the whole point: nowhere near one call per iteration";
}

/// The other half of "every k iterations... at most every 100 ms": on a loop fast enough
/// that many iterations pass within one wall-clock tick, the 100 ms ceiling is what
/// actually governs, independent of the iteration-count gate.
TEST(SolveControl, CallbackRateIsBoundedByWallClock) {
  SolveControl control;
  int calls = 0;
  control.set_callback([&](const Progress&) {
    ++calls;
    return false;
  });
  // every_iterations = 1: the count gate never binds, so only the 100 ms floor does.
  control.set_throttle(/*every_iterations=*/1, /*every_nodes=*/1, /*min_interval_seconds=*/0.1);

  Progress progress;
  progress.phase = SolvePhase::kLp;
  // 10,000 "iterations" spread over one second of simulated wall time: a real solve fast
  // enough to do 10,000 iterations/second, which the 100ms ceiling should cap at ~10 calls.
  for (Count i = 0; i < 10000; ++i) {
    progress.iterations = i;
    progress.elapsed_seconds = static_cast<double>(i) * 0.0001;
    (void)control.poll(progress);
  }
  EXPECT_GE(calls, 8);
  EXPECT_LE(calls, 13);
}

TEST(SolveControl, TheVeryFirstPollAlwaysFiresRegardlessOfThrottle) {
  // Every engine wants at least one snapshot even on a solve that finishes before any
  // throttle window elapses - otherwise a fast solve would report nothing at all.
  SolveControl control;
  int calls = 0;
  control.set_callback([&](const Progress&) {
    ++calls;
    return false;
  });
  Progress progress;
  progress.phase = SolvePhase::kLp;
  progress.iterations = 0;
  (void)control.poll(progress);
  EXPECT_EQ(calls, 1);
}

TEST(SolveControl, InterruptIsSeenImmediatelyEvenBetweenThrottledCallbackFirings) {
  // "External interruption checks are not throttled" (#223): interrupt() must be visible on
  // the very next poll(), never delayed by the callback's own rate limit.
  SolveControl control;
  int calls = 0;
  control.set_callback([&](const Progress&) {
    ++calls;
    return false;
  });
  control.set_throttle(1000000, 1000000, 1000.0);  // the callback itself would never fire
  Progress progress;
  progress.phase = SolvePhase::kLp;
  progress.iterations = 0;
  EXPECT_FALSE(control.poll(progress));  // first call: unthrottled, fires once
  EXPECT_EQ(calls, 1);

  control.interrupt();
  progress.iterations = 1;
  EXPECT_TRUE(control.poll(progress));
  EXPECT_EQ(calls, 1) << "the callback was never called again - is_interrupted() short-circuits";
}

TEST(SolveControl, ResetClearsInterruptAndThrottleHistoryButKeepsTheCallback) {
  SolveControl control;
  int calls = 0;
  control.set_callback([&](const Progress&) {
    ++calls;
    return false;
  });
  Progress progress;
  progress.phase = SolvePhase::kLp;
  progress.iterations = 0;
  (void)control.poll(progress);
  control.interrupt();
  ASSERT_TRUE(control.is_interrupted());

  control.reset();
  EXPECT_FALSE(control.is_interrupted());
  // The callback is still registered and the throttle history was cleared, so the very
  // next poll() fires again exactly like a fresh SolveControl's first call would.
  progress.iterations = 0;
  EXPECT_FALSE(control.poll(progress));
  EXPECT_EQ(calls, 2);
}

// =========================================================================================
// Integration: a real solve honouring interruption.
// =========================================================================================

TEST(SolveControl, InterruptsAPrimalSimplexSolveAndReportsKInterrupted) {
  // min -x - y  s.t.  x + y <= 100, x, y >= 0. Trivial to solve, but callback-driven
  // interruption doesn't care how easy the problem is: it stops on the FIRST poll(), which
  // fires unconditionally regardless of throttle (see TheVeryFirstPollAlwaysFiresAbove).
  const Model model = make_lp({{1.0, 1.0}}, {-kInfinity}, {100.0}, {-1.0, -1.0}, {0.0, 0.0},
                              {kInfinity, kInfinity});
  SolveControl control;
  control.set_callback([](const Progress&) { return true; });

  const Solution solution = solve(model, quiet_options(), &control);
  EXPECT_EQ(solution.status, SolveStatus::kInterrupted);
  EXPECT_EQ(solution.col_value.size(), 2u);
}

TEST(SolveControl, ExternalInterruptStopsASolveWithoutACallback) {
  // sankhya_model_interrupt()'s C API contract, exercised at the C++ level: interrupt()
  // called before solve() starts still takes effect, with no callback installed at all.
  const Model model = make_lp({{1.0, 1.0}}, {-kInfinity}, {100.0}, {-1.0, -1.0}, {0.0, 0.0},
                              {kInfinity, kInfinity});
  SolveControl control;
  control.interrupt();

  const Solution solution = solve(model, quiet_options(), &control);
  EXPECT_EQ(solution.status, SolveStatus::kInterrupted);
}

TEST(SolveControl, ANullControlBehavesExactlyAsBeforeTheFeatureExisted) {
  // Every pre-#223 call site passes no SolveControl at all; this is the one guarantee that
  // matters most for not breaking them: nothing about their behaviour may change.
  const Model model = make_lp({{1.0, 1.0}}, {-kInfinity}, {100.0}, {-1.0, -1.0}, {0.0, 0.0},
                              {kInfinity, kInfinity});
  const Solution solution = solve(model, quiet_options());
  EXPECT_EQ(solution.status, SolveStatus::kOptimal);
}

// =========================================================================================
// Integration: branch and bound, stopped mid-tree.
// =========================================================================================

/// #223 acceptance criterion: "a test that installs a callback which stops after the 5th
/// call on a MILP that takes thousands of nodes, and asserts the status is interrupted, the
/// returned point is feasible, and nodes <= 5 * (rate)".
TEST(SolveControl, StopsAMilpAfterTheFifthCallbackCallWithAFeasiblePoint) {
  // Measured (not assumed, per CLAUDE.md): this instance, run to a proven optimum with the
  // gap targets forced to zero, takes on the order of tens of thousands of nodes over
  // several seconds - comfortably "thousands of nodes" - which this test never gets close
  // to needing, because it stops after 5 callback calls, at 1 node per call (the rate this
  // test's own SolveControl is configured with).
  const Model model = make_hard_knapsack(/*n=*/50, /*seed=*/2024);
  Options options = quiet_options();
  options.set_int("node_limit", 5000000);
  options.set_double("mip_relative_gap", 0.0);
  options.set_double("mip_absolute_gap", 0.0);

  SolveControl control;
  // every_nodes = 1, no wall-clock floor: the tree fires at exactly 1 node per callback
  // call, deterministically, regardless of how fast this machine happens to be.
  // every_iterations is set enormous so a NODE'S OWN LP relaxation (every node solves one,
  // reporting SolvePhase::kLp on this same SolveControl - see the comment on poll() about
  // the two phases sharing one control) never itself reaches the count gate; the one
  // exception is the very first poll() of the whole solve, which always fires regardless of
  // any throttle (TheVeryFirstPollAlwaysFiresRegardlessOfThrottle above) and happens to be
  // the root node's LP, not the tree - so the phase check below allows exactly that one.
  control.set_throttle(/*every_iterations=*/1'000'000'000, /*every_nodes=*/1,
                       /*min_interval_seconds=*/0.0);
  int calls = 0;
  constexpr int kStopAfterCall = 5;
  constexpr Count kRateNodesPerCall = 1;
  control.set_callback([&](const Progress& progress) {
    ++calls;
    EXPECT_TRUE(calls == 1 || progress.phase == SolvePhase::kTree)
        << "only the very first call may be the root LP's own report; call " << calls;
    return calls >= kStopAfterCall;
  });

  const Solution solution = solve(model, options, &control);

  EXPECT_EQ(solution.status, SolveStatus::kInterrupted);
  EXPECT_EQ(calls, kStopAfterCall);
  EXPECT_LE(solution.nodes, kStopAfterCall * kRateNodesPerCall + 1)
      << "+1: the node whose LP was already running when the 5th call asked to stop";
  // "the returned point is feasible": a real incumbent, not the MIP's "nothing found"
  // convention (worst representable objective, no point) that a node/time limit can also
  // report - claims_a_point(kInterrupted) is true either way, so the point itself is what
  // has to be checked.
  ASSERT_EQ(solution.col_value.size(), static_cast<std::size_t>(model.num_cols()));
  EXPECT_LE(solution.integrality_violation, tol::kIntegrality);
  EXPECT_LE(solution.primal_infeasibility, tol::kPrimalFeasibility);
}

}  // namespace
}  // namespace sankhya

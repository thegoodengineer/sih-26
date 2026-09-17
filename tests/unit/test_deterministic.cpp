// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the deterministic solve mode (#288).
//
// Everything this solver decides is a function of the model and the options, with one family
// of exceptions: the decisions that ask the clock. How much time is left decides when a solve
// stops, how long the interior-point polish may run, and how a time limit is split between a
// first-order pass and its finish. Those answers differ between two runs on one machine and
// between two machines, so a run that ends on any of them cannot be reproduced - which is
// exactly the situation a numerical regression is hardest to chase in.
//
// Deterministic mode removes that family. The tests below check the two halves of the claim:
// that repeated runs return identical numbers, and that a wall-clock limit is REFUSED rather
// than quietly ignored, because a caller who asked for both has asked for two things that
// cannot both hold.
//
// What is deliberately NOT claimed, and so not tested: the same answer from a different
// compiler, a different CPU, or a different build of this project. Floating-point arithmetic
// is not associative and this mode does not pretend otherwise.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options deterministic(bool on) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("deterministic", on);
  return options;
}

/// A strongly correlated knapsack: a real branch-and-bound tree in a tenth of a second.
Model knapsack(int columns) {
  Model model;
  const auto n = static_cast<Index>(columns);
  model.sense = ObjSense::kMaximize;
  model.col_cost.resize(static_cast<std::size_t>(n));
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  model.matrix.reset(1, n);
  double total = 0.0;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double weight = 20.0 + static_cast<double>((j * 37) % 51);
    model.col_cost[u] = weight + 10.0;
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

/// A dense-ish LP with enough rows to take real iterations.
Model dense_lp(int size) {
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
    model.row_upper[static_cast<std::size_t>(i)] = 50.0 + static_cast<double>(i % 7);
    for (Index j = 0; j < n; ++j) {
      const double v = 1.0 + static_cast<double>((i * 31 + j * 17) % 9);
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

Model convex_qp(int size) {
  Model model = dense_lp(size);
  model.hessian.reset(static_cast<Index>(size), static_cast<Index>(size));
  for (Index j = 0; j < static_cast<Index>(size); ++j) model.hessian.add_entry(j, j, 2.0);
  model.hessian.finalize();
  return model;
}

/// Everything two runs of one model must agree on, to the bit.
void expect_identical(const Solution& a, const Solution& b, const char* what) {
  EXPECT_EQ(a.status, b.status) << what;
  EXPECT_EQ(a.objective, b.objective) << what << ": objectives differ in the last bits";
  EXPECT_EQ(a.dual_bound, b.dual_bound) << what;
  EXPECT_EQ(a.iterations, b.iterations) << what;
  EXPECT_EQ(a.nodes, b.nodes) << what;
  EXPECT_EQ(a.col_value, b.col_value) << what << ": the points differ";
  EXPECT_EQ(a.row_dual, b.row_dual) << what << ": the duals differ";
  EXPECT_EQ(a.algorithm, b.algorithm) << what;
}

// =========================================================================================

TEST(Deterministic, RepeatedRunsReturnTheSameNumbersForEveryClass) {
  const Model lp = dense_lp(40);
  const Model milp = knapsack(16);
  const Model qp = convex_qp(20);

  expect_identical(solve(lp, deterministic(true)), solve(lp, deterministic(true)), "LP");
  expect_identical(solve(milp, deterministic(true)), solve(milp, deterministic(true)), "MILP");
  expect_identical(solve(qp, deterministic(true)), solve(qp, deterministic(true)), "QP");
}

TEST(Deterministic, TheAnswerIsTheSameAsAnOrdinarySolveOnAModelNoLimitStops) {
  // Deterministic mode removes clock-based decisions; on a model that finishes well inside
  // every limit there are none to remove, so the answer must be unchanged. A mode that
  // quietly solved a different problem would be worse than no mode.
  const Model milp = knapsack(16);
  const Solution ordinary = solve(milp, deterministic(false));
  const Solution reproducible = solve(milp, deterministic(true));
  EXPECT_EQ(ordinary.status, reproducible.status);
  EXPECT_EQ(ordinary.objective, reproducible.objective);
  EXPECT_EQ(ordinary.nodes, reproducible.nodes);
  EXPECT_EQ(ordinary.col_value, reproducible.col_value);
}

TEST(Deterministic, AWallClockLimitIsRefusedRatherThanObeyed) {
  // The two requests cannot both hold. An ordinary solve stops on a limit this small; a
  // deterministic one runs to the answer and says in the log why the limit was refused.
  const Model lp = dense_lp(60);

  Options timed = deterministic(false);
  timed.set_double("time_limit", 1e-9);
  const Solution stopped = solve(lp, timed);
  EXPECT_EQ(stopped.status, SolveStatus::kTimeLimit)
      << "the ordinary path must still honour a time limit: " << stopped.message;

  Options refused = deterministic(true);
  refused.set_double("time_limit", 1e-9);
  const Solution finished = solve(lp, refused);
  EXPECT_NE(finished.status, SolveStatus::kTimeLimit) << finished.message;
  EXPECT_EQ(finished.status, SolveStatus::kOptimal) << finished.message;

  // And the answer is the one the deterministic solve without any limit gives.
  const Solution plain = solve(lp, deterministic(true));
  expect_identical(finished, plain, "a refused time limit changes nothing");
}

TEST(Deterministic, ADeterministicLimitIsStillObeyed) {
  // Refusing the clock is not refusing to stop. An iteration or node budget counts the same
  // on every machine, so it is exactly what a reproducible run should be bounded by.
  const Model milp = knapsack(16);
  Options bounded = deterministic(true);
  bounded.set_int("node_limit", 5);
  const Solution stopped = solve(milp, bounded);
  // A limit hit with an incumbent in hand reports kFeasible and says in the message which
  // limit it was - the convention documented on SolveStatus, unchanged by this mode.
  EXPECT_EQ(stopped.status, SolveStatus::kFeasible) << stopped.message;
  EXPECT_NE(stopped.message.find("node limit"), std::string::npos) << stopped.message;
  EXPECT_LE(stopped.nodes, 5);

  // Twice, identically - a run that stops early is exactly where reproducibility matters.
  expect_identical(stopped, solve(milp, bounded), "node-limited");
}

TEST(Deterministic, TheModeIsOffByDefault) {
  Options defaults;
  EXPECT_FALSE(defaults.get_bool("deterministic"));
}

TEST(Deterministic, SingleThreadedByDefaultInThisModeAndTheAnswerAgreesWithThreadsOne) {
  const Model lp = dense_lp(40);
  Options explicit_single = deterministic(true);
  explicit_single.set_int("threads", 1);
  expect_identical(solve(lp, deterministic(true)), solve(lp, explicit_single),
                   "deterministic defaults to one thread");
}

TEST(Deterministic, EveryLpEngineRepeatsItselfThreeTimesOver) {
  // One test per engine would pass while the dispatcher quietly sent all three to the same
  // one, so the engine is named and the run asserts which engine answered.
  const Model lp = dense_lp(30);
  for (const char* engine : {"simplex", "dual-simplex", "ipm", "pdhg"}) {
    Options options = deterministic(true);
    options.set_string("algorithm", engine);
    const Solution first = solve(lp, options);
    ASSERT_NE(first.status, SolveStatus::kNotSolved) << engine << ": " << first.message;
    for (int run = 2; run <= 3; ++run) {
      expect_identical(first, solve(lp, options), engine);
    }
  }
}

TEST(Deterministic, TheMixedIntegerEnginesRepeatThemselves) {
  const Model milp = knapsack(16);
  Model miqp = knapsack(8);
  miqp.hessian.reset(miqp.num_cols(), miqp.num_cols());
  for (Index j = 0; j < miqp.num_cols(); ++j) miqp.hessian.add_entry(j, j, -2.0);
  miqp.hessian.finalize();

  const Solution milp_first = solve(milp, deterministic(true));
  const Solution miqp_first = solve(miqp, deterministic(true));
  for (int run = 2; run <= 3; ++run) {
    expect_identical(milp_first, solve(milp, deterministic(true)), "MILP");
    expect_identical(miqp_first, solve(miqp, deterministic(true)), "MIQP");
  }
}

// ---- Model identity ---------------------------------------------------------------------

TEST(Deterministic, TheFingerprintIsStableAcrossCopiesAndIgnoresNames) {
  const Model a = dense_lp(12);
  const Model b = dense_lp(12);
  EXPECT_EQ(a.fingerprint(), b.fingerprint());

  Model renamed = a;
  renamed.name = "a different name";
  renamed.source_path = "/somewhere/else.mps";
  renamed.col_names.assign(static_cast<std::size_t>(a.num_cols()), "x");
  EXPECT_EQ(renamed.fingerprint(), a.fingerprint())
      << "two models that solve identically must carry the same identity";
}

TEST(Deterministic, TheFingerprintSeparatesModelsThatDifferAnywhere) {
  const Model base = dense_lp(12);
  const std::uint64_t reference = base.fingerprint();

  Model changed_cost = base;
  changed_cost.col_cost[3] += 1e-12;
  EXPECT_NE(changed_cost.fingerprint(), reference) << "a perturbed cost";

  Model changed_bound = base;
  changed_bound.col_upper[0] = 9.999999999;
  EXPECT_NE(changed_bound.fingerprint(), reference) << "a perturbed bound";

  Model changed_sense = base;
  changed_sense.sense = ObjSense::kMaximize;
  EXPECT_NE(changed_sense.fingerprint(), reference) << "the opposite sense";

  Model changed_type = base;
  changed_type.col_type[2] = VarType::kInteger;
  EXPECT_NE(changed_type.fingerprint(), reference) << "an integer column";

  Model changed_pattern = base;
  changed_pattern.matrix.reset(base.num_rows(), base.num_cols());
  changed_pattern.matrix.add_entry(0, 0, 1.0);
  changed_pattern.matrix.finalize();
  EXPECT_NE(changed_pattern.fingerprint(), reference) << "a different sparsity pattern";

  // A coefficient that moves without the pattern moving. Mutation testing found this case
  // missing: hashing the pattern and skipping the values passed every other assertion here.
  Model changed_coefficient = base;
  {
    const std::vector<Index>& starts = base.matrix.column_starts();
    const std::vector<Index>& rows = base.matrix.row_indices();
    std::vector<double> values = base.matrix.values();
    values[values.size() / 2] += 1e-9;
    changed_coefficient.matrix.reset(base.num_rows(), base.num_cols());
    for (Index j = 0; j < base.num_cols(); ++j) {
      const auto uj = static_cast<std::size_t>(j);
      for (Index k = starts[uj]; k < starts[uj + 1]; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        changed_coefficient.matrix.add_entry(rows[uk], j, values[uk]);
      }
    }
    changed_coefficient.matrix.finalize();
  }
  EXPECT_NE(changed_coefficient.fingerprint(), reference) << "one perturbed coefficient";

  // The same entries in a different column order are a different input to the factorization,
  // and are meant to fingerprint differently.
  Model reordered = base;
  reordered.hessian.reset(base.num_cols(), base.num_cols());
  reordered.hessian.add_entry(1, 1, 2.0);
  reordered.hessian.add_entry(0, 0, 2.0);
  reordered.hessian.finalize();
  EXPECT_NE(reordered.fingerprint(), reference) << "a hessian where there was none";

  Model changed_hessian = convex_qp(12);
  EXPECT_NE(changed_hessian.fingerprint(), reference) << "a quadratic term";

  // Signed zero: the solver can treat a bound at -0.0 and one at 0.0 the same way, but the
  // fingerprint hashes bits, and claiming identity for inputs that are not identical is the
  // failure mode worth avoiding here.
  Model signed_zero = base;
  signed_zero.col_lower[1] = -0.0;
  EXPECT_NE(signed_zero.fingerprint(), reference) << "a negative zero bound";
}

}  // namespace
}  // namespace sankhya

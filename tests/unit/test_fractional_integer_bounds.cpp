// SPDX-License-Identifier: Apache-2.0
// SANKHYA - integer columns with fractional bounds, searched without presolve's help (#328).
//
// An integer column bounded [0.5, 2.5] is legal MPS: a bound computed as a ratio, a converted
// formulation. The branch and bound rounded such a column only while walking a row it appears
// in, so a column that appears in NO row - it only prices the objective - kept its fractional
// box. The down child x <= floor(0.5) = 0 then left the box crossed at [0.5, 0], the node LP
// came back at x = 0.5, and the search branched on the same column again, forever: the model
// below ran to any node limit reporting `feasible` with the root bound, and to 142 seconds
// without one. Presolve (#301) rounds these bounds first, which hid the defect by default;
// every test here therefore runs with presolve OFF, where the search has to stand on its own.
//
// The enumeration test is the one that matters beyond the reported model: small models with
// fractional integer bounds, some columns in rows and some not, solved by brute force over
// every integer point and by the search, which must agree on the status and the optimum.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options search_only(bool presolve) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", presolve);
  options.set_int("node_limit", 2000);  // a loop runs into this; a proof never gets near it
  return options;
}

/// The model from #328: min -x0 + x1, x0 <= 4 through a row, x0 in [0, 3], x1 in [0.5, 2.5],
/// both integer, x1 in no row. x1 can only be 1 or 2, so the optimum is -3 + 1 = -2.
Model reported_model() {
  Model model;
  model.col_cost = {-1.0, 1.0};
  model.col_lower = {0.0, 0.5};
  model.col_upper = {3.0, 2.5};
  model.col_type = {VarType::kInteger, VarType::kInteger};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.finalize();
  model.row_lower = {-kInfinity};
  model.row_upper = {4.0};
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  return model;
}

TEST(FractionalIntegerBounds, TheReportedModelClosesWithoutPresolve) {
  const Solution solved = solve(reported_model(), search_only(false));
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, -2.0, 1e-9);
  EXPECT_NEAR(solved.dual_bound, -2.0, 1e-9) << "the gap must CLOSE, not stop at a limit";
  EXPECT_LE(solved.nodes, 10) << "tens of nodes at most; this model needs one";
  ASSERT_EQ(solved.col_value.size(), 2u);
  EXPECT_NEAR(solved.col_value[1], 1.0, 1e-9);
}

TEST(FractionalIntegerBounds, PresolveOnAndOffAgree) {
  const Solution with = solve(reported_model(), search_only(true));
  const Solution without = solve(reported_model(), search_only(false));
  ASSERT_EQ(with.status, SolveStatus::kOptimal) << with.message;
  ASSERT_EQ(without.status, SolveStatus::kOptimal) << without.message;
  EXPECT_NEAR(with.objective, without.objective, 1e-9);
}

TEST(FractionalIntegerBounds, ABoxWithNoIntegerInItIsInfeasibleNotALoop) {
  // x1 in [0.3, 0.7] holds no integer at all. The search must say infeasible - quickly, and
  // not by running out of nodes, which is what an unrounded box would do.
  Model model = reported_model();
  model.col_lower[1] = 0.3;
  model.col_upper[1] = 0.7;
  const Solution solved = solve(model, search_only(false));
  EXPECT_EQ(solved.status, SolveStatus::kInfeasible) << solved.message;
  EXPECT_LE(solved.nodes, 10) << solved.message;
}

TEST(FractionalIntegerBounds, AMaximisationRoundsTheSameWay) {
  // The same column under a maximise sense: the relaxation now wants x1 at its UPPER bound
  // 2.5, so it is the up child that would cross. max -x0 - x1 over the same box, x1 in no row.
  Model model = reported_model();
  model.sense = ObjSense::kMaximize;
  model.col_cost = {1.0, 1.0};  // max x0 + x1: x0 = 3, x1 = 2 -> 5
  const Solution solved = solve(model, search_only(false));
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, 5.0, 1e-9);
  EXPECT_NEAR(solved.dual_bound, 5.0, 1e-9);
}

// ---- Against brute force --------------------------------------------------------------------

struct Enumerated {
  bool feasible = false;
  double best = std::numeric_limits<double>::infinity();  // minimise space
};

/// Every integer point of the box, checked against every row. Small boxes only.
Enumerated enumerate(const Model& model) {
  const std::size_t n = model.col_cost.size();
  std::vector<std::int64_t> lo(n);
  std::vector<std::int64_t> hi(n);
  for (std::size_t j = 0; j < n; ++j) {
    lo[j] = static_cast<std::int64_t>(std::ceil(model.col_lower[j] - 1e-9));
    hi[j] = static_cast<std::int64_t>(std::floor(model.col_upper[j] + 1e-9));
    if (lo[j] > hi[j]) return {};
  }
  Enumerated result;
  std::vector<std::int64_t> x = lo;
  const double sense = model.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  for (;;) {
    std::vector<double> point(n);
    for (std::size_t j = 0; j < n; ++j) point[j] = static_cast<double>(x[j]);
    bool ok = true;
    for (Index i = 0; i < model.num_rows() && ok; ++i) {
      double activity = 0.0;
      for (std::size_t j = 0; j < n; ++j)
        activity += model.matrix.at(i, static_cast<Index>(j)) * point[j];
      const auto ui = static_cast<std::size_t>(i);
      if (activity > model.row_upper[ui] + 1e-9 || activity < model.row_lower[ui] - 1e-9)
        ok = false;
    }
    if (ok) {
      double objective = 0.0;
      for (std::size_t j = 0; j < n; ++j) objective += model.col_cost[j] * point[j];
      result.feasible = true;
      result.best = std::min(result.best, sense * objective);
    }
    std::size_t k = 0;
    while (k < n && x[k] == hi[k]) {
      x[k] = lo[k];
      ++k;
    }
    if (k == n) break;
    ++x[k];
  }
  return result;
}

TEST(FractionalIntegerBounds, TheSearchAgreesWithBruteForceOnRandomFractionalBoxes) {
  std::mt19937 rng(328);
  std::uniform_real_distribution<double> fraction(0.05, 0.95);
  std::uniform_int_distribution<int> whole(-2, 2);
  std::uniform_int_distribution<int> width(0, 3);
  std::uniform_int_distribution<int> coefficient(-3, 3);
  std::bernoulli_distribution in_a_row(0.6);
  std::bernoulli_distribution fractional(0.7);

  int compared = 0;
  int infeasible = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const std::size_t n = 2 + static_cast<std::size_t>(trial % 3);
    const Index rows = 1 + trial % 2;
    Model model;
    model.sense = trial % 4 == 0 ? ObjSense::kMaximize : ObjSense::kMinimize;
    model.col_cost.resize(n);
    model.col_lower.resize(n);
    model.col_upper.resize(n);
    model.col_type.assign(n, VarType::kInteger);
    std::vector<bool> used(n);
    for (std::size_t j = 0; j < n; ++j) {
      const double base = static_cast<double>(whole(rng));
      model.col_lower[j] = fractional(rng) ? base + fraction(rng) : base;
      model.col_upper[j] =
          base + static_cast<double>(width(rng)) + (fractional(rng) ? fraction(rng) : 0.0);
      if (model.col_upper[j] < model.col_lower[j])
        std::swap(model.col_lower[j], model.col_upper[j]);
      model.col_cost[j] = static_cast<double>(coefficient(rng));
      used[j] = in_a_row(rng);  // a column in no row is the case #328 was about
    }
    model.matrix.reset(rows, static_cast<Index>(n));
    model.row_lower.assign(static_cast<std::size_t>(rows), -kInfinity);
    model.row_upper.assign(static_cast<std::size_t>(rows), 0.0);
    for (Index i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (!used[j]) continue;
        const int a = coefficient(rng);
        if (a != 0) model.matrix.add_entry(i, static_cast<Index>(j), static_cast<double>(a));
      }
      model.row_upper[static_cast<std::size_t>(i)] = static_cast<double>(whole(rng) + 2);
    }
    model.matrix.finalize();
    model.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
    model.hessian.finalize();

    const Enumerated truth = enumerate(model);
    const Solution solved = solve(model, search_only(false));
    ++compared;
    if (!truth.feasible) {
      ++infeasible;
      EXPECT_EQ(solved.status, SolveStatus::kInfeasible)
          << "trial " << trial << ": no integer point exists; " << solved.message;
      continue;
    }
    ASSERT_EQ(solved.status, SolveStatus::kOptimal)
        << "trial " << trial << ": enumeration found " << truth.best << "; " << solved.message;
    const double sense = model.sense == ObjSense::kMaximize ? -1.0 : 1.0;
    EXPECT_NEAR(sense * solved.objective, truth.best, 1e-6) << "trial " << trial;
    EXPECT_LT(solved.nodes, 2000) << "trial " << trial << " ran to the node limit";
  }
  EXPECT_EQ(compared, 300);
  EXPECT_GT(infeasible, 0) << "the generator should exercise the empty-box case too";
  EXPECT_LT(infeasible, compared) << "and mostly feasible ones";
}

}  // namespace
}  // namespace sankhya

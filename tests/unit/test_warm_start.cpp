// SPDX-License-Identifier: Apache-2.0
// SANKHYA - warm start through the public entry point (#218).
//
// THE NUMBER IS THE PROOF. A planner solves, moves a bound or a price, and solves again;
// the re-solve from the previous basis must take a small fraction of the cold solve's
// pivots, and the answer must be the cold solve's answer. Wall clock proves nothing on a
// machine whose speed changes minute to minute; the pivot count does not.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Options quiet(const char* algorithm) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", algorithm);
  return options;
}

Model netlib(const char* name) {
  Model model;
  const std::string path =
      (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
       "data/netlib" / (std::string(name) + ".mps"))
          .string();
  const io::ReadResult read = io::read_model(path, &model);
  EXPECT_TRUE(read.ok) << path << ": " << read.error;
  return model;
}

// SolveControl holds an atomic and cannot be moved, so the basis is copied into one in place.
void start_from(const Solution& previous, SolveControl* control) {
  control->start_col_status = previous.col_status;
  control->start_row_status = previous.row_status;
}

TEST(WarmStart, ATightenedBoundResolvesInAFewDualPivots) {
  // Solve (presolve on, the default), tighten one bound of a basic column, and re-solve
  // from the previous basis: the dual simplex restart. The old basis stays dual feasible,
  // so the pivots are the ones that restore primal feasibility - a handful.
  for (const char* name : {"afiro", "adlittle", "share2b", "israel"}) {
    Model model = netlib(name);
    const Solution first = solve(model, quiet("auto"));
    ASSERT_EQ(first.status, SolveStatus::kOptimal) << name << ": " << first.message;

    // The basic column with the largest value: halving its upper bound (or capping it when
    // it has none) is a change the optimum has to react to.
    Index pick = -1;
    for (Index j = 0; j < model.num_cols(); ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (first.col_status[u] != BasisStatus::kBasic) continue;
      if (pick < 0 || first.col_value[u] > first.col_value[static_cast<std::size_t>(pick)]) {
        pick = j;
      }
    }
    ASSERT_GE(pick, 0) << name;
    const auto p = static_cast<std::size_t>(pick);
    if (first.col_value[p] <= 0.0) continue;  // nothing to tighten on this instance
    model.col_upper[p] = 0.5 * first.col_value[p];

    const Solution cold = solve(model, quiet("auto"));
    SolveControl control;
    start_from(first, &control);
    const Solution warm = solve(model, quiet("auto"), &control);
    ASSERT_EQ(warm.status, cold.status) << name << ": " << warm.message;
    if (cold.status != SolveStatus::kOptimal) continue;  // tightening can make it infeasible
    EXPECT_NEAR(warm.objective, cold.objective, 1e-6 * std::max(1.0, std::fabs(cold.objective)))
        << name;
    EXPECT_NE(warm.message.find("warm start"), std::string::npos) << warm.message;
    EXPECT_LE(warm.iterations * 5, cold.iterations + 5)
        << name << ": warm " << warm.iterations << " pivots against " << cold.iterations
        << " cold";
    EXPECT_LE(warm.primal_infeasibility, tol::kPrimalFeasibility);
  }
}

TEST(WarmStart, AChangedCostResolvesInAFewPrimalPivots) {
  // A cost change keeps the old basis primal feasible, so the PRIMAL simplex is the restart.
  for (const char* name : {"afiro", "adlittle", "share2b"}) {
    Model model = netlib(name);
    const Solution first = solve(model, quiet("simplex"));
    ASSERT_EQ(first.status, SolveStatus::kOptimal) << name << ": " << first.message;
    // Make the cheapest basic column expensive: the optimum has to move away from it.
    Index pick = -1;
    for (Index j = 0; j < model.num_cols(); ++j) {
      const auto u = static_cast<std::size_t>(j);
      if (first.col_status[u] != BasisStatus::kBasic || first.col_value[u] <= 0.0) continue;
      if (pick < 0 || model.col_cost[u] < model.col_cost[static_cast<std::size_t>(pick)]) {
        pick = j;
      }
    }
    ASSERT_GE(pick, 0) << name;
    const auto p = static_cast<std::size_t>(pick);
    model.col_cost[p] += 10.0 * std::max(1.0, std::fabs(model.col_cost[p]));

    const Solution cold = solve(model, quiet("simplex"));
    SolveControl control;
    start_from(first, &control);
    const Solution warm = solve(model, quiet("simplex"), &control);
    ASSERT_EQ(warm.status, SolveStatus::kOptimal) << name << ": " << warm.message;
    ASSERT_EQ(cold.status, SolveStatus::kOptimal) << name << ": " << cold.message;
    EXPECT_NEAR(warm.objective, cold.objective, 1e-6 * std::max(1.0, std::fabs(cold.objective)))
        << name;
    EXPECT_LE(warm.iterations * 5, cold.iterations + 5)
        << name << ": warm " << warm.iterations << " pivots against " << cold.iterations
        << " cold";
  }
}

TEST(WarmStart, AWarmResolveThatTurnsInfeasibleStillCarriesACertificate) {
  // #218 meeting #217: the certificate is a property of the verdict, not of the start.
  Model model = netlib("afiro");
  const Solution first = solve(model, quiet("auto"));
  ASSERT_EQ(first.status, SolveStatus::kOptimal) << first.message;
  // Force a contradiction: cap every column at 1e3 and demand a row activity of 1e9, which
  // no point inside that box can reach (raising the row alone would make afiro unbounded
  // instead, through its free directions).
  for (double& upper : model.col_upper) upper = std::min(upper, 1e3);
  model.row_lower[0] = 1e9;
  model.row_upper[0] = kInfinity;
  SolveControl control;
  start_from(first, &control);
  const Solution warm = solve(model, quiet("auto"), &control);
  ASSERT_EQ(warm.status, SolveStatus::kInfeasible) << warm.message;
  EXPECT_EQ(static_cast<Index>(warm.farkas_dual.size()), model.num_rows()) << warm.message;
}

TEST(WarmStart, ABasisOfTheWrongShapeIsIgnoredAndTheSolveRunsCold) {
  Model model = netlib("afiro");
  SolveControl control;
  control.start_col_status.assign(3, BasisStatus::kBasic);
  control.start_row_status.assign(2, BasisStatus::kBasic);
  const Solution s = solve(model, quiet("auto"), &control);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.objective, -464.75314286, 1e-6 * 464.0);
  EXPECT_EQ(s.message.find("warm start"), std::string::npos) << s.message;
}

}  // namespace
}  // namespace sankhya

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MILP primal heuristics (#290).
//
// A heuristic that returns an infeasible point costs time, because offer_incumbent() checks
// it; one that makes the SEARCH return a wrong answer is the failure that matters. So the
// tests check each heuristic's claims independently here - every point it calls feasible is
// re-checked against every row, bound and integrality in this file - and then check that a
// search running all of them still agrees with brute force on every model.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mip/heuristics.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Rows given densely, all columns integer in [lower, upper].
Model integer_model(const std::vector<std::vector<double>>& rows,
                    const std::vector<double>& row_lower, const std::vector<double>& row_upper,
                    const std::vector<double>& cost, double lower, double upper) {
  Model m;
  const auto n = cost.size();
  m.col_cost = cost;
  m.col_lower.assign(n, lower);
  m.col_upper.assign(n, upper);
  m.col_type.assign(n, VarType::kInteger);
  m.matrix.reset(static_cast<Index>(rows.size()), static_cast<Index>(n));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (rows[i][j] != 0.0) {
        m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(j), rows[i][j]);
      }
    }
  }
  m.matrix.finalize();
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

std::vector<Index> all_columns(const Model& m) {
  std::vector<Index> cols;
  for (Index j = 0; j < m.num_cols(); ++j) cols.push_back(j);
  return cols;
}

/// Independent of everything under test: integral, inside the box, every row satisfied.
bool feasible(const Model& m, const std::vector<double>& x) {
  if (x.size() != static_cast<std::size_t>(m.num_cols())) return false;
  for (std::size_t j = 0; j < x.size(); ++j) {
    if (m.col_type[j] == VarType::kInteger && std::fabs(x[j] - std::round(x[j])) > 1e-9) {
      return false;
    }
    if (x[j] < m.col_lower[j] - 1e-9 || x[j] > m.col_upper[j] + 1e-9) return false;
  }
  for (Index i = 0; i < m.num_rows(); ++i) {
    double activity = 0.0;
    for (Index j = 0; j < m.num_cols(); ++j) {
      activity += m.matrix.at(i, j) * x[static_cast<std::size_t>(j)];
    }
    const auto u = static_cast<std::size_t>(i);
    if (activity < m.row_lower[u] - 1e-9 || activity > m.row_upper[u] + 1e-9) return false;
  }
  return true;
}

TEST(Heuristics, LocksCountTheRowsEachDirectionCouldBreak) {
  // row 0: x0 + x1 <= 3   (moving either UP can break it)
  // row 1: x0 - x2 >= 0   (x0 down, x2 up can break it)
  // row 2: x1 + x2 == 1   (both directions, both columns)
  const Model m = integer_model({{1, 1, 0}, {1, 0, -1}, {0, 1, 1}}, {-kInf, 0.0, 1.0},
                                {3.0, kInf, 1.0}, {0, 0, 0}, 0.0, 5.0);
  const Locks locks = compute_locks(m);
  EXPECT_EQ(locks.up[0], 1);
  EXPECT_EQ(locks.down[0], 1);
  EXPECT_EQ(locks.up[1], 2);
  EXPECT_EQ(locks.down[1], 1);
  EXPECT_EQ(locks.up[2], 2);
  EXPECT_EQ(locks.down[2], 1);
}

TEST(Heuristics, LockRoundingGoesTheWayNoRowObjectsTo) {
  // Covering: x0 + x1 >= 1 with x = (0.5, 0.5). Nearest rounding gives (0, 0) or (1, 1) by
  // the tie rule; locks say both columns can only be broken by moving DOWN, so both round up,
  // which is feasible.
  const Model m = integer_model({{1, 1}}, {1.0}, {kInf}, {1, 1}, 0.0, 1.0);
  const std::vector<double> x = lock_round(m, compute_locks(m), all_columns(m), {0.4, 0.4});
  EXPECT_EQ(x, (std::vector<double>{1.0, 1.0}));
  EXPECT_TRUE(feasible(m, x));
}

TEST(Heuristics, RepairShiftsTheCheapestColumnOutOfAViolation) {
  // x0 + x1 + x2 >= 2, costs 5, 1, 3; start at zero. Two shifts fix it; the cheapest column
  // is taken first, and the tie-break keeps the result the same on every run.
  const Model m = integer_model({{1, 1, 1}}, {2.0}, {kInf}, {5, 1, 3}, 0.0, 1.0);
  std::vector<double> x{0.0, 0.0, 0.0};
  Count moves = 0;
  ASSERT_TRUE(repair(m, all_columns(m), &x, 10, 1e-9, &moves));
  EXPECT_EQ(moves, 2);
  EXPECT_TRUE(feasible(m, x));
  EXPECT_EQ(x[1], 1.0) << "the column costing 1 moves first";
  EXPECT_EQ(x[2], 1.0) << "then the one costing 3";
  EXPECT_EQ(x[0], 0.0);
}

TEST(Heuristics, RepairAdmitsWhenItCannotHelp) {
  // x0 + x1 >= 3 with both columns in [0, 1]: no integer point exists.
  const Model m = integer_model({{1, 1}}, {3.0}, {kInf}, {1, 1}, 0.0, 1.0);
  std::vector<double> x{0.0, 0.0};
  Count moves = 0;
  EXPECT_FALSE(repair(m, all_columns(m), &x, 10, 1e-9, &moves));
}

TEST(Heuristics, RinsFixesWhereTheRelaxationAndTheIncumbentAgree) {
  const Model m = integer_model({{1, 1, 1, 1}}, {-kInf}, {3.0}, {1, 1, 1, 1}, 0.0, 1.0);
  Model sub;
  Count fixed = 0;
  ASSERT_TRUE(rins_submodel(m, all_columns(m), {1.0, 0.0, 0.6, 1.0}, {1.0, 0.0, 1.0, 1.0}, 0.5,
                            1e-6, &sub, &fixed));
  EXPECT_EQ(fixed, 3);
  EXPECT_EQ(sub.col_lower[0], 1.0);
  EXPECT_EQ(sub.col_upper[0], 1.0);
  EXPECT_EQ(sub.col_upper[1], 0.0);
  EXPECT_EQ(sub.col_lower[2], 0.0) << "the column they disagree on stays free";
  EXPECT_EQ(sub.col_upper[2], 1.0);
  EXPECT_FALSE(rins_submodel(m, all_columns(m), {0.5, 0.5, 0.5, 0.5}, {1.0, 0.0, 1.0, 1.0}, 0.5,
                             1e-6, &sub, &fixed))
      << "a neighbourhood with nothing fixed is not a neighbourhood";
}

TEST(Heuristics, TheFeasibilityPumpFindsAPointOnACoveringModel) {
  // Five items, three covering rows; the LP optimum is fractional.
  const Model m = integer_model({{1, 1, 0, 0, 1}, {0, 1, 1, 1, 0}, {1, 0, 1, 0, 1}},
                                {1.0, 1.0, 1.0}, {kInf, kInf, kInf}, {3, 2, 2, 4, 1}, 0.0, 1.0);
  Options lp;
  lp.set_bool("log_to_console", false);
  Count solves = 0;
  const std::vector<double> x =
      feasibility_pump(m, all_columns(m), {0.5, 0.5, 0.5, 0.5, 0.5}, lp, 20, 1e-6, &solves);
  ASSERT_FALSE(x.empty()) << "after " << solves << " projections";
  EXPECT_TRUE(feasible(m, x));
  EXPECT_GE(solves, 1);
}

TEST(Heuristics, EveryPointAHeuristicCallsFeasibleIsFeasible) {
  // Random covering / packing mixes. Whatever repair() returns true for, and whatever the
  // pump returns, is re-checked here by a function that shares no code with them.
  std::mt19937 rng(290);
  std::uniform_int_distribution<int> coefficient(0, 3);
  std::uniform_real_distribution<double> lp_value(0.0, 1.0);
  int repaired = 0;
  int pumped = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const std::size_t n = 6;
    std::vector<std::vector<double>> rows(3, std::vector<double>(n));
    std::vector<double> lower(3, -kInf);
    std::vector<double> upper(3, kInf);
    for (std::size_t i = 0; i < 3; ++i) {
      double sum = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        rows[i][j] = coefficient(rng);
        sum += rows[i][j];
      }
      if (i % 2 == 0) {
        lower[i] = std::floor(sum / 3.0);  // covering
      } else {
        upper[i] = std::ceil(sum / 2.0);  // packing
      }
    }
    std::vector<double> cost(n);
    for (double& c : cost) c = 1.0 + coefficient(rng);
    const Model m = integer_model(rows, lower, upper, cost, 0.0, 1.0);
    std::vector<double> x(n);
    for (double& v : x) v = lp_value(rng);

    std::vector<double> candidate = lock_round(m, compute_locks(m), all_columns(m), x);
    Count moves = 0;
    if (repair(m, all_columns(m), &candidate, 20, 1e-9, &moves)) {
      ++repaired;
      EXPECT_TRUE(feasible(m, candidate)) << "trial " << trial << ": repair claimed feasible";
    }
    Options lp;
    lp.set_bool("log_to_console", false);
    Count solves = 0;
    const std::vector<double> pumped_point =
        feasibility_pump(m, all_columns(m), x, lp, 10, 1e-6, &solves);
    if (!pumped_point.empty()) {
      ++pumped;
      EXPECT_TRUE(feasible(m, pumped_point)) << "trial " << trial << ": the pump returned it";
    }
  }
  EXPECT_GT(repaired, 50) << "the generator should give repair something to succeed at";
  EXPECT_GT(pumped, 50);
}

TEST(Heuristics, TheSearchWithEveryHeuristicForcedOnAgreesWithBruteForce) {
  // RINS at every node and the pump at every root: far more often than the defaults, so the
  // sub-MIP path and the pump's candidates are exercised on every model, and still the search
  // must report exactly what enumeration says - status and optimum.
  std::mt19937 rng(2905);
  std::uniform_int_distribution<int> coefficient(0, 4);
  std::uniform_int_distribution<int> cost_value(-5, 5);
  int compared = 0;
  int infeasible = 0;
  for (int trial = 0; trial < 120; ++trial) {
    const std::size_t n = 5 + static_cast<std::size_t>(trial % 5);
    std::vector<std::vector<double>> rows(3, std::vector<double>(n));
    std::vector<double> lower(3, -kInf);
    std::vector<double> upper(3, kInf);
    for (std::size_t i = 0; i < 3; ++i) {
      double sum = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        rows[i][j] = coefficient(rng);
        sum += rows[i][j];
      }
      if (i == 0) {
        upper[i] = std::floor(sum / 2.0);
      } else {
        lower[i] = std::floor(sum / 3.0);
      }
    }
    std::vector<double> cost(n);
    for (double& c : cost) c = cost_value(rng);
    Model m = integer_model(rows, lower, upper, cost, 0.0, 1.0);
    if (trial % 2 == 1) m.sense = ObjSense::kMaximize;

    // Brute force over all 2^n assignments.
    bool any = false;
    double best = 0.0;
    for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
      std::vector<double> x(n);
      for (std::size_t j = 0; j < n; ++j) x[j] = (mask >> j) & 1u ? 1.0 : 0.0;
      if (!feasible(m, x)) continue;
      double value = 0.0;
      for (std::size_t j = 0; j < n; ++j) value += cost[j] * x[j];
      if (!any || (m.sense == ObjSense::kMaximize ? value > best : value < best)) best = value;
      any = true;
    }

    for (const bool heuristics : {true, false}) {
      Options options;
      options.set_bool("log_to_console", false);
      options.set_bool("presolve", false);
      options.set_bool("mip_heuristics", heuristics);
      options.set_int("mip_rins_frequency", 1);
      options.set_int("mip_pump_rounds", 30);
      const Solution solved = solve(m, options);
      ++compared;
      if (!any) {
        ++infeasible;
        EXPECT_EQ(solved.status, SolveStatus::kInfeasible)
            << "trial " << trial << ": " << solved.message;
        continue;
      }
      ASSERT_EQ(solved.status, SolveStatus::kOptimal)
          << "trial " << trial << " heuristics=" << heuristics << ": " << solved.message;
      EXPECT_NEAR(solved.objective, best, 1e-6)
          << "trial " << trial << " heuristics=" << heuristics;
      EXPECT_TRUE(feasible(m, solved.col_value)) << "trial " << trial;
    }
  }
  EXPECT_EQ(compared, 240);
  EXPECT_LT(infeasible, compared);
}

}  // namespace
}  // namespace sankhya::mip

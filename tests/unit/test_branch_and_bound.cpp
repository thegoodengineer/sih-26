// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound tests.
//
// The gate here is FuzzAgainstTheExactMilpOracle, and it is the MILP counterpart of the LP
// fuzz in tests/oracles/. A branch and bound has a failure mode the LP solver does not: it
// can fathom a node it should have explored, discard the optimum, and then prove that the
// second-best answer is optimal. Nothing about that looks wrong from outside - the status
// says optimal, the point is integral and feasible, the bound matches the objective. Only a
// second, independent search over the same instance catches it.

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

Options mip_options() {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("node_limit", 20000);
  return options;
}

/// Build a MILP directly, so the expected optimum is arithmetic in the comment.
Model make_milp(const std::vector<std::vector<double>>& rows, const std::vector<double>& lower,
                const std::vector<double>& upper, const std::vector<double>& cost,
                const std::vector<double>& col_upper, const std::vector<bool>& integral) {
  Model model;
  const auto cols = static_cast<Index>(cost.size());
  const auto num_rows = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower.assign(static_cast<std::size_t>(cols), 0.0);
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(cols), VarType::kContinuous);
  for (Index j = 0; j < cols; ++j) {
    if (integral[static_cast<std::size_t>(j)]) {
      model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
    }
  }
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

TEST(BranchAndBound, ZeroOneKnapsack) {
  // values 10 7 4 3, weights 5 4 3 2, capacity 9. Taking the first two gives value 17 at
  // weight exactly 9; nothing else reaches 17. Minimising the negated value gives -17.
  const Model model =
      make_milp({{5.0, 4.0, 3.0, 2.0}}, {-kInfinity}, {9.0}, {-10.0, -7.0, -4.0, -3.0},
                {1.0, 1.0, 1.0, 1.0}, {true, true, true, true});
  const Solution s = solve(model, mip_options());
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, -17.0, 1e-9);
  EXPECT_NEAR(s.col_value[0], 1.0, 1e-6);
  EXPECT_NEAR(s.col_value[1], 1.0, 1e-6);
  EXPECT_DOUBLE_EQ(s.integrality_violation, 0.0);
}

TEST(BranchAndBound, TheRelaxationIsNotTheAnswer) {
  // max 5x s.t. 2x <= 5, x integer in [0, 10]. The relaxation gives x = 2.5 for 12.5; the
  // integer optimum is x = 2 for 10. A search that reported the relaxation would say 12.5.
  const Model model = make_milp({{2.0}}, {-kInfinity}, {5.0}, {-5.0}, {10.0}, {true});
  const Solution s = solve(model, mip_options());
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, -10.0, 1e-9);
  EXPECT_NEAR(s.col_value[0], 2.0, 1e-6);
}

TEST(BranchAndBound, MixedIntegerAndContinuous) {
  // min -x - y  s.t.  x + y <= 3.5,  x integer in [0,10], y continuous in [0,10].
  // x takes 3, y takes 0.5, objective -3.5.
  const Model model =
      make_milp({{1.0, 1.0}}, {-kInfinity}, {3.5}, {-1.0, -1.0}, {10.0, 10.0}, {true, false});
  const Solution s = solve(model, mip_options());
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, -3.5, 1e-7);
  EXPECT_NEAR(s.col_value[0], 3.0, 1e-6);
}

TEST(BranchAndBound, DetectsIntegerInfeasibility) {
  // 2x = 3 with x integer has no solution, though the relaxation is perfectly happy.
  const Model model = make_milp({{2.0}}, {3.0}, {3.0}, {1.0}, {10.0}, {true});
  const Solution s = solve(model, mip_options());
  EXPECT_EQ(s.status, SolveStatus::kInfeasible);
}

TEST(BranchAndBound, AnIntegralRelaxationClosesAtTheRoot) {
  // min -x s.t. x <= 4, x integer. The relaxation is already integral, so no branching is
  // needed and the bound equals the objective immediately.
  const Model model = make_milp({{1.0}}, {-kInfinity}, {4.0}, {-1.0}, {10.0}, {true});
  const Solution s = solve(model, mip_options());
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, -4.0, 1e-9);
  EXPECT_NEAR(s.dual_bound, s.objective, 1e-9);
  EXPECT_LE(s.nodes, 2);
}

TEST(BranchAndBound, ReportsFeasibleRatherThanOptimalAtTheNodeLimit) {
  // A limit stops the PROOF, not the search's ability to have found something. The status
  // must distinguish the two, or a caller cannot tell a proven answer from a good guess.
  //
  // Capacity 10, not 9: at 9 the relaxation is already integral (a and b exactly fill it),
  // the root closes immediately and no limit is ever reached. At 10 the relaxation is
  // fractional, so the search really does have to branch and really can be interrupted.
  const Model model =
      make_milp({{5.0, 4.0, 3.0, 2.0}}, {-kInfinity}, {10.0}, {-10.0, -7.0, -4.0, -3.0},
                {1.0, 1.0, 1.0, 1.0}, {true, true, true, true});
  Options options = mip_options();
  options.set_int("node_limit", 1);
  const Solution s = solve(model, options);
  EXPECT_NE(s.status, SolveStatus::kOptimal);
  if (s.status == SolveStatus::kFeasible) {
    // The bound must remain a genuine bound: at least as good as the incumbent.
    EXPECT_LE(s.dual_bound, s.objective + 1e-9);
  }
}

TEST(BranchAndBound, MaximisationIsReportedInTheOriginalSense) {
  Model model =
      make_milp({{5.0, 4.0}}, {-kInfinity}, {9.0}, {10.0, 7.0}, {1.0, 1.0}, {true, true});
  model.sense = ObjSense::kMaximize;
  const Solution s = solve(model, mip_options());
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, 17.0, 1e-9);
}

TEST(BranchAndBound, CarriesTheObjectiveOffset) {
  Model model = make_milp({{2.0}}, {-kInfinity}, {5.0}, {-5.0}, {10.0}, {true});
  model.objective_offset = 100.0;
  const Solution s = solve(model, mip_options());
  EXPECT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_NEAR(s.objective, 90.0, 1e-9);  // -10 + 100
}

// =========================================================================================
// The gate
// =========================================================================================

TEST(BranchAndBound, FuzzAgainstTheExactMilpOracle) {
  std::mt19937_64 rng(20260906);
  oracle::GeneratorConfig config;
  // Small: the oracle explores the tree in exact arithmetic and copies both bound vectors
  // per node, so it is deliberately slow.
  config.min_rows = 2;
  config.max_rows = 5;
  config.min_cols = 2;
  config.max_cols = 5;
  config.magnitude = 4;

  int agreed_optimal = 0;
  int agreed_infeasible = 0;
  int oracle_abstained = 0;
  int mismatched = 0;
  std::vector<std::string> failures;

  for (int trial = 0; trial < 600; ++trial) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    // Every column integral, and bounded, so the tree is finite.
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 1);
    for (Index j = 0; j < lp.num_cols; ++j) {
      if (lp.upper[static_cast<std::size_t>(j)] == oracle::kNoUpperBound) {
        lp.upper[static_cast<std::size_t>(j)] = 6;
      }
    }

    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    if (exact.status == oracle::OracleStatus::kOverflow ||
        exact.status == oracle::OracleStatus::kIterationLimit ||
        exact.status == oracle::OracleStatus::kUnbounded) {
      ++oracle_abstained;
      continue;
    }

    Model model = oracle::to_model(lp);
    for (Index j = 0; j < model.num_cols(); ++j) {
      model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
      model.col_upper[static_cast<std::size_t>(j)] =
          static_cast<double>(lp.upper[static_cast<std::size_t>(j)]);
    }
    const Solution s = solve(model, mip_options());

    const auto disagree = [&](const std::string& why) {
      ++mismatched;
      if (failures.size() < 4) {
        failures.push_back(why + "\n  oracle: " + oracle::to_string(exact.status) +
                           (exact.status == oracle::OracleStatus::kOptimal
                                ? "  objective " + std::to_string(exact.objective.to_double())
                                : "") +
                           "\n  solver: " + std::string(to_string(s.status)) +
                           (s.status == SolveStatus::kOptimal
                                ? "  objective " + std::to_string(s.objective)
                                : "") +
                           "\n" + lp.to_text());
      }
    };

    if (exact.status == oracle::OracleStatus::kInfeasible) {
      if (s.status != SolveStatus::kInfeasible) {
        disagree("the instance has no integer feasible point");
      } else {
        ++agreed_infeasible;
      }
      continue;
    }

    if (s.status != SolveStatus::kOptimal) {
      disagree("an integer optimum exists but the search did not prove one");
      continue;
    }
    const double expected = exact.objective.to_double();
    const double scale = std::max(1.0, std::fabs(expected));
    if (std::fabs(s.objective - expected) > 1e-6 * scale) {
      disagree("objectives differ by " + std::to_string(std::fabs(s.objective - expected)));
      continue;
    }
    ++agreed_optimal;
  }

  std::cout << "\n=== MILP fuzz against the exact oracle ===\n"
            << "  agreed optimal      " << agreed_optimal << "\n"
            << "  agreed infeasible   " << agreed_infeasible << "\n"
            << "  oracle abstained    " << oracle_abstained << "\n"
            << "  MISMATCHED          " << mismatched << "\n";
  for (const std::string& failure : failures) {
    std::cout << "\n--- failing instance ---\n" << failure << "\n";
  }

  EXPECT_EQ(mismatched, 0);
  EXPECT_GT(agreed_optimal + agreed_infeasible, 300)
      << "too few instances were actually compared for this to mean anything";
  EXPECT_GT(agreed_optimal, 50) << "the generator produced almost no feasible MILPs";
}

}  // namespace
}  // namespace sankhya

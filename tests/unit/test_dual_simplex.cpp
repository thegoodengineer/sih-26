// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dual simplex tests (issue #65).
//
// A second simplex method must give the SAME answers as the first, and that is the property
// pinned here: on every model where both terminate, the dual and the primal agree on the
// status and, at an optimum, on the objective; and the dual's own certificate passes the same
// KKT measurement the dispatcher applies. What is deliberately not asserted is that the dual
// takes fewer iterations - it usually does with a warm start, and that is measured on
// MIPLIB rather than promised on random models.
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "simplex/primal_simplex.hpp"

namespace sankhya {
namespace {

Model make_model(ObjSense sense, const std::vector<double>& cost,
                 const std::vector<double>& col_lower, const std::vector<double>& col_upper,
                 const std::vector<std::vector<double>>& rows,
                 const std::vector<double>& row_lower, const std::vector<double>& row_upper) {
  Model model;
  model.name = "dual-test";
  model.sense = sense;
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(cost.size(), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
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
  EXPECT_TRUE(model.validate().empty()) << model.validate();
  return model;
}

Options engine_options(const char* algorithm) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", false);
  options.set_string("algorithm", algorithm);
  return options;
}

Solution run_dual(const Model& model) {
  return solve(model, engine_options("dual-simplex"));
}
Solution run_primal(const Model& model) {
  return solve(model, engine_options("simplex"));
}

/// The dispatcher's own measurement of the point and its certificate, in original units.
void expect_certified_optimal(const Solution& solution) {
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_LE(solution.primal_infeasibility_scaled, tol::kPrimalFeasibility) << solution.message;
  EXPECT_LE(solution.dual_infeasibility_scaled, tol::kDualFeasibility) << solution.message;
  EXPECT_NEAR(solution.absolute_gap, 0.0, 1e-7);
}

void expect_agrees_with_primal(const Model& model) {
  const Solution dual = run_dual(model);
  const Solution primal = run_primal(model);
  ASSERT_EQ(dual.status, primal.status)
      << "dual: " << dual.message << " / primal: " << primal.message;
  if (primal.status != SolveStatus::kOptimal) return;
  expect_certified_optimal(dual);
  EXPECT_NEAR(dual.objective, primal.objective,
              1e-7 * std::max(1.0, std::fabs(primal.objective)));
}

constexpr double kInf = std::numeric_limits<double>::infinity();

// =========================================================================================
// The dual on the same models the primal is tested on
// =========================================================================================

TEST(DualSimplex, SolvesATwoVariableMaximization) {
  //   max 3x + 2y  s.t.  x + y <= 4,  x + 3y <= 6,  x, y >= 0   ->  x = 4, y = 0, 12
  const Model model = make_model(ObjSense::kMaximize, {3.0, 2.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, 1.0}, {1.0, 3.0}}, {-kInf, -kInf}, {4.0, 6.0});
  const Solution solution = run_dual(model);
  expect_certified_optimal(solution);
  EXPECT_NEAR(solution.objective, 12.0, 1e-9);
  EXPECT_NEAR(solution.col_value[0], 4.0, 1e-9);
  EXPECT_NEAR(solution.col_value[1], 0.0, 1e-9);
  EXPECT_EQ(solution.algorithm.rfind("simplex-dual", 0), 0u) << solution.algorithm;
}

TEST(DualSimplex, StartsDualFeasibleOnACoveringProblem) {
  // min x + y s.t. x + 2y >= 2, 3x + y >= 3: costs positive and rows >=, the textbook case
  // where the slack basis is dual feasible and primal infeasible - the dual's home ground.
  const Model model = make_model(ObjSense::kMinimize, {1.0, 1.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, 2.0}, {3.0, 1.0}}, {2.0, 3.0}, {kInf, kInf});
  const Solution solution = run_dual(model);
  expect_certified_optimal(solution);
  EXPECT_NEAR(solution.objective, 1.4, 1e-9);
  expect_agrees_with_primal(model);
}

TEST(DualSimplex, BoxesAColumnWithNoBoundToFlipTo) {
  // min -x s.t. x <= 5 (as a row), x >= 0. At the slack basis d_x = -1 at the lower bound
  // with no upper bound: the dual must invent one, solve, and end with x basic at 5.
  const Model model =
      make_model(ObjSense::kMinimize, {-1.0}, {0.0}, {kInf}, {{1.0}}, {-kInf}, {5.0});
  const Solution solution = run_dual(model);
  expect_certified_optimal(solution);
  EXPECT_NEAR(solution.objective, -5.0, 1e-9);
  EXPECT_NEAR(solution.col_value[0], 5.0, 1e-9);
}

TEST(DualSimplex, HandsOverToThePrimalWhenAnArtificialBoundBinds) {
  // max x + y s.t. x - y <= 1, x, y >= 0: unbounded. The dual boxes both columns, finds the
  // boxed optimum with a box active, hands the basis to the primal, and the primal proves
  // the ray. The answer must be about the caller's model, never the boxed one.
  const Model model = make_model(ObjSense::kMaximize, {1.0, 1.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, -1.0}}, {-kInf}, {1.0});
  const Solution solution = run_dual(model);
  EXPECT_EQ(solution.status, SolveStatus::kUnbounded) << solution.message;
  expect_agrees_with_primal(model);
}

TEST(DualSimplex, ProvesInfeasibilityByADualRay) {
  // x + y >= 4 and x + y <= 2 with x, y >= 0: no point. The dual's ratio test finds no
  // candidate on fresh factors, which is the proof.
  const Model model = make_model(ObjSense::kMinimize, {1.0, 1.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, 1.0}, {1.0, 1.0}}, {4.0, -kInf}, {kInf, 2.0});
  const Solution solution = run_dual(model);
  EXPECT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
}

TEST(DualSimplex, FlipsBoundsInsteadOfPivoting) {
  // Every column boxed in [0, 1] with a positive cost, one >= row: the long-step ratio
  // test flips columns to their upper bound until the row is satisfied.
  //   min sum j*x_j  s.t.  sum x_j >= 3.5,  0 <= x_j <= 1, j = 1..6
  // Optimum: x1 = x2 = x3 = 1, x4 = 0.5, objective 1 + 2 + 3 + 2 = 8.
  const Model model = make_model(ObjSense::kMinimize, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                 std::vector<double>(6, 0.0), std::vector<double>(6, 1.0),
                                 {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}}, {3.5}, {kInf});
  const Solution solution = run_dual(model);
  expect_certified_optimal(solution);
  EXPECT_NEAR(solution.objective, 8.0, 1e-9);
  EXPECT_LE(solution.iterations, 3) << "flips should do most of the work";
}

TEST(DualSimplex, HandlesFreeColumnsRangeRowsAndEqualities) {
  // min x - 2y + z  s.t.  1 <= x + y + z <= 3,  x - y = 0,  x free, 0 <= y <= 2, z >= -1.
  const Model model =
      make_model(ObjSense::kMinimize, {1.0, -2.0, 1.0}, {-kInf, 0.0, -1.0}, {kInf, 2.0, kInf},
                 {{1.0, 1.0, 1.0}, {1.0, -1.0, 0.0}}, {1.0, 0.0}, {3.0, 0.0});
  expect_agrees_with_primal(model);
}

TEST(DualSimplex, AppliesTheObjectiveOffsetAndSense) {
  Model model = make_model(ObjSense::kMaximize, {2.0, 1.0}, {0.0, 0.0}, {3.0, 4.0},
                           {{1.0, 1.0}}, {-kInf}, {5.0});
  model.objective_offset = 10.0;
  const Solution solution = run_dual(model);
  expect_certified_optimal(solution);
  EXPECT_NEAR(solution.objective, 10.0 + 2.0 * 3.0 + 1.0 * 2.0, 1e-9);
}

// =========================================================================================
// Fuzz: the two methods must agree
// =========================================================================================

TEST(DualSimplex, AgreesWithThePrimalOnRandomInstances) {
  // Random bounded and one-sided LPs, feasible by construction around a known point and
  // occasionally unbounded or infeasible by accident - which is fine, since the assertion
  // is agreement on the status, and on the objective when both are optimal.
  std::mt19937 rng(65);
  std::uniform_real_distribution<double> coefficient(-4.0, 4.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_int_distribution<int> dimension(2, 9);

  int optimal = 0;
  int other = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const auto m = static_cast<std::size_t>(dimension(rng));
    const auto n = static_cast<std::size_t>(dimension(rng));
    std::vector<double> point(n);
    std::vector<double> cost(n);
    std::vector<double> lower(n);
    std::vector<double> upper(n);
    for (std::size_t j = 0; j < n; ++j) {
      point[j] = 4.0 * unit(rng);
      cost[j] = coefficient(rng);
      const double roll = unit(rng);
      if (roll < 0.5) {
        lower[j] = 0.0;
        upper[j] = 4.0 + 4.0 * unit(rng);
      } else if (roll < 0.8) {
        lower[j] = 0.0;
        upper[j] = kInf;
      } else if (roll < 0.9) {
        lower[j] = -kInf;
        upper[j] = kInf;
      } else {
        lower[j] = point[j];
        upper[j] = point[j];
      }
    }
    std::vector<std::vector<double>> rows(m, std::vector<double>(n, 0.0));
    std::vector<double> row_lower(m);
    std::vector<double> row_upper(m);
    for (std::size_t i = 0; i < m; ++i) {
      double activity = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        if (unit(rng) < 0.6) rows[i][j] = coefficient(rng);
        activity += rows[i][j] * point[j];
      }
      const double roll = unit(rng);
      if (roll < 0.4) {
        row_lower[i] = -kInf;
        row_upper[i] = activity + 2.0 * unit(rng);
      } else if (roll < 0.7) {
        row_lower[i] = activity - 2.0 * unit(rng);
        row_upper[i] = kInf;
      } else if (roll < 0.85) {
        row_lower[i] = activity;
        row_upper[i] = activity;
      } else {
        row_lower[i] = activity - unit(rng);
        row_upper[i] = activity + unit(rng);
      }
    }
    const Model model = make_model(unit(rng) < 0.5 ? ObjSense::kMinimize : ObjSense::kMaximize,
                                   cost, lower, upper, rows, row_lower, row_upper);
    if (!model.validate().empty()) continue;

    const Solution dual = run_dual(model);
    const Solution primal = run_primal(model);
    ASSERT_EQ(dual.status, primal.status)
        << "trial " << trial << ": dual " << dual.message << " / primal " << primal.message;
    if (primal.status == SolveStatus::kOptimal) {
      ++optimal;
      EXPECT_LE(dual.primal_infeasibility_scaled, tol::kPrimalFeasibility) << "trial " << trial;
      EXPECT_LE(dual.dual_infeasibility_scaled, tol::kDualFeasibility) << "trial " << trial;
      EXPECT_NEAR(dual.objective, primal.objective,
                  1e-6 * std::max(1.0, std::fabs(primal.objective)))
          << "trial " << trial;
    } else {
      ++other;
    }
  }
  EXPECT_GT(optimal, 150) << "too few optimal instances to prove anything (" << other
                          << " ended otherwise)";
}

// =========================================================================================
// Warm starts
// =========================================================================================

TEST(DualSimplex, WarmStartFromATightenedParentTakesFewPivots) {
  // A 0/1-flavoured relaxation, solved cold; then one column's bound is tightened the way
  // branching would and the dual is warm-started from the parent's basis. Same answer as a
  // cold solve, and the pivot count is what a node solve should cost, not a re-solve.
  std::mt19937 rng(651);
  std::uniform_real_distribution<double> coefficient(1.0, 9.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const std::size_t m = 12;
  const std::size_t n = 30;
  std::vector<double> cost(n);
  std::vector<std::vector<double>> rows(m, std::vector<double>(n, 0.0));
  std::vector<double> row_lower(m, -kInf);
  std::vector<double> row_upper(m);
  for (std::size_t j = 0; j < n; ++j) cost[j] = -coefficient(rng);
  for (std::size_t i = 0; i < m; ++i) {
    double total = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      if (unit(rng) < 0.5) rows[i][j] = coefficient(rng);
      total += rows[i][j];
    }
    row_upper[i] = 0.4 * total;
  }
  Model model = make_model(ObjSense::kMinimize, cost, std::vector<double>(n, 0.0),
                           std::vector<double>(n, 1.0), rows, row_lower, row_upper);

  const Solution parent = run_primal(model);
  ASSERT_EQ(parent.status, SolveStatus::kOptimal) << parent.message;

  // Branch on the first fractional column, downward.
  Index branch = -1;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const double v = parent.col_value[static_cast<std::size_t>(j)];
    if (std::fabs(v - std::round(v)) > 1e-6) {
      branch = j;
      break;
    }
  }
  ASSERT_GE(branch, 0) << "the relaxation of this instance is expected to be fractional";
  model.col_upper[static_cast<std::size_t>(branch)] =
      std::floor(parent.col_value[static_cast<std::size_t>(branch)]);

  WarmStart warm;
  warm.col_status = parent.col_status;
  warm.row_status = parent.row_status;
  Options options = engine_options("dual-simplex");
  Logger logger(nullptr);
  const Solution child_warm = solve_dual_simplex(model, options, logger, &warm);
  const Solution child_cold = run_primal(model);
  ASSERT_EQ(child_warm.status, SolveStatus::kOptimal) << child_warm.message;
  ASSERT_EQ(child_cold.status, SolveStatus::kOptimal) << child_cold.message;
  EXPECT_NEAR(child_warm.objective, child_cold.objective, 1e-7);
  EXPECT_LE(child_warm.primal_infeasibility_scaled, tol::kPrimalFeasibility);
  EXPECT_LE(child_warm.dual_infeasibility_scaled, tol::kDualFeasibility);
  EXPECT_LT(child_warm.iterations, child_cold.iterations)
      << "warm " << child_warm.iterations << " vs cold " << child_cold.iterations;
  EXPECT_LE(child_warm.iterations, 6) << child_warm.iterations;
}

TEST(DualSimplex, AWarmStartThatIsNotABasisFallsBackToTheSlackBasis) {
  const Model model = make_model(ObjSense::kMinimize, {1.0, 1.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, 2.0}, {3.0, 1.0}}, {2.0, 3.0}, {kInf, kInf});
  WarmStart bogus;
  bogus.col_status = {BasisStatus::kBasic, BasisStatus::kBasic};
  bogus.row_status = {BasisStatus::kBasic, BasisStatus::kBasic};  // four basic, two rows
  Options options = engine_options("dual-simplex");
  Logger logger(nullptr);
  const Solution solution = solve_dual_simplex(model, options, logger, &bogus);
  expect_certified_optimal(solution);
  EXPECT_NEAR(solution.objective, 1.4, 1e-9);
}

}  // namespace
}  // namespace sankhya

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - primal simplex tests.
//
// A simplex that returns the wrong number does not crash, does not warn, and prints a
// beautifully formatted answer. So almost nothing here asserts on the objective alone.
// Every "optimal" claim is put through expect_kkt_optimal(), which re-derives the reduced
// costs from the duals INDEPENDENTLY of the solver, checks the sign conditions, checks
// complementary slackness, and checks strong duality by reconstructing the dual objective
// from the active bounds and comparing it against c'x. That identity holds only at a true
// optimum, so a simplex that stops one pivot early fails it even though its point is
// perfectly feasible and its objective looks reasonable.
//
// This is the stand-in for the exact rational oracle until Phase 3 builds the real one.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

/// Build a model from a dense row-major matrix. Entries that are exactly zero are dropped,
/// which is what a reader would do.
Model make_model(ObjSense sense, const std::vector<double>& cost,
                 const std::vector<double>& col_lower, const std::vector<double>& col_upper,
                 const std::vector<std::vector<double>>& rows,
                 const std::vector<double>& row_lower, const std::vector<double>& row_upper) {
  Model model;
  model.name = "test";
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

  const std::string problem = model.validate();
  EXPECT_TRUE(problem.empty()) << problem;
  return model;
}

Solution run(const Model& model) {
  Options options;
  options.set_bool("log_to_console", false);
  return solve(model, options);
}

/// Independent optimality certificate. Nothing here reads a quantity the simplex computed
/// except the primal values and the row duals; everything else is rebuilt from the model.
void expect_kkt_optimal(const Model& model, const Solution& solution, double tolerance = 1e-7) {
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;

  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const double sense = model.sense_multiplier();

  // ---- primal feasibility, recomputed from the matrix -----------------------------------
  std::vector<double> activity(static_cast<std::size_t>(m), 0.0);
  if (m > 0) model.matrix.multiply(solution.col_value.data(), activity.data());
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (is_finite_bound(model.row_lower[u])) {
      EXPECT_GE(activity[u], model.row_lower[u] - tolerance) << "row " << i;
    }
    if (is_finite_bound(model.row_upper[u])) {
      EXPECT_LE(activity[u], model.row_upper[u] + tolerance) << "row " << i;
    }
  }
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (is_finite_bound(model.col_lower[u])) {
      EXPECT_GE(solution.col_value[u], model.col_lower[u] - tolerance) << "column " << j;
    }
    if (is_finite_bound(model.col_upper[u])) {
      EXPECT_LE(solution.col_value[u], model.col_upper[u] + tolerance) << "column " << j;
    }
  }

  // ---- reduced costs, rebuilt as d = c - A' y -------------------------------------------
  std::vector<double> y(static_cast<std::size_t>(m), 0.0);
  for (Index i = 0; i < m; ++i) {
    y[static_cast<std::size_t>(i)] = sense * solution.row_dual[static_cast<std::size_t>(i)];
  }
  std::vector<double> d(static_cast<std::size_t>(n), 0.0);
  for (Index j = 0; j < n; ++j) {
    d[static_cast<std::size_t>(j)] = sense * model.col_cost[static_cast<std::size_t>(j)];
  }
  if (m > 0) model.matrix.transpose_multiply_add(y.data(), d.data(), -1.0);

  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    EXPECT_NEAR(d[u], sense * solution.col_dual[u], 1e-6)
        << "the reported reduced cost of column " << j << " is not c - A'y";
  }

  // ---- dual feasibility and complementary slackness --------------------------------------
  const auto at_bound = [&](double value, double bound) {
    return is_finite_bound(bound) && std::fabs(value - bound) <= 1e-7;
  };

  double dual_objective = 0.0;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double x = solution.col_value[u];
    const bool lo = at_bound(x, model.col_lower[u]);
    const bool hi = at_bound(x, model.col_upper[u]);
    if (lo && hi) {
      dual_objective += d[u] * model.col_lower[u];  // fixed column: either bound serves
      continue;
    }
    if (lo) {
      EXPECT_GE(d[u], -tolerance) << "column " << j << " at its lower bound needs d >= 0";
      dual_objective += d[u] * model.col_lower[u];
    } else if (hi) {
      EXPECT_LE(d[u], tolerance) << "column " << j << " at its upper bound needs d <= 0";
      dual_objective += d[u] * model.col_upper[u];
    } else {
      EXPECT_NEAR(d[u], 0.0, tolerance)
          << "column " << j << " is strictly between its bounds, so d must vanish";
    }
  }

  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double a = activity[u];
    const bool lo = at_bound(a, model.row_lower[u]);
    const bool hi = at_bound(a, model.row_upper[u]);
    if (lo && hi) {
      dual_objective += y[u] * model.row_lower[u];
      continue;
    }
    if (lo) {
      EXPECT_GE(y[u], -tolerance) << "row " << i << " active at its lower bound needs y >= 0";
      dual_objective += y[u] * model.row_lower[u];
    } else if (hi) {
      EXPECT_LE(y[u], tolerance) << "row " << i << " active at its upper bound needs y <= 0";
      dual_objective += y[u] * model.row_upper[u];
    } else {
      EXPECT_NEAR(y[u], 0.0, tolerance)
          << "row " << i << " is inactive, so its dual must vanish";
    }
  }

  // ---- strong duality --------------------------------------------------------------------
  // c'x = y'(Ax) + d'x, and every term of that sum collapses onto an active bound at an
  // optimum. Reconstructing the right-hand side from the bounds alone and matching it
  // against c'x is the check a merely-feasible point cannot pass.
  double primal_objective = 0.0;
  for (Index j = 0; j < n; ++j) {
    primal_objective += sense * model.col_cost[static_cast<std::size_t>(j)] *
                        solution.col_value[static_cast<std::size_t>(j)];
  }
  const double scale = std::max({1.0, std::fabs(primal_objective), std::fabs(dual_objective)});
  EXPECT_NEAR(primal_objective / scale, dual_objective / scale, 1e-7)
      << "strong duality fails: primal " << primal_objective << " vs dual " << dual_objective;
}

constexpr double kInf = kInfinity;

// =========================================================================================
// Hand-checkable problems
// =========================================================================================

TEST(PrimalSimplex, SolvesATwoVariableMaximization) {
  //   max 3 x + 5 y
  //   s.t.  x        <= 4
  //              2 y <= 12
  //        3 x + 2 y <= 18
  // Classic textbook problem; the optimum is (2, 6) with objective 36.
  const Model model = make_model(ObjSense::kMaximize, {3.0, 5.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, {-kInf, -kInf, -kInf},
                                 {4.0, 12.0, 18.0});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 36.0, 1e-9);
  EXPECT_NEAR(solution.col_value[0], 2.0, 1e-9);
  EXPECT_NEAR(solution.col_value[1], 6.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, SolvesAMinimizationWithGreaterThanRows) {
  //   min 2 x + 3 y
  //   s.t. x + y >= 10
  //        x      >= 2
  // Optimum: x = 10, y = 0, objective 20.
  const Model model = make_model(ObjSense::kMinimize, {2.0, 3.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, 1.0}, {1.0, 0.0}}, {10.0, 2.0}, {kInf, kInf});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 20.0, 1e-9);
  EXPECT_NEAR(solution.col_value[0], 10.0, 1e-9);
  EXPECT_NEAR(solution.col_value[1], 0.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, HandlesEqualityRows) {
  //   min x + y  s.t.  x + y = 5,  x - y = 1   ->  x = 3, y = 2, objective 5
  const Model model = make_model(ObjSense::kMinimize, {1.0, 1.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, 1.0}, {1.0, -1.0}}, {5.0, 1.0}, {5.0, 1.0});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.col_value[0], 3.0, 1e-9);
  EXPECT_NEAR(solution.col_value[1], 2.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, RespectsFiniteUpperBoundsOnColumns) {
  //   max x + y  s.t. x + y <= 10, 0 <= x <= 3, 0 <= y <= 4
  // The row is not the binding constraint; the column bounds are. A solver that treats
  // upper bounds as ordinary rows would still get this, but one that ignores them reports 10.
  const Model model = make_model(ObjSense::kMaximize, {1.0, 1.0}, {0.0, 0.0}, {3.0, 4.0},
                                 {{1.0, 1.0}}, {-kInf}, {10.0});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 7.0, 1e-9);
  EXPECT_NEAR(solution.col_value[0], 3.0, 1e-9);
  EXPECT_NEAR(solution.col_value[1], 4.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, HandlesNegativeLowerBounds) {
  //   min x  s.t.  x >= -7,  -10 <= x <= 10   ->  x = -7
  const Model model =
      make_model(ObjSense::kMinimize, {1.0}, {-10.0}, {10.0}, {{1.0}}, {-7.0}, {kInf});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.col_value[0], -7.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, HandlesFreeColumns) {
  //   min x + y  s.t. x + y = 3, x free, y free  ->  objective 3 regardless of the split
  const Model model = make_model(ObjSense::kMinimize, {1.0, 1.0}, {-kInf, -kInf}, {kInf, kInf},
                                 {{1.0, 1.0}}, {3.0}, {3.0});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 3.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, HandlesRangeRows) {
  //   min x  s.t. 2 <= x + y <= 6, y <= 1, x, y >= 0   ->  x = 1, y = 1
  const Model model = make_model(ObjSense::kMinimize, {1.0, 0.0}, {0.0, 0.0}, {kInf, 1.0},
                                 {{1.0, 1.0}}, {2.0}, {6.0});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 1.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, AppliesTheObjectiveOffset) {
  Model model = make_model(ObjSense::kMinimize, {1.0}, {0.0}, {kInf}, {{1.0}}, {4.0}, {kInf});
  model.objective_offset = 100.0;
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.col_value[0], 4.0, 1e-9);
  EXPECT_NEAR(solution.objective, 104.0, 1e-9);
}

TEST(PrimalSimplex, ReportsShadowPricesWithTheDocumentedSign) {
  //   min 2 x  s.t.  x >= 5.   Raising the requirement by one unit costs 2, so the dual on
  //   a row active at its LOWER bound is +2 for a minimization problem.
  const Model model =
      make_model(ObjSense::kMinimize, {2.0}, {0.0}, {kInf}, {{1.0}}, {5.0}, {kInf});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.row_dual[0], 2.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, ShadowPriceSignFlipsOnAnUpperBoundRow) {
  //   max 3 x  s.t.  x <= 5.  Relaxing the cap by one unit gains 3, and the row is active
  //   at its UPPER bound. Reported in the sense of the original (maximization) file.
  const Model model =
      make_model(ObjSense::kMaximize, {3.0}, {0.0}, {kInf}, {{1.0}}, {-kInf}, {5.0});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 15.0, 1e-9);
  EXPECT_NEAR(solution.row_dual[0], 3.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, SolvesASmallTransportationProblem) {
  // Two plants, three depots. Supplies 20 and 30; demands 10, 25, 15. Costs chosen so the
  // optimum is unique: plant 1 serves depot 1 and 2, plant 2 serves the rest.
  //   variables x11 x12 x13 x21 x22 x23
  const Model model =
      make_model(ObjSense::kMinimize, {4.0, 6.0, 9.0, 5.0, 3.0, 8.0},
                 {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {kInf, kInf, kInf, kInf, kInf, kInf},
                 {{1.0, 1.0, 1.0, 0.0, 0.0, 0.0},   // supply 1
                  {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},   // supply 2
                  {1.0, 0.0, 0.0, 1.0, 0.0, 0.0},   // demand 1
                  {0.0, 1.0, 0.0, 0.0, 1.0, 0.0},   // demand 2
                  {0.0, 0.0, 1.0, 0.0, 0.0, 1.0}},  // demand 3
                 {-kInf, -kInf, 10.0, 25.0, 15.0}, {20.0, 30.0, 10.0, 25.0, 15.0});

  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  expect_kkt_optimal(model, solution);

  double shipped = 0.0;
  for (double v : solution.col_value) shipped += v;
  EXPECT_NEAR(shipped, 50.0, 1e-7) << "total shipment must equal total demand";
  EXPECT_LE(solution.objective, 315.0);
}

// =========================================================================================
// Terminal states other than optimal
// =========================================================================================

TEST(PrimalSimplex, DetectsAnInfeasibleModel) {
  //   x >= 5 and x <= 2 simultaneously.
  const Model model = make_model(ObjSense::kMinimize, {1.0}, {0.0}, {kInf}, {{1.0}, {1.0}},
                                 {5.0, -kInf}, {kInf, 2.0});
  const Solution solution = run(model);
  EXPECT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
}

TEST(PrimalSimplex, DetectsAnInfeasibleEqualitySystem) {
  //   x + y = 1 and x + y = 2.
  const Model model = make_model(ObjSense::kMinimize, {1.0, 1.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, 1.0}, {1.0, 1.0}}, {1.0, 2.0}, {1.0, 2.0});
  const Solution solution = run(model);
  EXPECT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
}

TEST(PrimalSimplex, DetectsAnUnboundedModel) {
  //   min -x  s.t.  x >= 0, no upper bound anywhere.
  const Model model =
      make_model(ObjSense::kMinimize, {-1.0}, {0.0}, {kInf}, {{1.0}}, {-kInf}, {kInf});
  const Solution solution = run(model);
  EXPECT_EQ(solution.status, SolveStatus::kUnbounded) << solution.message;
}

TEST(PrimalSimplex, AnEmptyFeasibleRegionIsInfeasibleNotUnbounded) {
  // Unboundedness is only meaningful over a nonempty region. Phase 1 must run first and
  // report infeasibility even though the objective direction is unbounded.
  const Model model = make_model(ObjSense::kMinimize, {-1.0}, {0.0}, {kInf}, {{1.0}, {1.0}},
                                 {5.0, -kInf}, {kInf, 2.0});
  const Solution solution = run(model);
  EXPECT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
}

TEST(PrimalSimplex, HandlesAModelWithNoRows) {
  //   min -x  s.t. 0 <= x <= 7. No constraints at all: the basis is 0 x 0.
  const Model model = make_model(ObjSense::kMinimize, {-1.0}, {0.0}, {7.0}, {}, {}, {});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.col_value[0], 7.0, 1e-12);
  EXPECT_NEAR(solution.objective, -7.0, 1e-12);
}

TEST(PrimalSimplex, HandlesFixedColumns) {
  //   min x + y  s.t. x + y >= 1, x fixed at 3.
  const Model model = make_model(ObjSense::kMinimize, {1.0, 1.0}, {3.0, 0.0}, {3.0, kInf},
                                 {{1.0, 1.0}}, {1.0}, {kInf});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.col_value[0], 3.0, 1e-9);
  EXPECT_NEAR(solution.col_value[1], 0.0, 1e-9);
}

TEST(PrimalSimplex, StopsAtTheIterationLimit) {
  const Model model = make_model(ObjSense::kMaximize, {3.0, 5.0}, {0.0, 0.0}, {kInf, kInf},
                                 {{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, {-kInf, -kInf, -kInf},
                                 {4.0, 12.0, 18.0});
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("iteration_limit", 1);
  const Solution solution = solve(model, options);
  EXPECT_EQ(solution.status, SolveStatus::kIterationLimit);
  EXPECT_EQ(solution.iterations, 1);
}

// =========================================================================================
// Degeneracy
// =========================================================================================

TEST(PrimalSimplex, SolvesADegenerateVertex) {
  // Three rows meet at (0, 0) in two variables, so the optimal vertex is degenerate and the
  // simplex must pass through zero-length steps without cycling.
  const Model model =
      make_model(ObjSense::kMinimize, {1.0, 1.0}, {0.0, 0.0}, {kInf, kInf},
                 {{1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}}, {-kInf, -kInf, -kInf}, {0.0, 0.0, 0.0});
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, 0.0, 1e-9);
  expect_kkt_optimal(model, solution);
}

TEST(PrimalSimplex, SolvesAHighlyDegenerateAssignmentLikeProblem) {
  // Every equality row forces a sum to 1 and the cost matrix has many ties, so most pivots
  // are degenerate. This is the smallest shape that reliably exercises the Bland fallback.
  constexpr Index k = 4;
  std::vector<double> cost;
  for (Index i = 0; i < k; ++i) {
    for (Index j = 0; j < k; ++j) {
      cost.push_back(static_cast<double>((i * j) % 3) + 1.0);
    }
  }
  const auto n = static_cast<std::size_t>(k * k);
  std::vector<std::vector<double>> rows;
  std::vector<double> bounds;
  for (Index i = 0; i < k; ++i) {
    std::vector<double> row(n, 0.0);
    for (Index j = 0; j < k; ++j) row[static_cast<std::size_t>(i * k + j)] = 1.0;
    rows.push_back(row);
    bounds.push_back(1.0);
  }
  for (Index j = 0; j < k; ++j) {
    std::vector<double> row(n, 0.0);
    for (Index i = 0; i < k; ++i) row[static_cast<std::size_t>(i * k + j)] = 1.0;
    rows.push_back(row);
    bounds.push_back(1.0);
  }

  const Model model = make_model(ObjSense::kMinimize, cost, std::vector<double>(n, 0.0),
                                 std::vector<double>(n, 1.0), rows, bounds, bounds);
  const Solution solution = run(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  expect_kkt_optimal(model, solution);
  EXPECT_NEAR(solution.objective, 4.0, 1e-7);
}

// =========================================================================================
// Randomised certification
// =========================================================================================

TEST(PrimalSimplex, FuzzAgainstTheKktCertificate) {
  // Every instance is generated AROUND A KNOWN FEASIBLE POINT: draw x0 inside the column
  // box, compute a = A x0, and place the row bounds so that a satisfies them. The feasible
  // region is therefore never empty by construction, which turns a kInfeasible answer from
  // an outcome to be counted into a hard failure - the strongest single assertion in this
  // file, because a phase 1 that gives up early is otherwise indistinguishable from a model
  // that really has no solution.
  //
  // Most columns get finite boxes so the LP is also bounded and the run ends at a vertex we
  // can certify. A minority are left free or half-open so that the nonbasic-free and
  // nonbasic-at-upper paths are still exercised; those may legitimately be unbounded.
  std::mt19937 rng(26119);
  std::uniform_real_distribution<double> coefficient(-4.0, 4.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  int optimal = 0;
  int unbounded = 0;
  int wrongly_infeasible = 0;
  int other = 0;

  for (int trial = 0; trial < 400; ++trial) {
    const Index n = 2 + static_cast<Index>(trial % 8);
    const Index m = 1 + static_cast<Index>(trial % 6);
    const auto un = static_cast<std::size_t>(n);
    const auto um = static_cast<std::size_t>(m);

    std::vector<double> cost(un);
    for (double& c : cost) c = coefficient(rng);

    std::vector<double> col_lower(un, 0.0);
    std::vector<double> col_upper(un, 0.0);
    std::vector<double> x0(un, 0.0);
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      const double centre = coefficient(rng);
      const double half_width = 1.0 + 4.0 * unit(rng);
      const double roll = unit(rng);
      if (roll < 0.10) {
        col_lower[u] = -kInf;
        col_upper[u] = kInf;
      } else if (roll < 0.20) {
        col_lower[u] = centre - half_width;
        col_upper[u] = kInf;
      } else if (roll < 0.25) {
        col_lower[u] = centre - half_width;
        col_upper[u] = centre - half_width;  // fixed column
      } else {
        col_lower[u] = centre - half_width;
        col_upper[u] = centre + half_width;
      }
      // A reference value that respects whichever bounds exist.
      double v = centre + (unit(rng) - 0.5) * half_width;
      if (is_finite_bound(col_lower[u])) v = std::max(v, col_lower[u]);
      if (is_finite_bound(col_upper[u])) v = std::min(v, col_upper[u]);
      x0[u] = v;
    }

    std::vector<std::vector<double>> rows;
    std::vector<double> row_lower(um, 0.0);
    std::vector<double> row_upper(um, 0.0);
    for (Index i = 0; i < m; ++i) {
      std::vector<double> row(un, 0.0);
      bool any = false;
      for (Index j = 0; j < n; ++j) {
        if (unit(rng) < 0.6) {
          row[static_cast<std::size_t>(j)] = coefficient(rng);
          any = true;
        }
      }
      if (!any) row[0] = 1.0;  // an all-zero row carries no information
      rows.push_back(row);

      double activity = 0.0;
      for (Index j = 0; j < n; ++j) {
        activity += row[static_cast<std::size_t>(j)] * x0[static_cast<std::size_t>(j)];
      }

      const auto u = static_cast<std::size_t>(i);
      const double slack_below = 0.5 + 3.0 * unit(rng);
      const double slack_above = 0.5 + 3.0 * unit(rng);
      const double roll = unit(rng);
      if (roll < 0.25) {
        row_lower[u] = activity;  // equality, tight on x0
        row_upper[u] = activity;
      } else if (roll < 0.5) {
        row_lower[u] = -kInf;
        row_upper[u] = activity + slack_above;
      } else if (roll < 0.75) {
        row_lower[u] = activity - slack_below;
        row_upper[u] = kInf;
      } else {
        row_lower[u] = activity - slack_below;
        row_upper[u] = activity + slack_above;
      }
    }

    const Model model = make_model(unit(rng) < 0.5 ? ObjSense::kMinimize : ObjSense::kMaximize,
                                   cost, col_lower, col_upper, rows, row_lower, row_upper);

    // Sanity-check the generator itself before trusting its verdict on the solver.
    ASSERT_TRUE(model.validate().empty());

    const Solution solution = run(model);

    switch (solution.status) {
      case SolveStatus::kOptimal:
        ++optimal;
        expect_kkt_optimal(model, solution, 1e-6);
        break;
      case SolveStatus::kUnbounded: ++unbounded; break;
      case SolveStatus::kInfeasible:
        ++wrongly_infeasible;
        ADD_FAILURE() << "trial " << trial
                      << " was built around an explicit feasible point but phase 1 reported "
                         "infeasible: "
                      << solution.message;
        break;
      default:
        ++other;
        ADD_FAILURE() << "trial " << trial << " ended in " << to_string(solution.status) << ": "
                      << solution.message;
        break;
    }
  }

  // Print the distribution so a generator that drifts towards trivial instances is visible
  // in the test log rather than silently making the assertions vacuous.
  std::cout << "fuzz outcomes: optimal " << optimal << ", unbounded " << unbounded
            << ", wrongly infeasible " << wrongly_infeasible << ", other " << other << "\n";
  EXPECT_GT(optimal, 250) << "too few certified instances for this test to mean anything";
  EXPECT_EQ(wrongly_infeasible, 0);
  EXPECT_EQ(other, 0);
}

}  // namespace
}  // namespace sankhya

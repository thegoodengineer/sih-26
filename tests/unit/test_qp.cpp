// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex QP tests.
//
// Every instance here has an optimum derived BY HAND from the KKT conditions and written out
// in the comment above it, so the assertion is against arithmetic rather than against
// whatever the engine printed the first time it ran. That matters more for a first-order
// method than for the simplex: there is no basis to inspect and no pivot sequence to reason
// about, so a QP that converges to the wrong point looks exactly like one that converges to
// the right point.
//
// The non-convex cases are as important as the convex ones. CLAUDE.md's rule is that a wrong
// answer scores zero, and a non-convex QP solved to a local minimum and reported as optimal
// is the wrong answer in its most convincing form.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

Options qp_options(double tolerance = 1e-10) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_double("qp_tolerance", tolerance);
  options.set_int("iteration_limit", 500000);
  return options;
}

/// Build a QP. `hessian_lower` holds (row, col, value) triples of the LOWER triangle of Q,
/// and the objective is offset + c'x + 0.5 x'Qx.
Model make_qp(const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper,
              const std::vector<std::tuple<Index, Index, double>>& hessian_lower,
              const std::vector<std::vector<double>>& rows = {},
              const std::vector<double>& row_lower = {},
              const std::vector<double>& row_upper = {}) {
  Model model;
  model.name = "qptest";
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(cost.size(), VarType::kContinuous);
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

  model.hessian.reset(n, n);
  for (const auto& [i, j, v] : hessian_lower) model.hessian.add_entry(i, j, v);
  model.hessian.finalize();

  EXPECT_EQ(model.validate(), "");
  return model;
}

// =========================================================================================
// Convex instances with hand-derived optima
// =========================================================================================

TEST(ConvexQp, UnconstrainedMinimumOfASeparableQuadratic) {
  // min 0.5(2x^2 + 4y^2) - 2x - 8y  over x, y >= 0 (the bound is inactive at the optimum).
  // Gradient: 2x - 2 = 0 -> x = 1;  4y - 8 = 0 -> y = 2.
  // Objective: 0.5(2*1 + 4*4) - 2 - 16 = 9 - 18 = -9.
  const Model model =
      make_qp({-2.0, -8.0}, {0.0, 0.0}, {kInfinity, kInfinity}, {{0, 0, 2.0}, {1, 1, 4.0}});
  const Solution s = solve(model, qp_options());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.col_value[0], 1.0, 1e-6);
  EXPECT_NEAR(s.col_value[1], 2.0, 1e-6);
  EXPECT_NEAR(s.objective, -9.0, 1e-6);
}

TEST(ConvexQp, ABoundCutsOffTheUnconstrainedMinimum) {
  // Same objective, but x <= 0.25. The unconstrained x* = 1 is outside, so the minimiser
  // sits on the bound at x = 0.25 and y is unaffected at 2.
  // Objective: 0.5(2*0.0625 + 4*4) - 2(0.25) - 8(2) = 0.0625 + 8 - 0.5 - 16 = -8.4375.
  const Model model =
      make_qp({-2.0, -8.0}, {0.0, 0.0}, {0.25, kInfinity}, {{0, 0, 2.0}, {1, 1, 4.0}});
  const Solution s = solve(model, qp_options());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.col_value[0], 0.25, 1e-6);
  EXPECT_NEAR(s.col_value[1], 2.0, 1e-6);
  EXPECT_NEAR(s.objective, -8.4375, 1e-6);
}

TEST(ConvexQp, ACouplingRowMovesTheOptimumOffBothBounds) {
  // min 0.5(x^2 + y^2)  s.t.  x + y = 2,  x, y free above 0.
  // Symmetry and the equality give x = y = 1; objective 0.5(1 + 1) = 1.
  // The Lagrangian is x - lambda = 0, so lambda = 1 at the optimum.
  const Model model = make_qp({0.0, 0.0}, {0.0, 0.0}, {kInfinity, kInfinity},
                              {{0, 0, 1.0}, {1, 1, 1.0}}, {{1.0, 1.0}}, {2.0}, {2.0});
  const Solution s = solve(model, qp_options());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.col_value[0], 1.0, 1e-5);
  EXPECT_NEAR(s.col_value[1], 1.0, 1e-5);
  EXPECT_NEAR(s.objective, 1.0, 1e-5);
}

TEST(ConvexQp, AnOffDiagonalHessianEntryCountsTwice) {
  // THE STORAGE TRAP. Q is held as its LOWER TRIANGLE and the objective is 0.5 x'Qx, so a
  // stored off-diagonal entry stands for TWO entries of the symmetric matrix. An engine that
  // reads the triangle literally solves a different problem and converges happily to it.
  //
  // Q = [[2, 1], [1, 2]], c = (-4, -4). Stored: (0,0)=2, (1,0)=1, (1,1)=2.
  // Stationarity: Qx = -c  ->  2x + y = 4,  x + 2y = 4  ->  x = y = 4/3.
  // Objective: 0.5 x'Qx + c'x = 0.5 * (4/3)(2*4/3 + 4/3) * 2 ... evaluated directly:
  //   0.5 * (2*(16/9) + 2*(16/9) + 2*(16/9)) - 4*(4/3) - 4*(4/3)
  //   = 0.5 * (32/9 + 32/9) ... use evaluate_objective instead of restating it here.
  // The check below pins x, and the objective is taken from the model's own evaluator so the
  // test cannot disagree with the convention it is testing.
  const Model model = make_qp({-4.0, -4.0}, {0.0, 0.0}, {kInfinity, kInfinity},
                              {{0, 0, 2.0}, {1, 0, 1.0}, {1, 1, 2.0}});
  const Solution s = solve(model, qp_options());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.col_value[0], 4.0 / 3.0, 1e-5);
  EXPECT_NEAR(s.col_value[1], 4.0 / 3.0, 1e-5);

  const std::vector<double> expected = {4.0 / 3.0, 4.0 / 3.0};
  EXPECT_NEAR(s.objective, model.evaluate_objective(expected.data()), 1e-5);
}

TEST(ConvexQp, MaximizationIsReportedInTheOriginalSense) {
  // max -(0.5 x^2) + 2x  ->  the concave maximum is at x = 2, value 2.
  Model model = make_qp({2.0}, {0.0}, {kInfinity}, {{0, 0, -1.0}});
  model.sense = ObjSense::kMaximize;
  const Solution s = solve(model, qp_options());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.col_value[0], 2.0, 1e-5);
  EXPECT_NEAR(s.objective, 2.0, 1e-5);
}

TEST(ConvexQp, TheReportedPointIsActuallyFeasible) {
  // solve() downgrades any status whose measured primal infeasibility exceeds tolerance, so
  // this asserts the engine earns kOptimal rather than merely claiming it.
  const Model model = make_qp({0.0, 0.0}, {0.0, 0.0}, {kInfinity, kInfinity},
                              {{0, 0, 1.0}, {1, 1, 1.0}}, {{1.0, 1.0}}, {2.0}, {2.0});
  const Solution s = solve(model, qp_options());
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_LE(s.primal_infeasibility, tol::kPrimalFeasibility);
}

// =========================================================================================
// Non-convex models must be REFUSED
// =========================================================================================

TEST(ConvexQp, ANegativeCurvatureDirectionIsRefused) {
  // Q = diag(1, -1) is indefinite: the objective falls without limit along y. An engine that
  // solved this would report a saddle point as an optimum.
  const Model model = make_qp({0.0, 0.0}, {-kInfinity, -kInfinity}, {kInfinity, kInfinity},
                              {{0, 0, 1.0}, {1, 1, -1.0}});
  const Solution s = solve(model, qp_options());
  EXPECT_EQ(s.status, SolveStatus::kModelError);
  EXPECT_NE(s.message.find("not convex"), std::string::npos) << s.message;
  EXPECT_FALSE(s.has_primal_values());
}

TEST(ConvexQp, IndefinitenessHiddenOffTheDiagonalIsAlsoRefused) {
  // Every diagonal entry is POSITIVE, so a diagonal-only check would call this convex.
  // Q = [[1, 3], [3, 1]] has eigenvalues 4 and -2: indefinite. This is the case that
  // separates an LDL^T test from a cheap sign check on the diagonal.
  const Model model = make_qp({0.0, 0.0}, {-kInfinity, -kInfinity}, {kInfinity, kInfinity},
                              {{0, 0, 1.0}, {1, 0, 3.0}, {1, 1, 1.0}});
  const Solution s = solve(model, qp_options());
  EXPECT_EQ(s.status, SolveStatus::kModelError);
  EXPECT_NE(s.message.find("not convex"), std::string::npos) << s.message;
}

TEST(ConvexQp, ASingularButSemidefiniteHessianIsAccepted) {
  // Q = [[1, 1], [1, 1]] is positive SEMIdefinite and singular - rank 1, eigenvalues 2 and 0.
  // Rank deficiency is not non-convexity, and refusing it would reject a large class of
  // legitimate models (least squares with redundant columns, for one).
  //   min 0.5 (x + y)^2 - 2(x + y)  is minimised wherever x + y = 2, value -2.
  const Model model =
      make_qp({-2.0, -2.0}, {0.0, 0.0}, {10.0, 10.0}, {{0, 0, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
  const Solution s = solve(model, qp_options(1e-9));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NEAR(s.col_value[0] + s.col_value[1], 2.0, 1e-4);
  EXPECT_NEAR(s.objective, -2.0, 1e-4);
}

TEST(ConvexQp, AnLpIsNotDivertedToTheQpEngine) {
  // A model with an empty Hessian is an LP and must keep going to the simplex, which is
  // exact and produces a basis. Classification is on the model, not on a user assertion.
  const Model model = make_qp({1.0, 1.0}, {0.0, 0.0}, {kInfinity, kInfinity}, {}, {{1.0, 1.0}},
                              {2.0}, {kInfinity});
  Options options;
  options.set_bool("log_to_console", false);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  // Either simplex engine: the dual is the automatic choice since #65, the primal stays
  // selectable, and both are exact and produce a basis.
  EXPECT_EQ(s.algorithm.rfind("simplex-", 0), 0u) << s.algorithm;
}

}  // namespace
}  // namespace sankhya

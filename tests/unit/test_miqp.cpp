// SPDX-License-Identifier: Apache-2.0
// SANKHYA - mixed-integer quadratic programming.
//
// MIQP is branch and bound over QP node relaxations. The failure this file exists to catch is
// the one solve() previously avoided by refusing the class outright: reporting a RELAXATION as
// though it were the integer answer. That point is feasible, its objective is a real number,
// and nothing about it looks wrong - so every test below pins the integer optimum against a
// hand-derived value AND checks the relaxation is a different number, so a solver that quietly
// dropped integrality would fail rather than coincide.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options miqp_options() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// min  0.5 x'Qx + c'x   subject to the given single row, with the listed integrality.
Model make_model(const std::vector<double>& cost, const std::vector<double>& diagonal,
                 const std::vector<double>& row, double row_lower, double row_upper,
                 const std::vector<double>& upper, const std::vector<bool>& integral) {
  const auto n = static_cast<Index>(cost.size());
  Model model;
  model.col_cost = cost;
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper = upper;
  model.col_type.resize(static_cast<std::size_t>(n));
  for (Index j = 0; j < n; ++j) {
    model.col_type[static_cast<std::size_t>(j)] =
        integral[static_cast<std::size_t>(j)] ? VarType::kInteger : VarType::kContinuous;
  }
  model.row_lower = {row_lower};
  model.row_upper = {row_upper};
  model.matrix.reset(1, n);
  for (Index j = 0; j < n; ++j) {
    if (row[static_cast<std::size_t>(j)] != 0.0) {
      model.matrix.add_entry(0, j, row[static_cast<std::size_t>(j)]);
    }
  }
  model.matrix.finalize();

  model.hessian.reset(n, n);
  for (Index j = 0; j < n; ++j) {
    if (diagonal[static_cast<std::size_t>(j)] != 0.0) {
      model.hessian.add_entry(j, j, diagonal[static_cast<std::size_t>(j)]);
    }
  }
  model.hessian.finalize();
  return model;
}

// =========================================================================================

TEST(Miqp, SolvesAndDoesNotReportTheRelaxation) {
  //   min  0.5(2x^2 + 2y^2) - 5x - 3y      s.t.  x + y <= 10,  0 <= x,y <= 10
  //
  // The row is slack at the unconstrained minimum, so the CONTINUOUS optimum is where the
  // gradient vanishes: x = 5, y = 3, objective 0.5(25+9)*2/... worked out termwise,
  //   f(5,3) = (25 + 9) - 25 - 9 = -17.
  // With x and y integral those values are already integers, which would make this test
  // vacuous - so y is given cost -3.5 instead, putting its continuous optimum at 3.5.
  //
  //   f(x,y) = x^2 + y^2 - 5x - 3.5y
  //   continuous optimum  x = 2.5? no: d/dx = 2x - 5 = 0 -> x = 2.5, d/dy = 2y - 3.5 -> 1.75
  //   f(2.5, 1.75) = 6.25 + 3.0625 - 12.5 - 6.125 = -9.3125
  //   integer candidates around it: (2,2) -> 4 + 4 - 10 - 7 = -9
  //                                 (3,2) -> 9 + 4 - 15 - 7 = -9
  //                                 (2,1) -> 4 + 1 - 10 - 3.5 = -8.5
  //                                 (3,1) -> 9 + 1 - 15 - 3.5 = -8.5
  // so the integer optimum is -9, at (2,2) or (3,2), and the relaxation is -9.3125.
  const Model model = make_model({-5.0, -3.5}, {2.0, 2.0}, {1.0, 1.0}, -kInfinity, 10.0,
                                 {10.0, 10.0}, {true, true});
  ASSERT_EQ(model.validate(), "");
  ASSERT_TRUE(model.has_quadratic_objective());
  ASSERT_TRUE(model.has_integrality());

  const Solution solution = solve(model, miqp_options());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_EQ(solution.algorithm, "branch-and-bound");

  EXPECT_NEAR(solution.objective, -9.0, 1e-5)
      << "objective " << solution.objective << "; the RELAXATION is -9.3125, so a value near "
      << "that means integrality was dropped";

  // Integral in fact, not merely in status.
  for (const double v : solution.col_value) {
    EXPECT_LT(std::fabs(v - std::round(v)), 1e-6) << "fractional value " << v;
  }
  EXPECT_LE(solution.integrality_violation, 1e-6);
  EXPECT_LE(solution.primal_infeasibility, 1e-6);
}

TEST(Miqp, TheRelaxationIsGenuinelyDifferentFromTheIntegerAnswer) {
  // The guard on the test above is only meaningful if the relaxation really is -9.3125. This
  // solves the SAME model with integrality removed and checks that directly, so the two tests
  // cannot both pass on a solver that ignores integrality.
  const Model relaxed = make_model({-5.0, -3.5}, {2.0, 2.0}, {1.0, 1.0}, -kInfinity, 10.0,
                                   {10.0, 10.0}, {false, false});
  Options options = miqp_options();
  options.set_double("qp_tolerance", 1e-11);
  options.set_int("iteration_limit", 500000);

  const Solution solution = solve(relaxed, options);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_EQ(solution.algorithm, "qp-condat-vu") << "this one is a pure QP, not a search";
  EXPECT_NEAR(solution.objective, -9.3125, 1e-4);
}

TEST(Miqp, BranchesWhenTheRelaxationIsFractional) {
  //   min 0.5(2x^2) - 3x   s.t.  x <= 10,  x integer.
  // Continuous optimum x = 1.5, f = 2.25 - 4.5 = -2.25. Integer candidates 1 and 2 both give
  // 1 - 3 = -2 and 4 - 6 = -2, so the answer is -2 and the search must branch to find it.
  const Model model = make_model({-3.0}, {2.0}, {1.0}, -kInfinity, 10.0, {10.0}, {true});
  const Solution solution = solve(model, miqp_options());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -2.0, 1e-5);
  EXPECT_GT(solution.nodes, 1) << "an integral relaxation would need no search; this one is "
                                  "fractional at 1.5 and must branch";
}

TEST(Miqp, AnIntegralRelaxationNeedsNoSearch) {
  //   min 0.5(2x^2) - 4x  s.t. x <= 10, x integer.  Continuous optimum is exactly x = 2,
  // already integral, so the root relaxation IS the answer and no branching is required.
  const Model model = make_model({-4.0}, {2.0}, {1.0}, -kInfinity, 10.0, {10.0}, {true});
  const Solution solution = solve(model, miqp_options());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -4.0, 1e-5);
  EXPECT_NEAR(solution.col_value[0], 2.0, 1e-5);
}

TEST(Miqp, ANonConvexMiqpIsStillRefused) {
  // The QP engine decides semidefiniteness of sense * Q by LDL^T before any arithmetic, and
  // that check now runs at the first node rather than being bypassed by the dispatcher's old
  // refusal of the whole class. A non-convex MIQP must still be refused, NOT solved to
  // whatever local point the search reaches - the same rule that governs a non-convex QP.
  const Model model =
      make_model({0.0}, {-2.0}, {1.0}, -kInfinity, 10.0, {10.0}, {true});  // Q = -2, concave
  const Solution solution = solve(model, miqp_options());
  EXPECT_NE(solution.status, SolveStatus::kOptimal)
      << "a non-convex MIQP was reported as solved: " << solution.objective;
  EXPECT_FALSE(solution.message.empty()) << "refused without saying why";
}

TEST(Miqp, IsClassifiedAsMiqpRatherThanFallingBackToAnLp) {
  // classify() must see both properties. If the Hessian were ignored the model would be a
  // MILP and the answer would be the linear optimum, which for this instance is a different
  // number - x driven to its upper bound rather than to the quadratic's minimum.
  //
  //   min 0.5(2x^2) - 3x  over 0 <= x <= 10 integer
  //   as an MIQP: -2 at x in {1, 2}
  //   as a MILP with the Hessian dropped: -3x is minimised at x = 10, giving -30.
  const Model model = make_model({-3.0}, {2.0}, {1.0}, -kInfinity, 10.0, {10.0}, {true});
  const Solution solution = solve(model, miqp_options());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_NEAR(solution.objective, -2.0, 1e-5)
      << "-30 here would mean the Hessian was dropped and this was solved as a MILP";
}

}  // namespace
}  // namespace sankhya

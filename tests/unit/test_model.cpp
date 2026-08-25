// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Model / Solution tests.
//
// Model is a frozen interface, so these tests are as much a specification as a check. In
// particular they pin down the two conventions that a later reader or engine is most likely
// to get quietly wrong:
//   * rows are two-sided, and an equality is lower == upper
//   * the Hessian is stored lower-triangular and the objective term is 0.5 x^T Q x, so a
//     stored off-diagonal entry stands for two entries of the symmetric matrix

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"

namespace sankhya {
namespace {

/// A tiny well-formed LP:
///     min  x0 + 2 x1
///     s.t. 1 <= x0 +   x1 <= 4      (range row)
///                x0 - 2 x1  = 0     (equality row)
///          0 <= x0 <= 10, x1 free
Model make_small_lp() {
  Model model;
  model.name = "small";
  model.sense = ObjSense::kMinimize;
  model.col_cost = {1.0, 2.0};
  model.col_lower = {0.0, -kInfinity};
  model.col_upper = {10.0, kInfinity};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.col_names = {"x0", "x1"};
  model.row_lower = {1.0, 0.0};
  model.row_upper = {4.0, 0.0};
  model.row_names = {"range", "equality"};

  model.matrix.reset(2, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.add_entry(1, 1, -2.0);
  model.matrix.finalize();
  return model;
}

TEST(Model, EmptyModelValidates) {
  Model model;
  model.matrix.reset(0, 0);
  model.matrix.finalize();
  EXPECT_EQ(model.validate(), "");
  EXPECT_EQ(model.num_rows(), 0);
  EXPECT_EQ(model.num_cols(), 0);
}

TEST(Model, SmallLpValidates) {
  const Model model = make_small_lp();
  EXPECT_EQ(model.validate(), "");
  EXPECT_EQ(model.num_rows(), 2);
  EXPECT_EQ(model.num_cols(), 2);
  EXPECT_EQ(model.num_nonzeros(), 4);
  EXPECT_FALSE(model.has_integrality());
  EXPECT_FALSE(model.has_quadratic_objective());
  EXPECT_EQ(model.num_integer_columns(), 0);
  EXPECT_TRUE(model.is_equality_row(1));
  EXPECT_FALSE(model.is_equality_row(0));
  EXPECT_FALSE(model.is_fixed_column(0));
  EXPECT_DOUBLE_EQ(model.sense_multiplier(), 1.0);
}

TEST(Model, ResizeUsesMpsDefaultBounds) {
  Model model;
  model.resize_columns(3);
  ASSERT_EQ(model.num_cols(), 3);
  for (Index j = 0; j < 3; ++j) {
    const auto u = static_cast<std::size_t>(j);
    EXPECT_DOUBLE_EQ(model.col_cost[u], 0.0);
    EXPECT_DOUBLE_EQ(model.col_lower[u], 0.0);
    EXPECT_TRUE(is_infinite(model.col_upper[u]));
    EXPECT_EQ(model.col_type[u], VarType::kContinuous);
  }
  model.resize_rows(2);
  ASSERT_EQ(model.num_rows(), 2);
  for (Index i = 0; i < 2; ++i) {
    const auto u = static_cast<std::size_t>(i);
    EXPECT_TRUE(is_infinite(model.row_lower[u]));
    EXPECT_TRUE(is_infinite(model.row_upper[u]));
  }
}

TEST(Model, ValidateRejectsCrossedColumnBounds) {
  Model model = make_small_lp();
  model.col_lower[0] = 5.0;
  model.col_upper[0] = 1.0;
  const std::string problem = model.validate();
  EXPECT_NE(problem, "");
  EXPECT_NE(problem.find("column 0"), std::string::npos);
}

TEST(Model, ValidateRejectsCrossedRowBounds) {
  Model model = make_small_lp();
  model.row_lower[0] = 9.0;
  model.row_upper[0] = 2.0;
  EXPECT_NE(model.validate().find("row 0"), std::string::npos);
}

TEST(Model, ValidateRejectsNaN) {
  Model model = make_small_lp();
  model.col_cost[1] = std::nan("");
  EXPECT_NE(model.validate(), "");
}

TEST(Model, ValidateRejectsAnUnfinalizedMatrix) {
  Model model = make_small_lp();
  model.matrix.unfreeze();
  EXPECT_NE(model.validate().find("not finalized"), std::string::npos);
}

TEST(Model, ValidateRejectsDimensionMismatch) {
  Model model = make_small_lp();
  model.col_cost.push_back(1.0);  // 3 costs, 2 bounds
  EXPECT_NE(model.validate(), "");
}

TEST(Model, IntegralityIsDetected) {
  Model model = make_small_lp();
  model.col_type[1] = VarType::kInteger;
  EXPECT_TRUE(model.has_integrality());
  EXPECT_EQ(model.num_integer_columns(), 1);
  EXPECT_EQ(model.validate(), "");
}

TEST(Model, LinearObjectiveIncludesTheOffset) {
  Model model = make_small_lp();
  model.objective_offset = 7.5;
  const std::vector<double> x = {2.0, 1.0};
  EXPECT_DOUBLE_EQ(model.evaluate_objective(x.data()), 7.5 + 2.0 + 2.0);
}

TEST(Model, QuadraticObjectiveUsesLowerTriangleAndHalfFactor) {
  // f(x) = 0.5 * x^T Q x with Q = [[2, 1], [1, 4]].
  // Stored lower triangle: (0,0)=2, (1,0)=1, (1,1)=4.
  // At x = (3, 5): 0.5*(2*9) + 1*(3*5) + 0.5*(4*25) = 9 + 15 + 50 = 74.
  Model model;
  model.col_cost = {0.0, 0.0};
  model.col_lower = {-kInfinity, -kInfinity};
  model.col_upper = {kInfinity, kInfinity};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.matrix.reset(0, 2);
  model.matrix.finalize();

  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.add_entry(1, 0, 1.0);
  model.hessian.add_entry(1, 1, 4.0);
  model.hessian.finalize();

  ASSERT_EQ(model.validate(), "");
  EXPECT_TRUE(model.has_quadratic_objective());
  const std::vector<double> x = {3.0, 5.0};
  EXPECT_DOUBLE_EQ(model.evaluate_objective(x.data()), 74.0);
}

TEST(Model, ValidateRejectsAnUpperTriangularHessianEntry) {
  Model model;
  model.col_cost = {0.0, 0.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {1.0, 1.0};
  model.col_type = {VarType::kContinuous, VarType::kContinuous};
  model.matrix.reset(0, 2);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.add_entry(0, 1, 1.0);  // above the diagonal
  model.hessian.finalize();
  EXPECT_NE(model.validate().find("lower triangle"), std::string::npos);
}

TEST(Model, MaximizeKeepsItsSenseMultiplier) {
  Model model = make_small_lp();
  model.sense = ObjSense::kMaximize;
  EXPECT_DOUBLE_EQ(model.sense_multiplier(), -1.0);
  // evaluate_objective always reports in the sense of the original file, so the value is
  // unchanged by the sense; only the engines' internal minimization flips.
  const std::vector<double> x = {2.0, 1.0};
  EXPECT_DOUBLE_EQ(model.evaluate_objective(x.data()), 4.0);
}

// =========================================================================================
// Solution
// =========================================================================================

TEST(Solution, AllocateForSizesEveryVector) {
  const Model model = make_small_lp();
  Solution solution;
  solution.allocate_for(model);
  EXPECT_EQ(solution.col_value.size(), 2u);
  EXPECT_EQ(solution.col_dual.size(), 2u);
  EXPECT_EQ(solution.col_status.size(), 2u);
  EXPECT_EQ(solution.row_activity.size(), 2u);
  EXPECT_EQ(solution.row_dual.size(), 2u);
  EXPECT_EQ(solution.row_status.size(), 2u);
  EXPECT_EQ(solution.status, SolveStatus::kNotSolved);
  EXPECT_FALSE(solution.has_primal_values());
}

TEST(Solution, RecomputeQualityFindsAFeasiblePoint) {
  const Model model = make_small_lp();
  Solution solution;
  solution.allocate_for(model);
  // x = (2, 1): row 0 activity 3 in [1, 4]; row 1 activity 0 == 0. Feasible.
  solution.col_value = {2.0, 1.0};
  solution.recompute_quality(model);

  EXPECT_DOUBLE_EQ(solution.row_activity[0], 3.0);
  EXPECT_DOUBLE_EQ(solution.row_activity[1], 0.0);
  EXPECT_DOUBLE_EQ(solution.objective, 4.0);
  EXPECT_LE(solution.primal_infeasibility, 0.0);
  EXPECT_DOUBLE_EQ(solution.integrality_violation, 0.0);
}

TEST(Solution, RecomputeQualityMeasuresTheWorstViolation) {
  const Model model = make_small_lp();
  Solution solution;
  solution.allocate_for(model);
  // x = (0, 0): row 0 activity 0, below its lower bound of 1 -> violation 1.
  solution.col_value = {0.0, 0.0};
  solution.recompute_quality(model);
  EXPECT_DOUBLE_EQ(solution.primal_infeasibility, 1.0);

  // x = (10, -5): x0 at its upper bound, row 0 activity 5 above its upper bound of 4,
  // row 1 activity 20 against an equality at 0 -> the worst violation is 20.
  solution.col_value = {10.0, -5.0};
  solution.recompute_quality(model);
  EXPECT_DOUBLE_EQ(solution.row_activity[0], 5.0);
  EXPECT_DOUBLE_EQ(solution.row_activity[1], 20.0);
  EXPECT_DOUBLE_EQ(solution.primal_infeasibility, 20.0);
}

TEST(Solution, RecomputeQualityMeasuresIntegralityViolation) {
  Model model = make_small_lp();
  model.col_type[0] = VarType::kInteger;
  Solution solution;
  solution.allocate_for(model);
  solution.col_value = {2.25, 1.125};
  solution.recompute_quality(model);
  // Only column 0 is integral, so column 1 being fractional must not register.
  EXPECT_DOUBLE_EQ(solution.integrality_violation, 0.25);
}

TEST(Solution, RecomputeQualityFillsTheGaps) {
  const Model model = make_small_lp();
  Solution solution;
  solution.allocate_for(model);
  solution.col_value = {2.0, 1.0};
  solution.dual_bound = 3.5;
  solution.recompute_quality(model);
  EXPECT_DOUBLE_EQ(solution.objective, 4.0);
  EXPECT_DOUBLE_EQ(solution.absolute_gap, 0.5);
  EXPECT_DOUBLE_EQ(solution.relative_gap, 0.5 / 4.0);
}

TEST(Solution, StatusStringsAreStable) {
  // The .sol file and the JSON blob are consumed by tools/verify_solution.py, which is
  // Python and matches on these exact strings.
  EXPECT_STREQ(to_string(SolveStatus::kOptimal), "optimal");
  EXPECT_STREQ(to_string(SolveStatus::kInfeasible), "infeasible");
  EXPECT_STREQ(to_string(SolveStatus::kUnbounded), "unbounded");
  EXPECT_STREQ(to_string(SolveStatus::kTimeLimit), "time_limit");
  EXPECT_STREQ(to_string(BasisStatus::kBasic), "basic");
  EXPECT_STREQ(to_string(BasisStatus::kAtLower), "at_lower");
  EXPECT_STREQ(to_string(VarType::kInteger), "integer");
}

// =========================================================================================
// solve() dispatcher
// =========================================================================================

TEST(Solve, ReportsAModelErrorRatherThanGuessing) {
  Model model = make_small_lp();
  model.col_lower[0] = 5.0;
  model.col_upper[0] = 1.0;
  Options options;
  options.set_bool("log_to_console", false);
  const Solution solution = solve(model, options);
  EXPECT_EQ(solution.status, SolveStatus::kModelError);
  EXPECT_NE(solution.message, "");
}

TEST(Solve, DispatchesAnLpToTheSimplex) {
  // Phase 2 registered the primal simplex, so an LP is now actually solved and the engine
  // that did it is named in the result. Anything that reaches solve() must either produce
  // a certified answer or say plainly that it could not.
  const Model model = make_small_lp();
  Options options;
  options.set_bool("log_to_console", false);
  const Solution solution = solve(model, options);
  EXPECT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_EQ(solution.algorithm, "simplex-primal");
  EXPECT_TRUE(solution.has_primal_values());
  EXPECT_LE(solution.primal_infeasibility, tol::kPrimalFeasibility);
}

TEST(Solve, RefusesAMilpRatherThanReportingItsRelaxation) {
  // The single most damaging thing the dispatcher could do is hand a MILP to the simplex
  // and report the fractional relaxation as optimal. Branch and cut lands in Phase 5; until
  // then this path must refuse. The relaxation is a valid bound, not a solution, and
  // nothing downstream is allowed to confuse the two.
  Model model = make_small_lp();
  model.col_type[0] = VarType::kInteger;
  Options options;
  options.set_bool("log_to_console", false);
  const Solution solution = solve(model, options);
  EXPECT_EQ(solution.status, SolveStatus::kNotSolved);
  EXPECT_EQ(solution.algorithm, "none");
  EXPECT_NE(solution.message.find("MILP"), std::string::npos);
  EXPECT_FALSE(solution.has_primal_values());
}

// =========================================================================================
// Coefficient values
//
// validate() is the LAST gate before an engine sees a model, and the only one a model built
// through the C API in Phase 10 will pass through at all - such a model never touches a
// reader. So the check has to live here as well as in the readers, not instead of.
// =========================================================================================

TEST(Model, ValidateRejectsANaNMatrixCoefficient) {
  Model model = make_small_lp();
  model.matrix.unfreeze();
  model.matrix.add_entry(0, 0, std::numeric_limits<double>::quiet_NaN());
  model.matrix.finalize();
  const std::string problem = model.validate();
  EXPECT_NE(problem.find("not a usable coefficient"), std::string::npos) << problem;
}

TEST(Model, ValidateRejectsAnInfiniteMatrixCoefficient) {
  Model model = make_small_lp();
  model.matrix.unfreeze();
  model.matrix.add_entry(0, 0, kInfinity);
  model.matrix.finalize();
  const std::string problem = model.validate();
  EXPECT_NE(problem.find("not a usable coefficient"), std::string::npos) << problem;
}

TEST(SparseMatrix, FinalizeDoesNotSilentlyDeleteANaN) {
  // The direct spelling of the drop test, `fabs(sum) >= drop_tol`, is false for NaN and so
  // deletes the entry - converting a corrupt model into a well formed model of a different
  // problem, which is the single most dangerous outcome in this codebase. The entry must
  // survive finalize() so that validate() can name it.
  SparseMatrix matrix;
  matrix.reset(2, 2);
  matrix.add_entry(0, 0, std::numeric_limits<double>::quiet_NaN());
  matrix.add_entry(1, 1, 1.0);
  matrix.finalize();
  EXPECT_EQ(matrix.num_nonzeros(), 2) << "the NaN entry was dropped instead of retained";
}

}  // namespace
}  // namespace sankhya

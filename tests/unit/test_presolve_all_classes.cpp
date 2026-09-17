// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve reaches every model class, and carries what each class needs (#301).
//
// Presolve used to run for an LP only. A MILP, QP or MIQP went straight to its engine with
// the `presolve` option silently ignored - the user asked for reductions and got none, and
// nothing said so.
//
// The reason it was LP-only is the interesting part: the reduced model's Hessian was reset to
// empty, so running the same pipeline on a QP would have handed the engine a model whose
// curvature had quietly vanished - an LP wearing a QP's name, answered confidently. The fix
// is not "call presolve from more places". It is to make the reductions safe for what each
// class carries: columns with Hessian entries are protected from removal and substitution,
// the Hessian travels with the reduced model, and the two reductions that could hand back a
// fractional value for an integer column now decline it.
//
// So the tests below are mostly the same shape: solve with presolve on and off, and require
// the same answer - after first checking that presolve actually removed something, because a
// reduction that fires on nothing proves nothing.

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "presolve/presolve.hpp"

namespace sankhya {
namespace {

Options base_options(bool presolve) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", presolve);
  options.set_int("node_limit", 100000);
  return options;
}

/// rows x cols dense coefficients, plus bounds, costs, integrality and an optional Hessian.
Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            const std::vector<bool>& integral,
            const std::vector<std::tuple<Index, Index, double>>& hessian_lower = {}) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  for (Index j = 0; j < n; ++j) {
    if (integral[static_cast<std::size_t>(j)]) {
      model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
    }
  }
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

/// What presolve did to a model, without solving it.
presolve::Result reduce(const Model& model) {
  Options options = base_options(true);
  Logger logger(nullptr);
  return presolve::presolve(model, options, logger);
}

// =========================================================================================
// It runs at all, for every class
// =========================================================================================

TEST(PresolveEveryClass, AMilpIsReducedAndTheAnswerIsUnchanged) {
  //   min -x0 - x1 + 0*x2   s.t.  x0 + x1 <= 3,  x2 <= 50 (redundant against x2 <= 1),
  //                               x0, x1 binary, x2 fixed at 1
  // The redundant row and the fixed column are both presolve's to remove; the answer is
  // x0 = x1 = 1, objective -2.
  const Model model =
      build({{1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {-kInfinity, -kInfinity}, {3.0, 50.0},
            {-1.0, -1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {true, true, true});

  const presolve::Result reduced = reduce(model);
  EXPECT_GT(reduced.rows_removed(), 0) << "the redundant row should go";
  EXPECT_GT(reduced.cols_removed(), 0) << "the fixed column should go";

  const Solution on = solve(model, base_options(true));
  const Solution off = solve(model, base_options(false));
  EXPECT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_EQ(off.status, SolveStatus::kOptimal) << off.message;
  EXPECT_NEAR(on.objective, off.objective, 1e-9);
  EXPECT_NEAR(on.objective, -2.0, 1e-9);
  ASSERT_EQ(on.col_value.size(), 3u);
  EXPECT_NEAR(on.col_value[2], 1.0, 1e-9) << "the removed column comes back at its value";
  EXPECT_LE(on.integrality_violation, 1e-6);
}

TEST(PresolveEveryClass, AQpKeepsItsCurvatureThroughPresolve) {
  //   min (x0 - 2)^2 + (x1 - 3)^2 written as x0^2 + x1^2 - 4x0 - 6x1 + 13, with a redundant
  //   row and a third column fixed at 5 carrying no curvature.
  // Unconstrained optimum (2, 3); objective -13 + 13 = 0 with the offset left out, so the
  // value to compare is simply "the same as with presolve off".
  const Model model =
      build({{1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {-kInfinity, -kInfinity}, {100.0, 50.0},
            {-4.0, -6.0, 1.0}, {0.0, 0.0, 5.0}, {10.0, 10.0, 5.0}, {false, false, false},
            {{0, 0, 2.0}, {1, 1, 2.0}});

  const presolve::Result reduced = reduce(model);
  EXPECT_GT(reduced.rows_removed(), 0);
  EXPECT_GT(reduced.cols_removed(), 0) << "the fixed non-quadratic column should go";
  // The point of the change: the quadratic objective is still there afterwards.
  EXPECT_EQ(reduced.model.hessian.num_nonzeros(), model.hessian.num_nonzeros())
      << "presolve used to hand the engine a model with an empty Hessian";

  const Solution on = solve(model, base_options(true));
  const Solution off = solve(model, base_options(false));
  EXPECT_EQ(on.status, off.status) << on.message << " / " << off.message;
  EXPECT_NEAR(on.objective, off.objective, 1e-6);
  ASSERT_EQ(on.col_value.size(), 3u);
  EXPECT_NEAR(on.col_value[0], 2.0, 1e-4);
  EXPECT_NEAR(on.col_value[1], 3.0, 1e-4);
  EXPECT_NEAR(on.col_value[2], 5.0, 1e-9);
}

TEST(PresolveEveryClass, AColumnWithCurvatureIsNeverRemovedEvenWhenItLooksRemovable) {
  // x0 is FIXED at 2 and carries curvature. The fixed-column reduction would fold c_j * v
  // into the objective constant and drop the column - and lose 0.5 * Q_00 * v^2 with it,
  // which is 4 here. Protecting the column is what keeps the objective right.
  const Model model = build({{1.0, 1.0}}, {-kInfinity}, {10.0}, {0.0, -2.0}, {2.0, 0.0},
                            {2.0, 5.0}, {false, false}, {{0, 0, 2.0}, {1, 1, 2.0}});

  const presolve::Result reduced = reduce(model);
  EXPECT_EQ(reduced.cols_removed(), 0) << "a column in the Hessian must survive presolve";
  EXPECT_EQ(reduced.model.hessian.num_nonzeros(), 2);

  const Solution on = solve(model, base_options(true));
  const Solution off = solve(model, base_options(false));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  EXPECT_NEAR(on.objective, off.objective, 1e-6);
  // x0 = 2 contributes 0.5 * 2 * 4 = 4; x1 minimises x1^2 - 2 x1 at x1 = 1, contributing -1.
  EXPECT_NEAR(on.objective, 3.0, 1e-4);
}

TEST(PresolveEveryClass, AMiqpIsReducedAndKeepsBothIntegralityAndCurvature) {
  //   min (x0 - 1.4)^2 + (x1 - 2)^2 with x0 integer, plus a redundant row and a fixed
  //   continuous column with no curvature.
  const Model model =
      build({{1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {-kInfinity, -kInfinity}, {100.0, 50.0},
            {-2.8, -4.0, 0.0}, {0.0, 0.0, 3.0}, {10.0, 10.0, 3.0}, {true, false, false},
            {{0, 0, 2.0}, {1, 1, 2.0}});

  const presolve::Result reduced = reduce(model);
  EXPECT_GT(reduced.rows_removed() + reduced.cols_removed(), 0);
  EXPECT_EQ(reduced.model.hessian.num_nonzeros(), model.hessian.num_nonzeros());

  const Solution on = solve(model, base_options(true));
  const Solution off = solve(model, base_options(false));
  EXPECT_EQ(on.status, off.status) << on.message << " / " << off.message;
  EXPECT_NEAR(on.objective, off.objective, 1e-4);
  ASSERT_EQ(on.col_value.size(), 3u);
  EXPECT_LE(std::fabs(on.col_value[0] - std::round(on.col_value[0])), 1e-6)
      << "the integer column must come back integral";
  EXPECT_NEAR(on.col_value[2], 3.0, 1e-9);
}

// =========================================================================================
// Integrality survives the reductions that used to be linear-only
// =========================================================================================

TEST(PresolveEveryClass, AFreeIntegerSingletonColumnIsLeftInTheModel) {
  // x1 is a FREE integer column appearing in one row. Substituting it out would recover it in
  // postsolve as (rhs - 3*x0) / 2, which is fractional whenever x0 is odd - an integer column
  // handed back fractional, with no branch ever taken on it.
  const Model model = build({{3.0, 2.0}}, {7.0}, {7.0}, {1.0, 0.0}, {0.0, -kInfinity},
                            {10.0, kInfinity}, {true, true});

  const presolve::Result reduced = reduce(model);
  EXPECT_EQ(reduced.cols_removed(), 0) << "an integer column cannot be substituted out";

  const Solution on = solve(model, base_options(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  ASSERT_EQ(on.col_value.size(), 2u);
  for (const double v : on.col_value) {
    EXPECT_LE(std::fabs(v - std::round(v)), 1e-6) << "value " << v << " is not integral";
  }
  EXPECT_LE(on.integrality_violation, 1e-6);
}

TEST(PresolveEveryClass, IntegerBoundsAreRoundedInwardBeforeAnythingBranchesOnThem) {
  // x1 appears in no row and is bounded [0.5, 2.5]; as an integer column it can only be 1 or
  // 2. Saying so is presolve's job (Achterberg et al. 2020) and it is worth more than the one
  // variable it settles: without it the relaxation the search branches on is weaker than the
  // model itself, and this model does not close at all - the branch and bound runs to its node
  // limit re-deriving x1 = 0.5 at every node. With the rounding, presolve removes the column
  // at an integer value and the model is decided outright.
  const Model model = build({{1.0, 0.0}}, {-kInfinity}, {4.0}, {-1.0, 1.0}, {0.0, 0.5},
                            {3.0, 2.5}, {true, true});

  const presolve::Result reduced = reduce(model);
  EXPECT_EQ(reduced.model.num_cols(), 0) << "both columns are settled by bound arithmetic";

  const Solution on = solve(model, base_options(true));
  ASSERT_EQ(on.status, SolveStatus::kOptimal) << on.message;
  ASSERT_EQ(on.col_value.size(), 2u);
  EXPECT_NEAR(on.col_value[0], 3.0, 1e-9);
  EXPECT_NEAR(on.col_value[1], 1.0, 1e-9) << "the cheapest integer inside [0.5, 2.5]";
  EXPECT_NEAR(on.objective, -2.0, 1e-9);
  EXPECT_LE(on.integrality_violation, 1e-6);
}

TEST(PresolveEveryClass, RoundingNeverWidensAnIntegerBoxAndCatchesAnEmptyOne) {
  // Inward only: [1.0, 3.0] must stay [1, 3], and [0.2, 0.8] holds no integer at all, which
  // is an infeasibility presolve can prove by itself.
  const Model keeps = build({{1.0}}, {-kInfinity}, {10.0}, {1.0}, {1.0}, {3.0}, {true});
  const presolve::Result untouched = reduce(keeps);
  EXPECT_FALSE(untouched.proved_infeasible);

  const Model empty_box = build({{1.0}}, {-kInfinity}, {10.0}, {1.0}, {0.2}, {0.8}, {true});
  const presolve::Result crossed = reduce(empty_box);
  EXPECT_TRUE(crossed.proved_infeasible) << "no integer lies in [0.2, 0.8]";

  const Solution solution = solve(empty_box, base_options(true));
  EXPECT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
}

// =========================================================================================
// What presolve proves on its own
// =========================================================================================

TEST(PresolveEveryClass, AMilpPresolveProvesInfeasibleWithTheSearchsOwnConvention) {
  // x >= 5 and x <= 2, so presolve settles it from bound arithmetic alone. The reported
  // objective and gaps have to match what the branch and bound says when it finds nothing:
  // the worst representable objective and infinite gaps, never zero - a gap of zero reads as
  // a closed search.
  const Model model =
      build({{1.0}, {1.0}}, {5.0, -kInfinity}, {kInfinity, 2.0}, {1.0}, {0.0}, {10.0}, {true});

  const Solution on = solve(model, base_options(true));
  ASSERT_EQ(on.status, SolveStatus::kInfeasible) << on.message;
  EXPECT_TRUE(std::isinf(on.objective) && on.objective > 0.0) << on.objective;
  EXPECT_TRUE(std::isinf(on.absolute_gap)) << on.absolute_gap;
  EXPECT_TRUE(std::isinf(on.relative_gap)) << on.relative_gap;

  const Solution off = solve(model, base_options(false));
  EXPECT_EQ(off.status, SolveStatus::kInfeasible) << off.message;
}

TEST(PresolveEveryClass, PresolveOffChangesNothingForAnyClass) {
  const Model milp = build({{1.0, 1.0}}, {-kInfinity}, {3.0}, {-1.0, -1.0}, {0.0, 0.0},
                           {1.0, 1.0}, {true, true});
  const Model qp = build({{1.0, 1.0}}, {-kInfinity}, {100.0}, {-4.0, -6.0}, {0.0, 0.0},
                         {10.0, 10.0}, {false, false}, {{0, 0, 2.0}, {1, 1, 2.0}});

  for (const Model* model : {&milp, &qp}) {
    const Solution on = solve(*model, base_options(true));
    const Solution off = solve(*model, base_options(false));
    EXPECT_EQ(on.status, off.status);
    EXPECT_NEAR(on.objective, off.objective, 1e-6);
  }
}

}  // namespace
}  // namespace sankhya

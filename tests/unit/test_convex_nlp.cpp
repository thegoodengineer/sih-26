// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the convex NLP engine (#226).
//
// Closed forms first: entropy on the simplex and a log barrier, whose optima are known
// exactly. Then the agreement the issue asks for - a QP handed over as an EXPRESSION reaches
// the objective the same QP reaches through the Q matrix - on random convex QPs, against both
// this engine's own Q path and the existing convex-QP engine. Then the refusals, because a
// local method that accepts a non-convex problem returns a local point labelled optimal.

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "nlp/convex_nlp.hpp"
#include "nlp/nonlinear_model.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::nlp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Options quiet() {
  Options o;
  o.set_bool("log_to_console", false);
  return o;
}

/// n continuous columns in [lower, upper], one row sum(x) in [row_lo, row_hi].
Model simplex(int n, double lower, double upper, double row_lo, double row_hi) {
  Model m;
  const auto u = static_cast<std::size_t>(n);
  m.col_cost.assign(u, 0.0);
  m.col_lower.assign(u, lower);
  m.col_upper.assign(u, upper);
  m.col_type.assign(u, VarType::kContinuous);
  m.matrix.reset(1, n);
  for (Index j = 0; j < n; ++j) m.matrix.add_entry(0, j, 1.0);
  m.matrix.finalize();
  m.row_lower = {row_lo};
  m.row_upper = {row_hi};
  m.hessian.reset(n, n);
  m.hessian.finalize();
  return m;
}

TEST(ConvexNlp, EntropyOnTheSimplexReachesTheUniformPoint) {
  // min sum x_j log x_j  s.t.  sum x_j = 1, x >= 0:  x_j = 1/n, value -log n. x log x is
  // convex but a product of two non-constants, which the composition rules do not prove, so
  // the caller asserts it - the route #226 describes - and the message says so.
  const int n = 5;
  NonlinearModel model(simplex(n, 1e-12, kInf, 1.0, 1.0));
  std::vector<ExprId> terms;
  for (Index j = 0; j < n; ++j) {
    const ExprId x = model.graph.variable(j);
    terms.push_back(model.graph.multiply(x, model.graph.log(x)));
  }
  model.objective = model.graph.sum(terms);

  Options options = quiet();
  const Solution refused = solve_convex_nlp(model, options);
  EXPECT_EQ(refused.status, SolveStatus::kNotSolved) << "not proved convex, and not asserted";

  options.set_bool("nlp_assume_convex", true);
  const Solution solved = solve_convex_nlp(model, options);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, -std::log(static_cast<double>(n)), 1e-5);
  for (const double v : solved.col_value) EXPECT_NEAR(v, 1.0 / n, 1e-4);
  EXPECT_NE(solved.message.find("ASSERTED"), std::string::npos)
      << "the answer must say it rests on the caller's word: " << solved.message;
}

TEST(ConvexNlp, AProvablyConvexProblemNeedsNoAssertion) {
  // max sum log x_j  s.t.  sum x_j <= 1, x in [1e-6, 1]:  x_j = 1/n, value -n log n. Log is
  // concave and increasing on the positive box, so maximising the sum is a convex problem the
  // rules prove on their own.
  const int n = 4;
  Model base = simplex(n, 1e-6, 1.0, -kInf, 1.0);
  base.sense = ObjSense::kMaximize;
  NonlinearModel model(std::move(base));
  std::vector<ExprId> terms;
  for (Index j = 0; j < n; ++j) terms.push_back(model.graph.log(model.graph.variable(j)));
  model.objective = model.graph.sum(terms);
  ASSERT_TRUE(model.convexity().convex);

  const Solution solved = solve_convex_nlp(model, quiet());
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  EXPECT_NEAR(solved.objective, -n * std::log(static_cast<double>(n)), 1e-5);
  for (const double v : solved.col_value) EXPECT_NEAR(v, 1.0 / n, 1e-4);
  EXPECT_NE(solved.message.find("proved"), std::string::npos) << solved.message;
}

/// A random convex QP: Q = M'M + 0.1 I held as a lower triangle, one or two rows.
Model random_qp(std::mt19937* rng, int n) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<std::vector<double>> mm(static_cast<std::size_t>(n),
                                      std::vector<double>(static_cast<std::size_t>(n)));
  for (auto& row : mm)
    for (double& v : row) v = u(*rng);
  Model m = simplex(n, -2.0, 2.0, -kInf, 1.0);
  for (double& c : m.col_cost) c = u(*rng);
  m.hessian.reset(n, n);
  for (Index j = 0; j < n; ++j) {
    for (Index i = j; i < n; ++i) {
      double q = i == j ? 0.1 : 0.0;
      for (int k = 0; k < n; ++k) {
        q += mm[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] *
             mm[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
      }
      m.hessian.add_entry(i, j, q);
    }
  }
  m.hessian.finalize();
  return m;
}

TEST(ConvexNlp, AQpAsAnExpressionMatchesTheQpThroughTheQMatrix) {
  // The same convex QP three ways: the existing convex-QP engine on the Q matrix, this
  // engine on the Q matrix, and this engine on 0.5 x'Qx written out as an expression. All
  // three optimise the same function, so they must agree - to the engines' tolerance, since
  // all three are first-order methods; there is no exact reference here to claim more.
  std::mt19937 rng(226);
  for (int trial = 0; trial < 20; ++trial) {
    const int n = 3 + trial % 4;
    const Model qp = random_qp(&rng, n);

    Options qp_options = quiet();
    qp_options.set_double("qp_tolerance", 1e-8);
    const Solution reference = solve(qp, qp_options);
    ASSERT_EQ(reference.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << reference.message;

    Options nlp_options = quiet();
    nlp_options.set_double("nlp_tolerance", 1e-8);
    const Solution through_q = solve_convex_nlp(NonlinearModel(qp), nlp_options);
    ASSERT_EQ(through_q.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << through_q.message;

    Model linear = qp;
    linear.hessian.reset(n, n);
    linear.hessian.finalize();
    NonlinearModel as_expression(std::move(linear));
    ExpressionGraph& g = as_expression.graph;
    std::vector<ExprId> terms;
    for (Index j = 0; j < n; ++j) {
      const ColumnView col = qp.hessian.column(j);
      for (Index k = 0; k < col.size; ++k) {
        const Index i = col.rows[k];
        // 0.5 x'Qx over the lower triangle: q_jj x_j^2 / 2 on the diagonal, q_ij x_i x_j off
        // it.
        const double weight = i == j ? 0.5 * col.values[k] : col.values[k];
        terms.push_back(
            g.multiply(g.constant(weight), g.multiply(g.variable(i), g.variable(j))));
      }
    }
    as_expression.objective = g.sum(terms);
    nlp_options.set_bool("nlp_assume_convex", true);  // cross terms are not DCP-provable
    const Solution through_expression = solve_convex_nlp(as_expression, nlp_options);
    ASSERT_EQ(through_expression.status, SolveStatus::kOptimal)
        << "trial " << trial << ": " << through_expression.message;

    const double scale = 1.0 + std::fabs(reference.objective);
    EXPECT_NEAR(through_q.objective, reference.objective, 1e-6 * scale) << "trial " << trial;
    EXPECT_NEAR(through_expression.objective, through_q.objective, 1e-6 * scale)
        << "trial " << trial;
  }
}

TEST(ConvexNlp, WhatItDoesNotTakeIsRefusedWithTheReason) {
  {
    NonlinearModel model(simplex(2, 0.0, 1.0, -kInf, 1.0));
    model.objective = model.graph.multiply(model.graph.variable(0), model.graph.variable(1));
    const Solution s = solve_convex_nlp(model, quiet());
    EXPECT_EQ(s.status, SolveStatus::kNotSolved);
    EXPECT_NE(s.message.find("convexity could not be proved"), std::string::npos) << s.message;
  }
  {
    Model base = simplex(2, 0.0, 1.0, -kInf, 1.0);
    base.col_type[0] = VarType::kInteger;
    NonlinearModel model(std::move(base));
    model.objective = model.graph.power(model.graph.variable(0), 2.0);
    const Solution s = solve_convex_nlp(model, quiet());
    EXPECT_EQ(s.status, SolveStatus::kNotSolved);
    EXPECT_NE(s.message.find("integer"), std::string::npos) << s.message;
  }
  {
    NonlinearModel model(simplex(2, 0.5, 1.0, -kInf, 2.0));
    model.objective = model.graph.power(model.graph.variable(0), 2.0);
    model.constraints.push_back({model.graph.log(model.graph.variable(1)), 0.0, kInf, "log"});
    const Solution s = solve_convex_nlp(model, quiet());
    EXPECT_EQ(s.status, SolveStatus::kNotSolved);
    EXPECT_NE(s.message.find("nonlinear constraints"), std::string::npos) << s.message;
  }
  {
    // log over a box that reaches 0: the rules cannot prove it, so it is refused rather than
    // run into its own domain.
    Model base = simplex(1, 0.0, 1.0, -kInf, kInf);
    base.sense = ObjSense::kMaximize;
    NonlinearModel model(std::move(base));
    model.objective = model.graph.log(model.graph.variable(0));
    EXPECT_EQ(solve_convex_nlp(model, quiet()).status, SolveStatus::kNotSolved);
  }
}

TEST(ConvexNlp, AnUnconvergedPointIsNotCalledOptimal) {
  const int n = 5;
  NonlinearModel model(simplex(n, 1e-12, kInf, 1.0, 1.0));
  std::vector<ExprId> terms;
  for (Index j = 0; j < n; ++j) {
    const ExprId x = model.graph.variable(j);
    terms.push_back(model.graph.multiply(x, model.graph.log(x)));
  }
  model.objective = model.graph.sum(terms);
  Options options = quiet();
  options.set_bool("nlp_assume_convex", true);
  options.set_int("iteration_limit", 3);
  const Solution s = solve_convex_nlp(model, options);
  EXPECT_NE(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_NE(s.message.find("not certified optimal"), std::string::npos) << s.message;
}

TEST(ConvexNlp, AFeasibleButNotStationaryPointIsNotCalledOptimal) {
  // min sum (x_j - 3)^2 over the box [0, 10]: every iterate is feasible - the projection sees
  // to that - so only the STATIONARITY measure can tell an unconverged point from the answer.
  // Mutation testing found the test above never reached it: stopped after three iterations
  // its point was still primal infeasible, so certifying on feasibility alone passed it too.
  const int n = 3;
  NonlinearModel model(simplex(n, 0.0, 10.0, -kInf, kInf));
  std::vector<ExprId> terms;
  for (Index j = 0; j < n; ++j) {
    terms.push_back(model.graph.power(
        model.graph.subtract(model.graph.variable(j), model.graph.constant(3.0)), 2.0));
  }
  model.objective = model.graph.sum(terms);
  Options options = quiet();
  options.set_int("iteration_limit", 1);
  const Solution early = solve_convex_nlp(model, options);
  EXPECT_LE(early.primal_infeasibility, 1e-12) << "the point is feasible";
  EXPECT_NE(early.status, SolveStatus::kOptimal) << early.message;

  options.set_int("iteration_limit", -1);
  const Solution solved = solve_convex_nlp(model, options);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  for (const double v : solved.col_value) EXPECT_NEAR(v, 3.0, 1e-5);
}

}  // namespace
}  // namespace sankhya::nlp

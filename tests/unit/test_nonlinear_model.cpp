// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convexity rules and the nonlinear model's classification (#296).

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "nlp/expression.hpp"
#include "nlp/nonlinear_model.hpp"

namespace sankhya::nlp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ---- Curvature, case by case -------------------------------------------------------------

TEST(Convexity, TheCompositionRulesOnTheCasesTheyAreFor) {
  ExpressionGraph g(2);
  const ExprId x = g.variable(0);
  const ExprId y = g.variable(1);
  const std::vector<double> all{-kInf, -kInf};
  const std::vector<double> none{kInf, kInf};
  const std::vector<double> pos_lo{1.0, 1.0};
  const std::vector<double> pos_hi{2.0, 2.0};
  const auto c = [&](ExprId f, const std::vector<double>& lo, const std::vector<double>& hi) {
    return g.curvature(f, lo, hi);
  };
  EXPECT_EQ(c(g.power(x, 2.0), all, none), Curvature::kConvex);
  EXPECT_EQ(c(g.multiply(x, x), all, none), Curvature::kConvex);
  EXPECT_EQ(c(g.exp(x), all, none), Curvature::kConvex);
  EXPECT_EQ(c(g.exp(g.power(x, 2.0)), all, none), Curvature::kConvex) << "exp is increasing";
  EXPECT_EQ(c(g.exp(g.negate(x)), all, none), Curvature::kConvex) << "affine argument";
  EXPECT_EQ(c(g.power(g.add(x, y), 2.0), all, none), Curvature::kConvex);
  EXPECT_EQ(c(g.add(g.power(x, 2.0), g.exp(y)), all, none), Curvature::kConvex);
  EXPECT_EQ(c(g.log(x), pos_lo, pos_hi), Curvature::kConcave);
  EXPECT_EQ(c(g.negate(g.log(x)), pos_lo, pos_hi), Curvature::kConvex);
  EXPECT_EQ(c(g.add(g.sqrt(x), g.log(y)), pos_lo, pos_hi), Curvature::kConcave);
  EXPECT_EQ(c(g.divide(g.constant(1.0), x), pos_lo, pos_hi), Curvature::kConvex);
  EXPECT_EQ(c(g.divide(g.constant(1.0), x), {-2.0, 0.0}, {-1.0, 0.0}), Curvature::kConcave);
  EXPECT_EQ(c(g.power(x, 3.0), {0.0, 0.0}, none), Curvature::kConvex);
  EXPECT_EQ(c(g.power(x, 3.0), all, {0.0, 0.0}), Curvature::kConcave);
  EXPECT_EQ(c(g.multiply(g.constant(-2.0), g.power(x, 2.0)), all, none), Curvature::kConcave);
  EXPECT_EQ(c(g.add(g.multiply(g.constant(3.0), x), g.constant(1.0)), all, none),
            Curvature::kAffine);
}

TEST(Convexity, UnknownWhereTheRulesDoNotReach) {
  ExpressionGraph g(2);
  const ExprId x = g.variable(0);
  const ExprId y = g.variable(1);
  const std::vector<double> all{-kInf, -kInf};
  const std::vector<double> none{kInf, kInf};
  // Genuinely not convex:
  EXPECT_EQ(g.curvature(g.multiply(x, y), all, none), Curvature::kUnknown);
  EXPECT_EQ(g.curvature(g.subtract(g.power(x, 2.0), g.power(y, 2.0)), all, none),
            Curvature::kUnknown);
  EXPECT_EQ(g.curvature(g.power(x, 3.0), {-1.0, 0.0}, {1.0, 0.0}), Curvature::kUnknown);
  // Domain the box can leave:
  EXPECT_EQ(g.curvature(g.log(x), {-1.0, 0.0}, {1.0, 0.0}), Curvature::kUnknown);
  EXPECT_EQ(g.curvature(g.divide(g.constant(1.0), x), {-1.0, 0.0}, {1.0, 0.0}),
            Curvature::kUnknown);
  // Convex in truth, but not by these rules - conservative, never guessed:
  EXPECT_EQ(g.curvature(g.log(g.exp(x)), all, none), Curvature::kUnknown);
  EXPECT_EQ(g.curvature(g.sqrt(g.power(x, 2.0)), all, none), Curvature::kUnknown);
}

/// A random expression over two columns, from every operation, domain kept inside by
/// squaring-and-shifting where an argument must be positive.
ExprId random_expression(ExpressionGraph* g, std::mt19937* rng, int depth) {
  std::uniform_int_distribution<int> pick(0, 11);
  std::uniform_int_distribution<Index> column(0, 1);
  std::uniform_real_distribution<double> c(-2.0, 2.0);
  if (depth == 0) return pick(*rng) < 8 ? g->variable(column(*rng)) : g->constant(c(*rng));
  const auto sub = [&] { return random_expression(g, rng, depth - 1); };
  switch (pick(*rng)) {
    case 0:
    case 1: return g->add(sub(), sub());
    case 2: return g->multiply(g->constant(c(*rng)), sub());
    case 3: return g->multiply(sub(), sub());
    case 4: return g->power(sub(), 2.0);
    case 5: return g->power(sub(), 3.0);
    case 6: return g->exp(sub());
    case 7: return g->log(sub());
    case 8: return g->sqrt(sub());
    case 9: return g->divide(g->constant(1.0), sub());
    case 10: return g->power(sub(), 1.5);
    default: return g->negate(sub());
  }
}

TEST(Convexity, EveryClaimSurvivesTheDefinitionOnRandomExpressions) {
  // The soundness test. Whatever the rules call convex must satisfy
  //   f((a + b) / 2) <= (f(a) + f(b)) / 2
  // at every pair of points in the box, and concave the reverse. A composition rule applied
  // with the wrong monotonicity, or a range computed too narrow, claims convexity for a
  // function that is not - and some random pair shows it.
  std::mt19937 rng(2006);
  int claims = 0;
  int pairs = 0;
  for (int trial = 0; trial < 3000; ++trial) {
    ExpressionGraph g(2);
    const ExprId f = random_expression(&g, &rng, 3);
    std::uniform_real_distribution<double> corner(-3.0, 3.0);
    double a0 = corner(rng);
    double a1 = corner(rng);
    const std::vector<double> lo{std::min(a0, a0 + 1.0), std::min(a1, a1 + 1.0)};
    const std::vector<double> hi{lo[0] + 1.5, lo[1] + 1.5};
    const Curvature claim = g.curvature(f, lo, hi);
    if (claim != Curvature::kConvex && claim != Curvature::kConcave) continue;
    ++claims;
    std::uniform_real_distribution<double> px(lo[0], hi[0]);
    std::uniform_real_distribution<double> py(lo[1], hi[1]);
    for (int k = 0; k < 20; ++k) {
      const std::vector<double> p{px(rng), py(rng)};
      const std::vector<double> q{px(rng), py(rng)};
      const std::vector<double> m{(p[0] + q[0]) / 2, (p[1] + q[1]) / 2};
      const Evaluation fp = g.evaluate(f, p);
      const Evaluation fq = g.evaluate(f, q);
      const Evaluation fm = g.evaluate(f, m);
      // A claim of curvature over the box is also a claim that f is DEFINED there: a domain
      // error is a false claim. Overflow is not - exp(exp(x)^2) is defined and convex
      // everywhere, and simply leaves double range at x = 3 - so those pairs are skipped.
      for (const Evaluation* e : {&fp, &fq, &fm}) {
        ASSERT_NE(e->error, EvalError::kDomain)
            << "claimed " << to_string(claim) << " but undefined in the box: " << g.to_string(f)
            << " (" << e->message << ")";
      }
      if (!fp.ok() || !fq.ok() || !fm.ok() || std::fabs(fp.value) > 1e12) break;
      ++pairs;
      const double chord = (fp.value + fq.value) / 2;
      const double slack = 1e-9 * (1.0 + std::fabs(chord));
      if (claim == Curvature::kConvex) {
        EXPECT_LE(fm.value, chord + slack) << "claimed convex: " << g.to_string(f);
      } else {
        EXPECT_GE(fm.value, chord - slack) << "claimed concave: " << g.to_string(f);
      }
    }
  }
  EXPECT_GT(claims, 300) << "the generator should produce plenty of claims to test";
  EXPECT_GT(pairs, 5000);
}

// ---- The model ---------------------------------------------------------------------------

Model columns(int n, bool integer = false) {
  Model m;
  const auto u = static_cast<std::size_t>(n);
  m.col_cost.assign(u, 0.0);
  m.col_lower.assign(u, 1.0);
  m.col_upper.assign(u, 3.0);
  m.col_type.assign(u, integer ? VarType::kInteger : VarType::kContinuous);
  m.matrix.reset(0, n);
  m.matrix.finalize();
  m.hessian.reset(n, n);
  m.hessian.finalize();
  return m;
}

TEST(NonlinearModel, ClassifiesFromTheStructure) {
  {
    NonlinearModel m(columns(2));
    EXPECT_EQ(m.classify(), ProblemClass::kLp);
  }
  {
    NonlinearModel m(columns(2, true));
    EXPECT_EQ(m.classify(), ProblemClass::kMilp);
  }
  {
    NonlinearModel m(columns(2));
    m.objective = m.graph.power(m.graph.variable(0), 2.0);
    EXPECT_EQ(m.classify(), ProblemClass::kQp) << "a quadratic objective expression";
  }
  {
    NonlinearModel m(columns(2));
    m.objective = m.graph.log(m.graph.variable(0));
    EXPECT_EQ(m.classify(), ProblemClass::kNlp);
  }
  {
    NonlinearModel m(columns(2, true));
    m.objective = m.graph.exp(m.graph.variable(1));
    EXPECT_EQ(m.classify(), ProblemClass::kMinlp);
  }
  {
    NonlinearModel m(columns(2));
    // A "nonlinear" constraint of degree 1 is linear: 2 x0 + x1 <= 3.
    m.constraints.push_back(
        {m.graph.add(m.graph.multiply(m.graph.constant(2.0), m.graph.variable(0)),
                     m.graph.variable(1)),
         -kInf, 3.0, "linear"});
    EXPECT_EQ(m.classify(), ProblemClass::kLp);
    // A quadratic constraint is not something the QP engines take.
    m.constraints.push_back({m.graph.power(m.graph.variable(0), 2.0), -kInf, 4.0, "ball"});
    EXPECT_EQ(m.classify(), ProblemClass::kNlp);
  }
}

TEST(NonlinearModel, ConvexityNamesThePartThatFails) {
  NonlinearModel m(columns(2));
  ExpressionGraph& g = m.graph;
  const ExprId x = g.variable(0);
  const ExprId y = g.variable(1);
  m.objective = g.add(g.power(x, 2.0), g.exp(y));
  m.constraints.push_back({g.log(x), 0.1, kInf, "log floor"});  // concave >= : convex set
  EXPECT_TRUE(m.convexity().convex);

  m.base.sense = ObjSense::kMaximize;  // maximising a convex function: not a convex problem
  ConvexityReport r = m.convexity();
  EXPECT_FALSE(r.convex);
  ASSERT_EQ(r.reasons.size(), 1u);
  EXPECT_NE(r.reasons[0].find("maximising"), std::string::npos) << r.reasons[0];

  m.base.sense = ObjSense::kMinimize;
  m.constraints.push_back({g.power(x, 2.0), 2.0, kInf, "outside a ball"});  // convex >= : not
  r = m.convexity();
  EXPECT_FALSE(r.convex);
  ASSERT_EQ(r.reasons.size(), 1u);
  EXPECT_NE(r.reasons[0].find("outside a ball"), std::string::npos) << r.reasons[0];
}

TEST(NonlinearModel, ValidationCatchesWhatWouldOtherwiseEvaluateWrongly) {
  NonlinearModel m(columns(2));
  EXPECT_TRUE(m.validate().empty()) << m.validate();
  m.constraints.push_back({m.graph.variable(0), 3.0, 1.0, "crossed"});
  EXPECT_NE(m.validate().find("crossed"), std::string::npos) << m.validate();
  m.constraints.clear();
  m.objective = 12345;
  EXPECT_NE(m.validate().find("12345"), std::string::npos) << m.validate();
  m.objective = m.graph.variable(7);
  EXPECT_NE(m.validate().find("column 7"), std::string::npos) << m.validate();
}

TEST(NonlinearModel, EvaluationSeparatesADomainFailureFromAViolation) {
  Model base = columns(2);
  base.col_cost = {1.0, 0.0};
  base.col_lower = {-5.0, -5.0};
  NonlinearModel m(std::move(base));
  m.objective = m.graph.power(m.graph.variable(1), 2.0);
  m.constraints.push_back({m.graph.log(m.graph.variable(0)), 0.0, kInf, "log"});

  PointReport inside = m.evaluate({2.0, 3.0});
  ASSERT_TRUE(inside.failure.ok());
  EXPECT_DOUBLE_EQ(inside.objective, 2.0 + 9.0);
  EXPECT_DOUBLE_EQ(inside.worst_nonlinear_violation, 0.0);

  PointReport violated = m.evaluate({0.5, 0.0});  // log(0.5) < 0: violates, but is defined
  ASSERT_TRUE(violated.failure.ok());
  EXPECT_NEAR(violated.worst_nonlinear_violation, -std::log(0.5), 1e-15);

  PointReport undefined = m.evaluate({-1.0, 0.0});  // log(-1): not a violation, no value
  EXPECT_EQ(undefined.failure.error, EvalError::kDomain) << undefined.failure.message;

  EXPECT_FALSE(m.domain_risks().empty()) << "x0 can go to -5, and log needs it positive";
}

}  // namespace
}  // namespace sankhya::nlp

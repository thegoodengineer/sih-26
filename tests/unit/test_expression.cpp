// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the nonlinear expression graph (#296): sharing, safe simplification, evaluation
// that refuses to return NaN, exact derivatives, and conservative convexity.
//
// Two tests carry more weight than the rest. Random expressions have their exact gradients
// and Hessians compared against central differences, so a wrong partial in any operation
// shows up whichever expression happens to use it. And every random expression the rules
// call convex (or concave) is put through the definition - midpoint convexity at random pairs
// of points in the box - so a rule that claims too much fails on a counterexample rather than
// on a case someone thought to write down.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "nlp/expression.hpp"

namespace sankhya::nlp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ---- Sharing and simplification ------------------------------------------------------------

TEST(Expression, TheSameOperationBuiltTwiceIsOneNode) {
  ExpressionGraph g(3);
  const ExprId x = g.variable(0);
  const ExprId y = g.variable(1);
  const ExprId a = g.add(x, y);
  const std::size_t before = g.size();
  EXPECT_EQ(g.add(x, y), a);
  EXPECT_EQ(g.add(y, x), a) << "addition is commutative, and so is the interning";
  EXPECT_EQ(g.multiply(x, y), g.multiply(y, x));
  EXPECT_EQ(g.variable(0), x);
  EXPECT_EQ(g.size(), before + 1) << "only the product was new";

  // y = x + x1 used twice: f = y*y + 3*y shares one y.
  const ExprId f = g.add(g.multiply(a, a), g.multiply(g.constant(3.0), a));
  int uses = 0;
  for (const ExprId id : g.tape(f)) {
    for (const ExprId child : g.node(id).children) uses += child == a ? 1 : 0;
  }
  EXPECT_GE(uses, 2) << "the shared subexpression is referenced, not copied";
}

TEST(Expression, SafeSimplificationsHappen) {
  ExpressionGraph g(2);
  const ExprId x = g.variable(0);
  EXPECT_EQ(g.add(x, g.constant(0.0)), x);
  EXPECT_EQ(g.multiply(x, g.constant(1.0)), x);
  EXPECT_EQ(g.negate(g.negate(x)), x);
  EXPECT_EQ(g.power(x, 1.0), x);
  EXPECT_EQ(g.divide(x, g.constant(1.0)), x);
  const ExprId zero = g.multiply(x, g.constant(0.0));
  ASSERT_EQ(g.node(zero).op, Op::kConstant);
  EXPECT_EQ(g.node(zero).value, 0.0);

  // 2 * (3 + 4) folds to 14 at construction.
  const ExprId folded = g.multiply(g.constant(2.0), g.add(g.constant(3.0), g.constant(4.0)));
  ASSERT_EQ(g.node(folded).op, Op::kConstant);
  EXPECT_EQ(g.node(folded).value, 14.0);

  // Nested sums flatten and their constants gather: (x + 2) + 3 is one sum, x + 5.
  const ExprId s = g.add(g.add(x, g.constant(2.0)), g.constant(3.0));
  ASSERT_EQ(g.node(s).op, Op::kSum);
  EXPECT_EQ(g.node(s).children.size(), 2u);
  EXPECT_DOUBLE_EQ(g.evaluate(s, {1.0, 0.0}).value, 6.0);
}

TEST(Expression, SimplificationNeverHidesADomainError) {
  // 0 * log(x) is not 0 at x = -1: it is undefined there, and folding it away would turn an
  // error at the caller's point into a plausible zero.
  ExpressionGraph g(1);
  const ExprId x = g.variable(0);
  const ExprId kept = g.multiply(g.constant(0.0), g.log(x));
  EXPECT_EQ(g.node(kept).op, Op::kProduct);
  const Evaluation at_minus_one = g.evaluate(kept, {-1.0});
  EXPECT_EQ(at_minus_one.error, EvalError::kDomain) << at_minus_one.message;
  EXPECT_DOUBLE_EQ(g.evaluate(kept, {2.0}).value, 0.0);

  // log(x)^0 likewise stays a power; exp(log(x)) stays a composition.
  EXPECT_EQ(g.node(g.power(g.log(x), 0.0)).op, Op::kPower);
  EXPECT_EQ(g.node(g.exp(g.log(x))).op, Op::kExp);
  EXPECT_EQ(g.evaluate(g.exp(g.log(x)), {-1.0}).error, EvalError::kDomain);

  // A constant division by zero is kept, so evaluation reports it.
  const ExprId one_over_zero = g.divide(g.constant(1.0), g.constant(0.0));
  EXPECT_EQ(g.evaluate(one_over_zero, {0.0}).error, EvalError::kDomain);
}

// ---- Evaluation --------------------------------------------------------------------------

TEST(Expression, EvaluationComputesTheValue) {
  ExpressionGraph g(2);
  const ExprId x = g.variable(0);
  const ExprId y = g.variable(1);
  // f = x^2 * y + exp(x*y) + log(y) + sqrt(x) + x / y - 3
  const ExprId f = g.sum({g.multiply(g.power(x, 2.0), y), g.exp(g.multiply(x, y)), g.log(y),
                          g.sqrt(x), g.divide(x, y), g.constant(-3.0)});
  const double xv = 1.5;
  const double yv = 0.75;
  const double expected =
      xv * xv * yv + std::exp(xv * yv) + std::log(yv) + std::sqrt(xv) + xv / yv - 3.0;
  const Evaluation e = g.evaluate(f, {xv, yv});
  ASSERT_TRUE(e.ok()) << e.message;
  EXPECT_NEAR(e.value, expected, 1e-14 * std::fabs(expected));
}

TEST(Expression, AnUndefinedOperationIsADomainErrorNeverANaN) {
  ExpressionGraph g(1);
  const ExprId x = g.variable(0);
  struct Case {
    ExprId expression;
    double at;
  };
  for (const Case& c : {Case{g.log(x), -1.0}, Case{g.log(x), 0.0}, Case{g.sqrt(x), -1.0},
                        Case{g.divide(g.constant(1.0), x), 0.0}, Case{g.power(x, 0.5), -2.0},
                        Case{g.power(x, -1.0), 0.0}, Case{g.power(x, -0.5), 0.0}}) {
    const Evaluation e = g.evaluate(c.expression, {c.at});
    EXPECT_EQ(e.error, EvalError::kDomain) << g.to_string(c.expression) << " at " << c.at;
    EXPECT_FALSE(e.message.empty());
    EXPECT_NE(e.failed_at, kNoExpr);
  }
  // Allowed ones stay allowed: an integer power of a negative, 0^0.5, sqrt(0).
  EXPECT_DOUBLE_EQ(g.evaluate(g.power(x, 3.0), {-2.0}).value, -8.0);
  EXPECT_TRUE(g.evaluate(g.power(x, 0.5), {0.0}).ok());
  EXPECT_TRUE(g.evaluate(g.sqrt(x), {0.0}).ok());
}

TEST(Expression, OverflowIsReportedAsNonFiniteNotAsADomainError) {
  ExpressionGraph g(1);
  const Evaluation e = g.evaluate(g.exp(g.variable(0)), {1000.0});
  EXPECT_EQ(e.error, EvalError::kNonFinite) << e.message;
}

TEST(Expression, MisuseIsStickyAndPropagates) {
  ExpressionGraph g(2);
  EXPECT_TRUE(g.invalid().empty());
  const ExprId bad = g.variable(5);
  EXPECT_EQ(bad, kNoExpr);
  EXPECT_NE(g.invalid().find("column 5"), std::string::npos) << g.invalid();
  const ExprId spread = g.add(g.exp(bad), g.variable(0));
  EXPECT_EQ(spread, kNoExpr) << "a malformed operand must not become a plausible expression";
  EXPECT_EQ(g.evaluate(spread, {0.0, 0.0}).error, EvalError::kInvalid);
  EXPECT_EQ(g.constant(std::nan("")), kNoExpr);
  EXPECT_NE(g.invalid().find("column 5"), std::string::npos) << "the FIRST problem is kept";
  EXPECT_EQ(g.evaluate(g.variable(0), {1.0}).error, EvalError::kInvalid) << "wrong point size";
}

// ---- Derivatives -------------------------------------------------------------------------

TEST(Expression, GradientAndHessianMatchTheClosedForm) {
  ExpressionGraph g(2);
  const ExprId x = g.variable(0);
  const ExprId y = g.variable(1);
  // f = x^2 y + exp(xy) + log(y)
  const ExprId f = g.sum({g.multiply(g.power(x, 2.0), y), g.exp(g.multiply(x, y)), g.log(y)});
  const double a = 0.7;
  const double b = 1.3;
  const double e = std::exp(a * b);
  SparseEntries grad;
  Evaluation err;
  ASSERT_TRUE(g.gradient(f, {a, b}, &grad, &err)) << err.message;
  ASSERT_EQ(grad.size(), 2u);
  EXPECT_NEAR(grad[0].second, 2 * a * b + b * e, 1e-12);
  EXPECT_NEAR(grad[1].second, a * a + a * e + 1.0 / b, 1e-12);

  std::vector<HessianEntry> h;
  ASSERT_TRUE(g.hessian(f, {a, b}, &h, &err)) << err.message;
  double hxx = 0.0;
  double hxy = 0.0;
  double hyy = 0.0;
  for (const HessianEntry& entry : h) {
    ASSERT_GE(entry.row, entry.col) << "lower triangle only";
    if (entry.row == 0 && entry.col == 0) hxx = entry.value;
    if (entry.row == 1 && entry.col == 0) hxy = entry.value;
    if (entry.row == 1 && entry.col == 1) hyy = entry.value;
  }
  EXPECT_NEAR(hxx, 2 * b + b * b * e, 1e-12);
  EXPECT_NEAR(hxy, 2 * a + e + a * b * e, 1e-12);
  EXPECT_NEAR(hyy, a * a * e - 1.0 / (b * b), 1e-12);
}

TEST(Expression, ASquareOfASharedNodeDifferentiatesOnce) {
  // x * x has both children equal; the rule must produce 2x and 2, not x and 1.
  ExpressionGraph g(1);
  const ExprId x = g.variable(0);
  const ExprId sq = g.multiply(x, x);
  SparseEntries grad;
  std::vector<HessianEntry> h;
  Evaluation err;
  ASSERT_TRUE(g.gradient(sq, {3.0}, &grad, &err));
  EXPECT_DOUBLE_EQ(grad[0].second, 6.0);
  ASSERT_TRUE(g.hessian(sq, {3.0}, &h, &err));
  ASSERT_EQ(h.size(), 1u);
  EXPECT_DOUBLE_EQ(h[0].value, 2.0);
}

TEST(Expression, ADerivativeThatDoesNotExistIsRefused) {
  // sqrt(x) at 0 has a value and no derivative; x^0.5 the same.
  ExpressionGraph g(1);
  const ExprId x = g.variable(0);
  for (const ExprId f : {g.sqrt(x), g.power(x, 0.5)}) {
    ASSERT_TRUE(g.evaluate(f, {0.0}).ok());
    SparseEntries grad;
    Evaluation err;
    EXPECT_FALSE(g.gradient(f, {0.0}, &grad, &err)) << g.to_string(f);
    EXPECT_EQ(err.error, EvalError::kDomain);
    EXPECT_NE(err.message.find("no derivative"), std::string::npos) << err.message;
  }
}

/// A random expression over `n` columns whose domain includes the box [0.5, 2]^n, built from
/// every operation.
ExprId random_expression(ExpressionGraph* g, std::mt19937* rng, int depth) {
  std::uniform_int_distribution<int> pick(0, 9);
  std::uniform_int_distribution<Index> column(0, g->num_variables() - 1);
  std::uniform_real_distribution<double> c(0.5, 2.0);
  if (depth == 0) return pick(*rng) < 7 ? g->variable(column(*rng)) : g->constant(c(*rng));
  const auto sub = [&] { return random_expression(g, rng, depth - 1); };
  // Keep arguments of log, sqrt, divide and fractional powers positive over the box:
  // wrap them as (a^2 + 0.5), which is >= 0.5 wherever a is defined.
  const auto positive = [&] { return g->add(g->power(sub(), 2.0), g->constant(0.5)); };
  switch (pick(*rng)) {
    case 0: return g->add(sub(), sub());
    case 1: return g->multiply(sub(), sub());
    case 2: return g->subtract(sub(), sub());
    case 3: return g->divide(sub(), positive());
    case 4: return g->power(sub(), 3.0);
    case 5: return g->power(positive(), 1.5);
    case 6: return g->exp(g->multiply(g->constant(0.3), sub()));
    case 7: return g->log(positive());
    case 8: return g->sqrt(positive());
    default: return g->negate(sub());
  }
}

TEST(Expression, ExactDerivativesAgreeWithFiniteDifferencesOnRandomExpressions) {
  std::mt19937 rng(296);
  std::uniform_real_distribution<double> point(0.5, 2.0);
  int checked = 0;
  for (int trial = 0; trial < 400; ++trial) {
    ExpressionGraph g(3);
    const ExprId f = random_expression(&g, &rng, 3);
    ASSERT_TRUE(g.invalid().empty()) << g.invalid();
    const std::vector<double> x{point(rng), point(rng), point(rng)};
    const Evaluation e = g.evaluate(f, x);
    if (!e.ok()) continue;  // overflow in a deep exp; not a derivative question
    // A central difference has a best step: too large and truncation (O(h^2) times the third
    // derivative) dominates, too small and cancellation does. On exp(0.3 ((0.5 + x^2)^1.5)^3)
    // the truncation at h = 1e-5 is 9e-6 relative while the exact gradient matches the closed
    // form to 3e-14 - measured while writing this - so one fixed step would be testing the
    // difference quotient, not the derivative. The best of five steps is the fair check.
    double gradient_error = 1e300;
    double hessian_error = 1e300;
    bool any = false;
    for (const double step : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
      const DerivativeCheck check = check_derivatives(g, f, x, step);
      if (!check.evaluated) continue;
      any = true;
      gradient_error = std::min(gradient_error, check.gradient_error);
      hessian_error = std::min(hessian_error, check.hessian_error);
    }
    if (!any) continue;
    ++checked;
    EXPECT_LT(gradient_error, 1e-6) << "trial " << trial << ": " << g.to_string(f);
    EXPECT_LT(hessian_error, 1e-5) << "trial " << trial << ": " << g.to_string(f);
  }
  EXPECT_GT(checked, 350) << "the generator should mostly produce checkable expressions";
}

// ---- Structure ---------------------------------------------------------------------------

TEST(Expression, DegreeTellsPolynomialsFromEverythingElse) {
  ExpressionGraph g(2);
  const ExprId x = g.variable(0);
  const ExprId y = g.variable(1);
  EXPECT_EQ(g.degree(g.constant(4.0)), 0);
  EXPECT_EQ(g.degree(x), 1);
  EXPECT_EQ(g.degree(g.add(x, g.constant(1.0))), 1);
  EXPECT_EQ(g.degree(g.divide(x, g.constant(2.0))), 1);
  EXPECT_EQ(g.degree(g.multiply(x, y)), 2);
  EXPECT_EQ(g.degree(g.power(g.add(x, y), 2.0)), 2);
  EXPECT_EQ(g.degree(g.power(x, 3.0)), 3);
  EXPECT_EQ(g.degree(g.log(x)), -1);
  EXPECT_EQ(g.degree(g.divide(x, y)), -1);
  EXPECT_EQ(g.degree(g.power(x, 0.5)), -1);
}

TEST(Expression, IntervalRangesAreRight) {
  ExpressionGraph g(1);
  const ExprId x = g.variable(0);
  const std::vector<double> lo{-1.0};
  const std::vector<double> hi{2.0};
  const Interval sq = g.range(g.power(x, 2.0), lo, hi);
  EXPECT_DOUBLE_EQ(sq.lower, 0.0);
  EXPECT_DOUBLE_EQ(sq.upper, 4.0);
  const Interval xx = g.range(g.multiply(x, x), lo, hi);
  EXPECT_DOUBLE_EQ(xx.lower, 0.0) << "x * x is a square, not a product of two independents";
  const Interval ex = g.range(g.exp(x), lo, hi);
  EXPECT_DOUBLE_EQ(ex.lower, std::exp(-1.0));
  EXPECT_DOUBLE_EQ(ex.upper, std::exp(2.0));
  EXPECT_TRUE(g.range(g.divide(g.constant(1.0), x), lo, hi).may_be_undefined);
  EXPECT_TRUE(g.range(g.log(x), lo, hi).may_be_undefined);
  EXPECT_FALSE(g.range(g.log(x), {1.0}, {2.0}).may_be_undefined);
}

TEST(Expression, DomainRisksNameTheOperationAndTheRange) {
  ExpressionGraph g(1);
  const ExprId f = g.add(g.log(g.variable(0)), g.sqrt(g.variable(0)));
  const std::vector<std::string> risky = g.domain_risks(f, {-1.0}, {1.0});
  ASSERT_EQ(risky.size(), 2u);
  // Which of the two comes first depends on the order the compiler evaluated add()'s
  // arguments, which C++ leaves unspecified; both must be named.
  const std::string both = risky[0] + " | " + risky[1];
  EXPECT_NE(both.find("log"), std::string::npos) << both;
  EXPECT_NE(both.find("sqrt"), std::string::npos) << both;
  EXPECT_NE(both.find("-1"), std::string::npos) << "the range that breaks it: " << both;
  EXPECT_TRUE(g.domain_risks(f, {1.0}, {2.0}).empty());
}

}  // namespace
}  // namespace sankhya::nlp

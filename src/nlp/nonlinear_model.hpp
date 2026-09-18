// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a model with nonlinear parts, and what class of problem it is (#296).
//
// THE FROZEN MODEL IS NOT CHANGED. A NonlinearModel HOLDS a sankhya::Model for everything the
// existing engines already understand - the columns with their bounds and integrality, the
// linear rows, the linear and quadratic objective - and adds an expression graph over the same
// columns for what they do not: a nonlinear objective term and nonlinear constraints. There is
// no second variable model to keep in step with the first (#296 item 5), and an LP or MILP
// never passes through an expression tree it does not need (item 1).
//
// NO ENGINE READS THIS YET. It is the representation #226 (a convex NLP engine behind the
// solve() seam) will consume; classifying a model as NLP says what it is, not that anything
// here can solve it.

#pragma once

#include <string>
#include <vector>

#include "nlp/expression.hpp"
#include "sankhya/model.hpp"

namespace sankhya::nlp {

/// lower <= g(x) <= upper, either side possibly infinite.
struct NonlinearConstraint {
  ExprId expression = kNoExpr;
  double lower = -kInfinity;
  double upper = kInfinity;
  std::string name;
};

enum class ProblemClass : std::uint8_t { kLp, kMilp, kQp, kMiqp, kNlp, kMinlp };
[[nodiscard]] const char* to_string(ProblemClass problem_class) noexcept;

/// Whether the continuous relaxation is a convex problem, and if not, which part stops it.
struct ConvexityReport {
  bool convex = false;
  /// One line per part that is not provably convex. Empty when `convex`.
  std::vector<std::string> reasons;
};

/// What a point does to the model: the objective, and how far it is from every constraint.
struct PointReport {
  double objective = 0.0;
  double worst_linear_violation = 0.0;
  double worst_nonlinear_violation = 0.0;
  /// kNone, or why some part could not be evaluated. A DOMAIN failure here is not
  /// infeasibility: log(x) at x = -1 is a point the model does not define, not one that
  /// violates it, and callers keep the two apart (#296 item 11).
  Evaluation failure;
};

class NonlinearModel {
 public:
  explicit NonlinearModel(Model base_model);

  /// Columns, bounds, integrality, linear rows, the linear and quadratic objective.
  Model base;
  /// Expressions over base's columns.
  ExpressionGraph graph;
  /// Added to base's objective. kNoExpr when the objective has no nonlinear part.
  ExprId objective = kNoExpr;
  std::vector<NonlinearConstraint> constraints;

  /// Empty when well formed, otherwise the first problem: base's own validation, a malformed
  /// expression, a constraint with lower > upper, an expression from nowhere.
  [[nodiscard]] std::string validate() const;

  /// From the structure, not from what the caller says it is. A nonlinear constraint of
  /// degree <= 1 is linear and does not make the model an NLP; a quadratic objective
  /// expression makes it a QP only when there are no nonlinear constraints (a quadratic
  /// CONSTRAINT is outside what the QP engines take, so it is NLP).
  [[nodiscard]] ProblemClass classify() const;

  /// Convexity of the continuous relaxation over the column bounds, by the documented
  /// composition rules for the expressions and by the sparse LDL^T test for base's Q.
  /// Conservative: `convex` is true only when every part is proved so.
  [[nodiscard]] ConvexityReport convexity() const;

  /// Restricted-domain operations the column bounds allow to be violated.
  [[nodiscard]] std::vector<std::string> domain_risks() const;

  [[nodiscard]] PointReport evaluate(const std::vector<double>& x) const;
};

}  // namespace sankhya::nlp

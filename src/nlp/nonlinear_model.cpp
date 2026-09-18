// SPDX-License-Identifier: Apache-2.0
// SANKHYA - nonlinear model validation, classification, convexity and evaluation (#296).

#include "nlp/nonlinear_model.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include <fmt/format.h>

#include "qp/convexity.hpp"

namespace sankhya::nlp {

const char* to_string(ProblemClass problem_class) noexcept {
  switch (problem_class) {
    case ProblemClass::kLp: return "LP";
    case ProblemClass::kMilp: return "MILP";
    case ProblemClass::kQp: return "QP";
    case ProblemClass::kMiqp: return "MIQP";
    case ProblemClass::kNlp: return "NLP";
    case ProblemClass::kMinlp: return "MINLP";
  }
  return "unknown";
}

NonlinearModel::NonlinearModel(Model base_model)
    : base(std::move(base_model)), graph(base.num_cols()) {}

std::string NonlinearModel::validate() const {
  std::string problem = base.validate();
  if (!problem.empty()) return problem;
  if (!graph.invalid().empty()) return "nonlinear expression: " + graph.invalid();
  if (graph.num_variables() != base.num_cols()) {
    return fmt::format("the expressions are over {} columns and the model has {}",
                       graph.num_variables(), base.num_cols());
  }
  if (objective != kNoExpr && !graph.contains(objective)) {
    return fmt::format("the nonlinear objective names node {}, which is not in the graph",
                       objective);
  }
  for (std::size_t k = 0; k < constraints.size(); ++k) {
    const NonlinearConstraint& c = constraints[k];
    const std::string label =
        c.name.empty() ? fmt::format("nonlinear constraint {}", k) : c.name;
    if (!graph.contains(c.expression)) {
      return fmt::format("{} names node {}, which is not in the graph", label, c.expression);
    }
    if (std::isnan(c.lower) || std::isnan(c.upper) || c.lower > c.upper) {
      return fmt::format("{} has lower bound {:g} above upper bound {:g}", label, c.lower,
                         c.upper);
    }
  }
  return {};
}

ProblemClass NonlinearModel::classify() const {
  const bool integral = base.has_integrality();
  bool nonlinear = false;
  bool quadratic = base.has_quadratic_objective();
  if (objective != kNoExpr) {
    const int d = graph.degree(objective);
    if (d < 0 || d > 2) {
      nonlinear = true;
    } else if (d == 2) {
      quadratic = true;
    }
  }
  for (const NonlinearConstraint& c : constraints) {
    const int d = graph.degree(c.expression);
    if (d < 0 || d > 1) nonlinear = true;
  }
  if (nonlinear) return integral ? ProblemClass::kMinlp : ProblemClass::kNlp;
  if (quadratic) return integral ? ProblemClass::kMiqp : ProblemClass::kQp;
  return integral ? ProblemClass::kMilp : ProblemClass::kLp;
}

ConvexityReport NonlinearModel::convexity() const {
  ConvexityReport report;
  const bool maximize = base.sense == ObjSense::kMaximize;
  // The objective to MINIMISE must be convex; maximising asks for a concave one.
  const Curvature wanted = maximize ? Curvature::kConcave : Curvature::kConvex;
  const auto fits = [](Curvature have, Curvature want) {
    return have == Curvature::kConstant || have == Curvature::kAffine || have == want;
  };

  if (base.has_quadratic_objective()) {
    const qp::ConvexityResult q = qp::check_convexity(base);
    if (q.verdict != qp::Convexity::kConvex) {
      report.reasons.push_back(
          fmt::format("the quadratic objective term is not proved convex: {}", q.detail));
    }
  }
  if (objective != kNoExpr) {
    const Curvature c = graph.curvature(objective, base.col_lower, base.col_upper);
    if (!fits(c, wanted)) {
      report.reasons.push_back(
          fmt::format("the nonlinear objective {} is {} over the bounds, "
                      "and {} needs it {}",
                      graph.to_string(objective), to_string(c),
                      maximize ? "maximising" : "minimising", to_string(wanted)));
    }
  }
  for (std::size_t k = 0; k < constraints.size(); ++k) {
    const NonlinearConstraint& con = constraints[k];
    const Curvature c = graph.curvature(con.expression, base.col_lower, base.col_upper);
    const std::string label =
        con.name.empty() ? fmt::format("nonlinear constraint {}", k) : con.name;
    // g <= u describes a convex set when g is convex; g >= l when g is concave; both sides
    // need both, which is only affine.
    if (std::isfinite(con.upper) && !fits(c, Curvature::kConvex)) {
      report.reasons.push_back(
          fmt::format("{}: {} <= {:g} needs a convex left side, and it is "
                      "{}",
                      label, graph.to_string(con.expression), con.upper, to_string(c)));
    }
    if (std::isfinite(con.lower) && !fits(c, Curvature::kConcave)) {
      report.reasons.push_back(
          fmt::format("{}: {} >= {:g} needs a concave left side, and it "
                      "is {}",
                      label, graph.to_string(con.expression), con.lower, to_string(c)));
    }
  }
  report.convex = report.reasons.empty();
  return report;
}

std::vector<std::string> NonlinearModel::domain_risks() const {
  std::vector<std::string> risks;
  const auto collect = [&](ExprId root, const std::string& where) {
    for (std::string& risk : graph.domain_risks(root, base.col_lower, base.col_upper)) {
      risks.push_back(where + ": " + std::move(risk));
    }
  };
  if (objective != kNoExpr) collect(objective, "objective");
  for (std::size_t k = 0; k < constraints.size(); ++k) {
    collect(constraints[k].expression, constraints[k].name.empty()
                                           ? fmt::format("nonlinear constraint {}", k)
                                           : constraints[k].name);
  }
  return risks;
}

PointReport NonlinearModel::evaluate(const std::vector<double>& x) const {
  PointReport report;
  if (x.size() != static_cast<std::size_t>(base.num_cols())) {
    report.failure.error = EvalError::kInvalid;
    report.failure.message =
        fmt::format("a point with {} entries for {} columns", x.size(), base.num_cols());
    return report;
  }
  report.objective = base.evaluate_objective(x.data());
  if (objective != kNoExpr) {
    const Evaluation e = graph.evaluate(objective, x);
    if (!e.ok()) {
      report.failure = e;
      return report;
    }
    report.objective += e.value;
  }
  // Linear rows, measured the way the rest of the project measures them: distance outside
  // [row_lower, row_upper] of the activity.
  std::vector<double> activity(static_cast<std::size_t>(base.num_rows()), 0.0);
  base.matrix.multiply_add(x.data(), activity.data());
  for (std::size_t i = 0; i < activity.size(); ++i) {
    const double below = base.row_lower[i] - activity[i];
    const double above = activity[i] - base.row_upper[i];
    report.worst_linear_violation = std::max({report.worst_linear_violation, below, above});
  }
  for (const NonlinearConstraint& c : constraints) {
    const Evaluation e = graph.evaluate(c.expression, x);
    if (!e.ok()) {
      report.failure = e;
      return report;
    }
    report.worst_nonlinear_violation =
        std::max({report.worst_nonlinear_violation, c.lower - e.value, e.value - c.upper});
  }
  return report;
}

}  // namespace sankhya::nlp

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - evaluation and exact derivatives of an expression graph (#296).
//
// One forward sweep computes every node's value, refusing an operation outside its domain
// instead of producing NaN. The gradient is one reverse sweep of adjoints (Griewank and
// Walther, "Evaluating Derivatives", 2nd ed., SIAM 2008, ch. 3). The Hessian is
// forward-over-reverse: for each column j the expression contains, a tangent sweep in the
// direction e_j followed by a reverse sweep that carries each adjoint's derivative along e_j,
// which yields column j of the Hessian exactly (same reference, ch. 5).

#include <algorithm>
#include <cmath>
#include <limits>

#include <fmt/format.h>

#include "nlp/expression.hpp"

namespace sankhya::nlp {
namespace {

bool is_integer(double value) {
  return std::isfinite(value) && std::floor(value) == value;
}

Evaluation failure(EvalError error, ExprId at, std::string message) {
  Evaluation result;
  result.error = error;
  result.failed_at = at;
  result.message = std::move(message);
  return result;
}

/// Every node's value, children first. Fills `values` (indexed by node id) for the ids on
/// the tape; returns the first failure, or an ok Evaluation holding the root's value.
Evaluation forward(const ExpressionGraph& graph, const std::vector<ExprId>& order,
                   const std::vector<double>& x, std::vector<double>* values) {
  values->assign(graph.size(), 0.0);
  std::vector<double>& v = *values;
  for (const ExprId id : order) {
    const Node& n = graph.node(id);
    const auto at = [&](std::size_t k) { return v[static_cast<std::size_t>(n.children[k])]; };
    double result = 0.0;
    switch (n.op) {
      case Op::kConstant: result = n.value; break;
      case Op::kVariable: result = x[static_cast<std::size_t>(n.variable)]; break;
      case Op::kSum:
        for (std::size_t k = 0; k < n.children.size(); ++k) result += at(k);
        break;
      case Op::kProduct: result = at(0) * at(1); break;
      case Op::kNegate: result = -at(0); break;
      case Op::kDivide:
        if (at(1) == 0.0) return failure(EvalError::kDomain, id, "division by zero");
        result = at(0) / at(1);
        break;
      case Op::kPower: {
        const double base = at(0);
        if (base < 0.0 && !is_integer(n.value)) {
          return failure(
              EvalError::kDomain, id,
              fmt::format("{:g} raised to the non-integer power {:g}", base, n.value));
        }
        if (base == 0.0 && n.value < 0.0) {
          return failure(EvalError::kDomain, id,
                         fmt::format("0 raised to the negative power {:g}", n.value));
        }
        result = std::pow(base, n.value);
        break;
      }
      case Op::kExp: result = std::exp(at(0)); break;
      case Op::kLog:
        if (at(0) <= 0.0) {
          return failure(EvalError::kDomain, id, fmt::format("log of {:g}", at(0)));
        }
        result = std::log(at(0));
        break;
      case Op::kSqrt:
        if (at(0) < 0.0) {
          return failure(EvalError::kDomain, id, fmt::format("sqrt of {:g}", at(0)));
        }
        result = std::sqrt(at(0));
        break;
    }
    if (!std::isfinite(result)) {
      return failure(EvalError::kNonFinite, id,
                     fmt::format("{} overflowed to {}", to_string(n.op), result));
    }
    v[static_cast<std::size_t>(id)] = result;
  }
  Evaluation ok;
  ok.value = order.empty() ? 0.0 : v[static_cast<std::size_t>(order.back())];
  return ok;
}

/// The partial derivative of node `id` with respect to its k-th child, and - when `tangent`
/// is given - that partial's own derivative along the direction whose node tangents it holds.
/// Returns false when the derivative does not exist at this point (sqrt at 0, x^0.5 at 0).
bool partial(const ExpressionGraph& graph, ExprId id, std::size_t k,
             const std::vector<double>& v, const std::vector<double>* tangent, double* d,
             double* d_dot) {
  const Node& n = graph.node(id);
  const auto val = [&](std::size_t c) { return v[static_cast<std::size_t>(n.children[c])]; };
  const auto dot = [&](std::size_t c) {
    return tangent == nullptr ? 0.0 : (*tangent)[static_cast<std::size_t>(n.children[c])];
  };
  double first = 0.0;
  double second_along = 0.0;
  switch (n.op) {
    case Op::kConstant:
    case Op::kVariable: return true;  // no children
    case Op::kSum: first = 1.0; break;
    case Op::kNegate: first = -1.0; break;
    case Op::kProduct:
      // d(ab)/da = b, whose derivative along the direction is b-dot; and symmetrically. For
      // x*x both children are x, and the two contributions sum to 2x - the rule needs no case.
      first = val(1 - k);
      second_along = dot(1 - k);
      break;
    case Op::kDivide: {
      const double a = val(0);
      const double b = val(1);
      if (k == 0) {
        first = 1.0 / b;
        second_along = -dot(1) / (b * b);
      } else {
        first = -a / (b * b);
        second_along = -dot(0) / (b * b) + 2.0 * a * dot(1) / (b * b * b);
      }
      break;
    }
    case Op::kPower: {
      const double a = val(0);
      const double p = n.value;
      first = p == 1.0 ? 1.0 : p * std::pow(a, p - 1.0);
      if (tangent != nullptr && dot(0) != 0.0) {
        const double curvature = p == 2.0 ? 2.0 : p * (p - 1.0) * std::pow(a, p - 2.0);
        second_along = curvature * dot(0);
      }
      break;
    }
    case Op::kExp:
      first = v[static_cast<std::size_t>(id)];
      second_along = first * dot(0);
      break;
    case Op::kLog:
      first = 1.0 / val(0);
      second_along = -dot(0) / (val(0) * val(0));
      break;
    case Op::kSqrt: {
      const double s = v[static_cast<std::size_t>(id)];
      first = 0.5 / s;
      second_along = -dot(0) / (4.0 * s * s * s);
      break;
    }
  }
  if (!std::isfinite(first) || !std::isfinite(second_along)) return false;
  *d = first;
  if (d_dot != nullptr) *d_dot = second_along;
  return true;
}

Evaluation no_derivative(const ExpressionGraph& graph, ExprId id) {
  return failure(
      EvalError::kDomain, id,
      fmt::format("{} has a value here but no derivative", to_string(graph.node(id).op)));
}

}  // namespace

Evaluation ExpressionGraph::evaluate(ExprId root, const std::vector<double>& x) const {
  if (!contains(root)) {
    return failure(EvalError::kInvalid, root,
                   invalid_.empty() ? std::string("not a node of this graph") : invalid_);
  }
  if (x.size() != static_cast<std::size_t>(num_variables_)) {
    return failure(
        EvalError::kInvalid, root,
        fmt::format("a point with {} entries for {} columns", x.size(), num_variables_));
  }
  std::vector<double> values;
  return forward(*this, tape(root), x, &values);
}

bool ExpressionGraph::gradient(ExprId root, const std::vector<double>& x, SparseEntries* out,
                               Evaluation* error) const {
  out->clear();
  Evaluation checked = evaluate(root, x);
  if (!checked.ok()) {
    if (error != nullptr) *error = std::move(checked);
    return false;
  }
  const std::vector<ExprId> order = tape(root);
  std::vector<double> v;
  (void)forward(*this, order, x, &v);
  std::vector<double> adjoint(size(), 0.0);
  adjoint[static_cast<std::size_t>(root)] = 1.0;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const ExprId id = *it;
    const double bar = adjoint[static_cast<std::size_t>(id)];
    if (bar == 0.0) continue;
    const Node& n = node(id);
    for (std::size_t k = 0; k < n.children.size(); ++k) {
      double d = 0.0;
      if (!partial(*this, id, k, v, nullptr, &d, nullptr)) {
        if (error != nullptr) *error = no_derivative(*this, id);
        return false;
      }
      adjoint[static_cast<std::size_t>(n.children[k])] += bar * d;
    }
  }
  for (const ExprId id : order) {
    if (node(id).op == Op::kVariable) {
      out->emplace_back(node(id).variable, adjoint[static_cast<std::size_t>(id)]);
    }
  }
  std::sort(out->begin(), out->end());
  return true;
}

bool ExpressionGraph::hessian(ExprId root, const std::vector<double>& x,
                              std::vector<HessianEntry>* out, Evaluation* error) const {
  out->clear();
  Evaluation checked = evaluate(root, x);
  if (!checked.ok()) {
    if (error != nullptr) *error = std::move(checked);
    return false;
  }
  const std::vector<ExprId> order = tape(root);
  std::vector<double> v;
  (void)forward(*this, order, x, &v);

  // One node per column, by interning, so a column's node id is well defined.
  std::vector<ExprId> column_node(static_cast<std::size_t>(num_variables_), kNoExpr);
  std::vector<Index> columns;
  for (const ExprId id : order) {
    if (node(id).op == Op::kVariable) {
      column_node[static_cast<std::size_t>(node(id).variable)] = id;
      columns.push_back(node(id).variable);
    }
  }
  std::sort(columns.begin(), columns.end());

  std::vector<double> tangent(size(), 0.0);
  std::vector<double> adjoint(size(), 0.0);
  std::vector<double> adjoint_dot(size(), 0.0);
  for (const Index j : columns) {
    // Tangent sweep along e_j: every node's derivative with respect to x_j.
    std::fill(tangent.begin(), tangent.end(), 0.0);
    for (const ExprId id : order) {
      const Node& n = node(id);
      if (n.op == Op::kVariable) {
        tangent[static_cast<std::size_t>(id)] = n.variable == j ? 1.0 : 0.0;
        continue;
      }
      double t = 0.0;
      for (std::size_t k = 0; k < n.children.size(); ++k) {
        double d = 0.0;
        if (!partial(*this, id, k, v, nullptr, &d, nullptr)) {
          if (error != nullptr) *error = no_derivative(*this, id);
          return false;
        }
        t += d * tangent[static_cast<std::size_t>(n.children[k])];
      }
      tangent[static_cast<std::size_t>(id)] = t;
    }
    // Reverse sweep carrying adjoints and their derivatives along e_j.
    std::fill(adjoint.begin(), adjoint.end(), 0.0);
    std::fill(adjoint_dot.begin(), adjoint_dot.end(), 0.0);
    adjoint[static_cast<std::size_t>(root)] = 1.0;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      const ExprId id = *it;
      const double bar = adjoint[static_cast<std::size_t>(id)];
      const double bar_dot = adjoint_dot[static_cast<std::size_t>(id)];
      if (bar == 0.0 && bar_dot == 0.0) continue;
      const Node& n = node(id);
      for (std::size_t k = 0; k < n.children.size(); ++k) {
        double d = 0.0;
        double d_dot = 0.0;
        if (!partial(*this, id, k, v, &tangent, &d, &d_dot)) {
          if (error != nullptr) *error = no_derivative(*this, id);
          return false;
        }
        const auto c = static_cast<std::size_t>(n.children[k]);
        adjoint[c] += bar * d;
        adjoint_dot[c] += bar_dot * d + bar * d_dot;
      }
    }
    // Column j of the Hessian, lower triangle: rows i >= j. An entry that is exactly zero at
    // this point is omitted - the pattern is of the values here, not the structure.
    for (const Index i : columns) {
      if (i < j) continue;
      const double value =
          adjoint_dot[static_cast<std::size_t>(column_node[static_cast<std::size_t>(i)])];
      if (!std::isfinite(value)) {
        if (error != nullptr) *error = no_derivative(*this, root);
        return false;
      }
      if (value != 0.0) out->push_back(HessianEntry{i, j, value});
    }
  }
  return true;
}

DerivativeCheck check_derivatives(const ExpressionGraph& graph, ExprId root,
                                  const std::vector<double>& x, double step) {
  DerivativeCheck check;
  SparseEntries exact;
  Evaluation error;
  if (!graph.gradient(root, x, &exact, &error)) return check;
  std::vector<HessianEntry> hessian;
  if (!graph.hessian(root, x, &hessian, &error)) return check;

  const auto n = static_cast<std::size_t>(graph.num_variables());
  std::vector<double> dense_gradient(n, 0.0);
  for (const auto& [j, value] : exact) dense_gradient[static_cast<std::size_t>(j)] = value;
  std::vector<double> dense_hessian(n * n, 0.0);
  for (const HessianEntry& e : hessian) {
    dense_hessian[static_cast<std::size_t>(e.row) * n + static_cast<std::size_t>(e.col)] =
        e.value;
    dense_hessian[static_cast<std::size_t>(e.col) * n + static_cast<std::size_t>(e.row)] =
        e.value;
  }
  const auto relative = [](double a, double b) {
    return std::fabs(a - b) / std::max(1.0, std::fabs(b));
  };

  for (const Index j : graph.variables_of(root)) {
    const auto u = static_cast<std::size_t>(j);
    // The step scales with the coordinate, as a central difference's should.
    const double h = step * std::max(1.0, std::fabs(x[u]));
    std::vector<double> plus = x;
    std::vector<double> minus = x;
    plus[u] += h;
    minus[u] -= h;
    const Evaluation fp = graph.evaluate(root, plus);
    const Evaluation fm = graph.evaluate(root, minus);
    if (!fp.ok() || !fm.ok()) return DerivativeCheck{};  // the stencil left the domain
    check.gradient_error = std::max(
        check.gradient_error, relative((fp.value - fm.value) / (2.0 * h), dense_gradient[u]));

    SparseEntries gp;
    SparseEntries gm;
    if (!graph.gradient(root, plus, &gp, &error) || !graph.gradient(root, minus, &gm, &error)) {
      return DerivativeCheck{};
    }
    std::vector<double> column(n, 0.0);
    for (const auto& [i, value] : gp) column[static_cast<std::size_t>(i)] += value;
    for (const auto& [i, value] : gm) column[static_cast<std::size_t>(i)] -= value;
    for (std::size_t i = 0; i < n; ++i) {
      check.hessian_error = std::max(check.hessian_error,
                                     relative(column[i] / (2.0 * h), dense_hessian[i * n + u]));
    }
  }
  check.evaluated = true;
  return check;
}

}  // namespace sankhya::nlp

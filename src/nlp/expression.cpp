// SPDX-License-Identifier: Apache-2.0
// SANKHYA - expression graph construction: interning, safe simplification, constant folding
// (#296). Evaluation and derivatives are in derivatives.cpp, ranges and curvature in
// convexity.cpp.

#include "nlp/expression.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <fmt/format.h>

namespace sankhya::nlp {

const char* to_string(Op op) noexcept {
  switch (op) {
    case Op::kConstant: return "constant";
    case Op::kVariable: return "variable";
    case Op::kSum: return "sum";
    case Op::kProduct: return "product";
    case Op::kNegate: return "negate";
    case Op::kDivide: return "divide";
    case Op::kPower: return "power";
    case Op::kExp: return "exp";
    case Op::kLog: return "log";
    case Op::kSqrt: return "sqrt";
  }
  return "unknown";
}

const char* to_string(Curvature curvature) noexcept {
  switch (curvature) {
    case Curvature::kConstant: return "constant";
    case Curvature::kAffine: return "affine";
    case Curvature::kConvex: return "convex";
    case Curvature::kConcave: return "concave";
    case Curvature::kUnknown: return "unknown";
  }
  return "unknown";
}

namespace {

/// Bits, not the value: 0.0 and -0.0 must not intern to one node, and a key built from a
/// printed decimal would merge constants that differ in the last place.
std::uint64_t bits_of(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool is_integer(double value) {
  return std::isfinite(value) && std::floor(value) == value;
}

}  // namespace

ExpressionGraph::ExpressionGraph(Index num_variables)
    : num_variables_(num_variables < 0 ? 0 : num_variables) {
  if (num_variables < 0) {
    mark_invalid(fmt::format("a graph over {} columns", num_variables));
  }
}

void ExpressionGraph::mark_invalid(std::string message) {
  if (invalid_.empty()) invalid_ = std::move(message);
}

bool ExpressionGraph::usable(ExprId id, const char* operation) {
  if (contains(id)) return true;
  // kNoExpr arriving here is an earlier failure propagating, already recorded; anything else
  // is an id from another graph or from nowhere, and that is new information.
  if (id != kNoExpr) {
    mark_invalid(
        fmt::format("{} was given node {}, which is not in this graph", operation, id));
  }
  return false;
}

const Node& ExpressionGraph::node(ExprId id) const {
  return nodes_[static_cast<std::size_t>(id)];
}

ExprId ExpressionGraph::intern(Node node) {
  std::string key = fmt::format("{}|{:016x}|{}|", static_cast<int>(node.op),
                                bits_of(node.value), node.variable);
  for (const ExprId child : node.children) key += fmt::format("{},", child);
  const auto found = interned_.find(key);
  if (found != interned_.end()) return found->second;
  const auto id = static_cast<ExprId>(nodes_.size());
  nodes_.push_back(std::move(node));
  interned_.emplace(std::move(key), id);
  return id;
}

bool ExpressionGraph::is_constant(ExprId id, double* value) const {
  if (!contains(id) || node(id).op != Op::kConstant) return false;
  if (value != nullptr) *value = node(id).value;
  return true;
}

bool ExpressionGraph::defined_everywhere(ExprId id) const {
  // Children precede parents, so one ascending pass over the tape answers for the root.
  for (const ExprId k : tape(id)) {
    const Node& n = node(k);
    switch (n.op) {
      case Op::kDivide:
      case Op::kLog:
      case Op::kSqrt: return false;
      case Op::kPower:
        if (!is_integer(n.value) || n.value < 0.0) return false;
        break;
      // exp is defined everywhere. It can overflow, but so can a product or a sum: overflow
      // is a numerical event at a point, not a domain, and folding 0 * f does not claim to
      // preserve it. What it must preserve is an operation the caller's point cannot take.
      default: break;
    }
  }
  return true;
}

// ---- Construction
// ----------------------------------------------------------------------------

ExprId ExpressionGraph::constant(double value) {
  if (!std::isfinite(value)) {
    mark_invalid(fmt::format("a constant must be finite, got {}", value));
    return kNoExpr;
  }
  Node n;
  n.op = Op::kConstant;
  n.value = value;
  return intern(std::move(n));
}

ExprId ExpressionGraph::variable(Index column) {
  if (column < 0 || column >= num_variables_) {
    mark_invalid(fmt::format("column {} is outside the {} columns of this model", column,
                             num_variables_));
    return kNoExpr;
  }
  Node n;
  n.op = Op::kVariable;
  n.variable = column;
  return intern(std::move(n));
}

ExprId ExpressionGraph::sum(std::vector<ExprId> terms) {
  // Flatten nested sums and gather the constants into one term: sum(sum(a, b), 2, 3) is
  // sum(a, b, 5). A zero constant is dropped - x + 0 is x everywhere x is defined.
  std::vector<ExprId> flat;
  double constant_part = 0.0;
  for (const ExprId t : terms) {
    if (!usable(t, "sum")) return kNoExpr;
    double c = 0.0;
    if (is_constant(t, &c)) {
      constant_part += c;
    } else if (node(t).op == Op::kSum) {
      for (const ExprId inner : node(t).children) {
        if (is_constant(inner, &c)) {
          constant_part += c;
        } else {
          flat.push_back(inner);
        }
      }
    } else {
      flat.push_back(t);
    }
  }
  if (constant_part != 0.0) {
    const ExprId folded = constant(constant_part);
    if (folded == kNoExpr) return kNoExpr;  // the constants overflowed
    flat.push_back(folded);
  }
  if (flat.empty()) return constant(0.0);
  if (flat.size() == 1) return flat.front();
  std::sort(flat.begin(), flat.end());  // commutative: x + y and y + x are one node
  Node n;
  n.op = Op::kSum;
  n.children = std::move(flat);
  return intern(std::move(n));
}

ExprId ExpressionGraph::add(ExprId a, ExprId b) {
  return sum({a, b});
}

ExprId ExpressionGraph::subtract(ExprId a, ExprId b) {
  return sum({a, negate(b)});
}

ExprId ExpressionGraph::negate(ExprId a) {
  if (!usable(a, "negate")) return kNoExpr;
  double c = 0.0;
  if (is_constant(a, &c)) return constant(-c);
  if (node(a).op == Op::kNegate) return node(a).children.front();  // -(-x) is x
  Node n;
  n.op = Op::kNegate;
  n.children = {a};
  return intern(std::move(n));
}

ExprId ExpressionGraph::multiply(ExprId a, ExprId b) {
  if (!usable(a, "multiply") || !usable(b, "multiply")) return kNoExpr;
  double ca = 0.0;
  double cb = 0.0;
  const bool a_const = is_constant(a, &ca);
  const bool b_const = is_constant(b, &cb);
  if (a_const && b_const) {
    const double product = ca * cb;
    if (std::isfinite(product)) return constant(product);
  } else if (a_const || b_const) {
    const double c = a_const ? ca : cb;
    const ExprId other = a_const ? b : a;
    if (c == 1.0) return other;
    if (c == -1.0) return negate(other);
    // x * 0 is 0 ONLY where x is defined: 0 * log(x) at x = -1 is an error, not a zero, and
    // folding it away would hide the error from the caller's point. Kept as a product when
    // the other side has a restricted domain.
    if (c == 0.0 && defined_everywhere(other)) return constant(0.0);
  }
  Node n;
  n.op = Op::kProduct;
  n.children = {std::min(a, b), std::max(a, b)};
  return intern(std::move(n));
}

ExprId ExpressionGraph::divide(ExprId numerator, ExprId denominator) {
  if (!usable(numerator, "divide") || !usable(denominator, "divide")) return kNoExpr;
  double cn = 0.0;
  double cd = 0.0;
  const bool n_const = is_constant(numerator, &cn);
  const bool d_const = is_constant(denominator, &cd);
  if (d_const && cd == 1.0) return numerator;
  // A constant quotient folds only when it is defined: 1 / 0 stays a node, so evaluation
  // reports the division by zero instead of construction inventing a value for it.
  if (n_const && d_const && cd != 0.0 && std::isfinite(cn / cd)) return constant(cn / cd);
  Node n;
  n.op = Op::kDivide;
  n.children = {numerator, denominator};
  return intern(std::move(n));
}

ExprId ExpressionGraph::power(ExprId base, double exponent) {
  if (!usable(base, "power")) return kNoExpr;
  if (!std::isfinite(exponent)) {
    mark_invalid(fmt::format("an exponent must be finite, got {}", exponent));
    return kNoExpr;
  }
  if (exponent == 1.0) return base;
  // x ^ 0 is 1 wherever x is defined, 0 ^ 0 included by the usual convention.
  if (exponent == 0.0 && defined_everywhere(base)) return constant(1.0);
  double c = 0.0;
  if (is_constant(base, &c)) {
    const bool defined =
        (c > 0.0) || (c == 0.0 && exponent > 0.0) || (c < 0.0 && is_integer(exponent));
    if (defined && std::isfinite(std::pow(c, exponent))) return constant(std::pow(c, exponent));
  }
  Node n;
  n.op = Op::kPower;
  n.value = exponent;
  n.children = {base};
  return intern(std::move(n));
}

namespace {
/// The one-argument atoms share their folding rule: fold a constant argument when the
/// function is defined there and the result is finite, keep the node otherwise so that
/// evaluation reports the problem at the point where it happens.
struct Unary {
  Op op;
  bool (*defined)(double);
  double (*apply)(double);
};
bool always(double) {
  return true;
}
bool positive(double v) {
  return v > 0.0;
}
bool non_negative(double v) {
  return v >= 0.0;
}
double apply_exp(double v) {
  return std::exp(v);
}
double apply_log(double v) {
  return std::log(v);
}
double apply_sqrt(double v) {
  return std::sqrt(v);
}
}  // namespace

ExprId ExpressionGraph::exp(ExprId a) {
  if (!usable(a, "exp")) return kNoExpr;
  const Unary atom{Op::kExp, always, apply_exp};
  double c = 0.0;
  if (is_constant(a, &c) && atom.defined(c) && std::isfinite(atom.apply(c))) {
    return constant(atom.apply(c));
  }
  // exp(log(x)) is NOT simplified to x: it is x only where log(x) is defined, and keeping the
  // composition keeps the domain.
  Node n;
  n.op = atom.op;
  n.children = {a};
  return intern(std::move(n));
}

ExprId ExpressionGraph::log(ExprId a) {
  if (!usable(a, "log")) return kNoExpr;
  const Unary atom{Op::kLog, positive, apply_log};
  double c = 0.0;
  if (is_constant(a, &c) && atom.defined(c)) return constant(atom.apply(c));
  Node n;
  n.op = atom.op;
  n.children = {a};
  return intern(std::move(n));
}

ExprId ExpressionGraph::sqrt(ExprId a) {
  if (!usable(a, "sqrt")) return kNoExpr;
  const Unary atom{Op::kSqrt, non_negative, apply_sqrt};
  double c = 0.0;
  if (is_constant(a, &c) && atom.defined(c)) return constant(atom.apply(c));
  Node n;
  n.op = atom.op;
  n.children = {a};
  return intern(std::move(n));
}

// ---- Structure
// -------------------------------------------------------------------------------

std::vector<ExprId> ExpressionGraph::tape(ExprId root) const {
  if (!contains(root)) return {};
  std::vector<char> reached(static_cast<std::size_t>(root) + 1, 0);
  reached[static_cast<std::size_t>(root)] = 1;
  // Walk down from the root in DESCENDING id order: every child has a smaller id than its
  // parent, so by the time an id is visited every parent that reaches it has been.
  for (ExprId id = root; id >= 0; --id) {
    if (reached[static_cast<std::size_t>(id)] == 0) continue;
    for (const ExprId child : node(id).children) reached[static_cast<std::size_t>(child)] = 1;
  }
  std::vector<ExprId> order;
  for (ExprId id = 0; id <= root; ++id) {
    if (reached[static_cast<std::size_t>(id)] != 0) order.push_back(id);
  }
  return order;  // ascending, so children before parents
}

std::vector<Index> ExpressionGraph::variables_of(ExprId root) const {
  std::vector<Index> columns;
  for (const ExprId id : tape(root)) {
    if (node(id).op == Op::kVariable) columns.push_back(node(id).variable);
  }
  std::sort(columns.begin(), columns.end());
  columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
  return columns;
}

std::string ExpressionGraph::to_string(ExprId root) const {
  if (!contains(root)) return "<invalid>";
  const Node& n = node(root);
  switch (n.op) {
    case Op::kConstant: return fmt::format("{:g}", n.value);
    case Op::kVariable: return fmt::format("x{}", n.variable);
    case Op::kSum: {
      std::string text = "(";
      for (std::size_t k = 0; k < n.children.size(); ++k) {
        if (k > 0) text += " + ";
        text += to_string(n.children[k]);
      }
      return text + ")";
    }
    case Op::kProduct:
      return fmt::format("({} * {})", to_string(n.children[0]), to_string(n.children[1]));
    case Op::kNegate: return fmt::format("-{}", to_string(n.children[0]));
    case Op::kDivide:
      return fmt::format("({} / {})", to_string(n.children[0]), to_string(n.children[1]));
    case Op::kPower: return fmt::format("{}^{:g}", to_string(n.children[0]), n.value);
    case Op::kExp: return fmt::format("exp({})", to_string(n.children[0]));
    case Op::kLog: return fmt::format("log({})", to_string(n.children[0]));
    case Op::kSqrt: return fmt::format("sqrt({})", to_string(n.children[0]));
  }
  return "<unknown>";
}

}  // namespace sankhya::nlp

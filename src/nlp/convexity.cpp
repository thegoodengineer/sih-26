// SPDX-License-Identifier: Apache-2.0
// SANKHYA - ranges, curvature, degree and domain risks of an expression graph (#296).
//
// Each is one ascending pass over the tape - children before parents - with a per-node
// result, so shared subexpressions are analysed once.
//
// CURVATURE is the disciplined-convex-programming composition rule (Grant, Boyd and Ye,
// "Disciplined convex programming", Global Optimization, Springer 2006; Boyd and
// Vandenberghe, "Convex Optimization", section 3.2.4): f = h(g) is convex when h is convex
// and either h is nondecreasing over the range of g and g is convex, or h is nonincreasing
// there and g is concave, or g is affine; concave symmetrically. The range of g comes from
// interval arithmetic over the column bounds, which is what lets x^3 be convex on x >= 0 and
// 1/x convex on x > 0 while staying unknown where the rules do not reach. Unknown is the
// answer whenever a rule does not apply: a bilinear x*y, a sum of a convex and a concave
// term, anything whose domain the box can leave.

#include <algorithm>
#include <cmath>
#include <limits>

#include <fmt/format.h>

#include "nlp/expression.hpp"

namespace sankhya::nlp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

bool is_integer(double value) {
  return std::isfinite(value) && std::floor(value) == value;
}

/// a * b with 0 * inf = 0: the zero is exact and the infinity is only a bound.
double times(double a, double b) {
  return (a == 0.0 || b == 0.0) ? 0.0 : a * b;
}

Interval hull(std::initializer_list<double> values, bool undefined) {
  Interval r;
  r.lower = kInf;
  r.upper = -kInf;
  for (const double v : values) {
    if (std::isnan(v)) continue;
    r.lower = std::min(r.lower, v);
    r.upper = std::max(r.upper, v);
  }
  if (r.lower > r.upper) r = Interval{};
  r.may_be_undefined = undefined;
  return r;
}

/// The range of t^p over [lo, hi], for a t the caller has already restricted to the domain.
Interval power_range(double lo, double hi, double p, bool undefined) {
  if (is_integer(p) && p > 0.0 && std::fmod(p, 2.0) == 0.0) {
    // Even power: smallest at the point of the interval nearest zero.
    const double near = (lo <= 0.0 && hi >= 0.0) ? 0.0 : std::min(std::fabs(lo), std::fabs(hi));
    const double far = std::max(std::fabs(lo), std::fabs(hi));
    return hull({std::pow(near, p), std::pow(far, p)}, undefined);
  }
  // Every other case is monotone on an interval that stays inside the domain.
  return hull({std::pow(lo, p), std::pow(hi, p)}, undefined);
}

enum class Monotone { kIncreasing, kDecreasing, kNeither };

/// f = h(g) by the composition rule, for h of curvature `outer` and monotonicity `direction`
/// over the range of g.
Curvature compose(Curvature outer, Monotone direction, Curvature inner) {
  if (inner == Curvature::kConstant) return Curvature::kConstant;
  if (outer == Curvature::kUnknown || inner == Curvature::kUnknown) return Curvature::kUnknown;
  if (inner == Curvature::kAffine) return outer;
  const bool inc = direction == Monotone::kIncreasing;
  const bool dec = direction == Monotone::kDecreasing;
  if (outer == Curvature::kConvex) {
    if ((inc && inner == Curvature::kConvex) || (dec && inner == Curvature::kConcave)) {
      return Curvature::kConvex;
    }
  } else if (outer == Curvature::kConcave) {
    if ((inc && inner == Curvature::kConcave) || (dec && inner == Curvature::kConvex)) {
      return Curvature::kConcave;
    }
  }
  return Curvature::kUnknown;
}

Curvature flip(Curvature c) {
  if (c == Curvature::kConvex) return Curvature::kConcave;
  if (c == Curvature::kConcave) return Curvature::kConvex;
  return c;
}

/// Curvature and monotonicity of t -> t^p over the range r of t, which the caller has
/// checked lies inside the domain.
void power_shape(double p, const Interval& r, Curvature* curvature, Monotone* direction) {
  const bool nonneg = r.lower >= 0.0;
  const bool nonpos = r.upper <= 0.0;
  *curvature = Curvature::kUnknown;
  *direction = Monotone::kNeither;
  if (is_integer(p) && p >= 2.0) {
    if (std::fmod(p, 2.0) == 0.0) {
      *curvature = Curvature::kConvex;  // even: convex on the whole line
      *direction = nonneg ? Monotone::kIncreasing
                          : (nonpos ? Monotone::kDecreasing : Monotone::kNeither);
    } else if (nonneg) {
      *curvature = Curvature::kConvex;  // odd: convex on t >= 0, concave on t <= 0
      *direction = Monotone::kIncreasing;
    } else if (nonpos) {
      *curvature = Curvature::kConcave;
      *direction = Monotone::kIncreasing;
    }
  } else if (is_integer(p) && p < 0.0) {
    // The domain excludes 0, so r lies on one side.
    if (r.lower > 0.0) {
      *curvature = Curvature::kConvex;
      *direction = Monotone::kDecreasing;
    } else if (r.upper < 0.0) {
      const bool even = std::fmod(p, 2.0) == 0.0;
      *curvature = even ? Curvature::kConvex : Curvature::kConcave;
      *direction = even ? Monotone::kIncreasing : Monotone::kDecreasing;
    }
  } else if (p > 1.0) {
    *curvature = Curvature::kConvex;  // non-integer p > 1, on t >= 0
    *direction = Monotone::kIncreasing;
  } else if (p > 0.0) {
    *curvature = Curvature::kConcave;  // 0 < p < 1, on t >= 0
    *direction = Monotone::kIncreasing;
  } else {
    *curvature = Curvature::kConvex;  // non-integer p < 0, on t > 0
    *direction = Monotone::kDecreasing;
  }
}

/// True when every point of r is in the domain of t^p.
bool power_defined(double p, const Interval& r) {
  if (r.may_be_undefined) return false;
  if (is_integer(p)) return p >= 0.0 || r.lower > 0.0 || r.upper < 0.0;
  return p > 0.0 ? r.lower >= 0.0 : r.lower > 0.0;
}

}  // namespace

Interval ExpressionGraph::range(ExprId root, const std::vector<double>& lower,
                                const std::vector<double>& upper) const {
  if (!contains(root)) return Interval{-kInf, kInf, true};
  return ranges(root, lower, upper)[static_cast<std::size_t>(root)];
}

std::vector<Interval> ExpressionGraph::ranges(ExprId root, const std::vector<double>& lower,
                                              const std::vector<double>& upper) const {
  std::vector<Interval> r(size());
  if (!contains(root)) return r;
  for (const ExprId id : tape(root)) {
    const Node& n = node(id);
    const auto child = [&](std::size_t k) {
      return r[static_cast<std::size_t>(n.children[k])];
    };
    bool undefined = false;
    for (const ExprId c : n.children)
      undefined = undefined || r[static_cast<std::size_t>(c)].may_be_undefined;
    Interval out;
    switch (n.op) {
      case Op::kConstant: out = Interval{n.value, n.value, false}; break;
      case Op::kVariable: {
        const auto u = static_cast<std::size_t>(n.variable);
        out = Interval{lower[u], upper[u], false};
        break;
      }
      case Op::kSum:
        out = Interval{0.0, 0.0, undefined};
        for (std::size_t k = 0; k < n.children.size(); ++k) {
          out.lower += child(k).lower;
          out.upper += child(k).upper;
        }
        break;
      case Op::kNegate: out = Interval{-child(0).upper, -child(0).lower, undefined}; break;
      case Op::kProduct:
        if (n.children[0] == n.children[1]) {
          out = power_range(child(0).lower, child(0).upper, 2.0, undefined);  // x * x
        } else {
          const Interval a = child(0);
          const Interval b = child(1);
          out = hull({times(a.lower, b.lower), times(a.lower, b.upper), times(a.upper, b.lower),
                      times(a.upper, b.upper)},
                     undefined);
        }
        break;
      case Op::kDivide: {
        const Interval b = child(1);
        if (b.lower <= 0.0 && b.upper >= 0.0) {
          out = Interval{-kInf, kInf, true};  // the denominator can be zero
        } else {
          const Interval a = child(0);
          const Interval inv{1.0 / b.upper, 1.0 / b.lower, false};
          out = hull({times(a.lower, inv.lower), times(a.lower, inv.upper),
                      times(a.upper, inv.lower), times(a.upper, inv.upper)},
                     undefined);
        }
        break;
      }
      case Op::kPower: {
        Interval a = child(0);
        const double p = n.value;
        bool bad = undefined || !power_defined(p, a);
        // Restrict to the domain before taking the range; the flag says the box left it.
        if (!is_integer(p)) a.lower = std::max(a.lower, p > 0.0 ? 0.0 : 1e-300);
        if (a.lower > a.upper) {
          out = Interval{-kInf, kInf, true};
        } else if (is_integer(p) && p < 0.0 && a.lower <= 0.0 && a.upper >= 0.0) {
          out = Interval{-kInf, kInf, true};
        } else {
          out = power_range(a.lower, a.upper, p, bad);
        }
        break;
      }
      case Op::kExp:
        out = Interval{std::exp(child(0).lower), std::exp(child(0).upper), undefined};
        break;
      case Op::kLog: {
        const Interval a = child(0);
        const bool bad = undefined || a.lower <= 0.0;
        out = Interval{a.lower > 0.0 ? std::log(a.lower) : -kInf,
                       a.upper > 0.0 ? std::log(a.upper) : -kInf, bad};
        break;
      }
      case Op::kSqrt: {
        const Interval a = child(0);
        const bool bad = undefined || a.lower < 0.0;
        out = Interval{std::sqrt(std::max(a.lower, 0.0)),
                       a.upper >= 0.0 ? std::sqrt(a.upper) : 0.0, bad};
        break;
      }
    }
    if (std::isnan(out.lower) || std::isnan(out.upper))
      out = Interval{-kInf, kInf, out.may_be_undefined};
    r[static_cast<std::size_t>(id)] = out;
  }
  return r;
}

Curvature ExpressionGraph::curvature(ExprId root, const std::vector<double>& lower,
                                     const std::vector<double>& upper) const {
  if (!contains(root)) return Curvature::kUnknown;
  const std::vector<Interval> r = ranges(root, lower, upper);
  std::vector<Curvature> c(size(), Curvature::kUnknown);
  for (const ExprId id : tape(root)) {
    const Node& n = node(id);
    const auto kid = [&](std::size_t k) { return c[static_cast<std::size_t>(n.children[k])]; };
    const auto kid_range = [&](std::size_t k) {
      return r[static_cast<std::size_t>(n.children[k])];
    };
    Curvature out = Curvature::kUnknown;
    switch (n.op) {
      case Op::kConstant: out = Curvature::kConstant; break;
      case Op::kVariable: out = Curvature::kAffine; break;
      case Op::kSum: {
        bool any_convex = false;
        bool any_concave = false;
        bool any_unknown = false;
        bool all_constant = true;
        for (std::size_t k = 0; k < n.children.size(); ++k) {
          const Curvature term = kid(k);
          any_convex = any_convex || term == Curvature::kConvex;
          any_concave = any_concave || term == Curvature::kConcave;
          any_unknown = any_unknown || term == Curvature::kUnknown;
          all_constant = all_constant && term == Curvature::kConstant;
        }
        if (any_unknown || (any_convex && any_concave)) {
          out = Curvature::kUnknown;
        } else if (all_constant) {
          out = Curvature::kConstant;
        } else if (any_convex) {
          out = Curvature::kConvex;
        } else if (any_concave) {
          out = Curvature::kConcave;
        } else {
          out = Curvature::kAffine;
        }
        break;
      }
      case Op::kNegate: out = flip(kid(0)); break;
      case Op::kProduct: {
        double factor = 0.0;
        if (is_constant(n.children[0], &factor) || is_constant(n.children[1], &factor)) {
          const Curvature other = is_constant(n.children[0]) ? kid(1) : kid(0);
          out = factor > 0.0 ? other : (factor < 0.0 ? flip(other) : Curvature::kConstant);
        } else if (n.children[0] == n.children[1]) {
          Curvature h = Curvature::kUnknown;
          Monotone m = Monotone::kNeither;
          power_shape(2.0, kid_range(0), &h, &m);
          out = compose(h, m, kid(0));
        }
        // x * y of two different non-constant factors: indefinite in general; left unknown.
        break;
      }
      case Op::kDivide: {
        double value = 0.0;
        if (is_constant(n.children[1], &value) && value != 0.0) {
          out = value > 0.0 ? kid(0) : flip(kid(0));  // scaling by 1/c
        } else if (is_constant(n.children[0], &value)) {
          // c / g: t -> c/t is, for c > 0, convex and decreasing on t > 0 and concave and
          // decreasing on t < 0; for c < 0 the mirror image. Undefined if g can reach 0.
          const Interval g = kid_range(1);
          if (value == 0.0) {
            out = Curvature::kConstant;
          } else if (!g.may_be_undefined && g.lower > 0.0) {
            out = compose(value > 0.0 ? Curvature::kConvex : Curvature::kConcave,
                          value > 0.0 ? Monotone::kDecreasing : Monotone::kIncreasing, kid(1));
          } else if (!g.may_be_undefined && g.upper < 0.0) {
            out = compose(value > 0.0 ? Curvature::kConcave : Curvature::kConvex,
                          value > 0.0 ? Monotone::kDecreasing : Monotone::kIncreasing, kid(1));
          }
        }
        break;
      }
      case Op::kPower: {
        const Interval g = kid_range(0);
        if (power_defined(n.value, g)) {
          Curvature h = Curvature::kUnknown;
          Monotone m = Monotone::kNeither;
          power_shape(n.value, g, &h, &m);
          out = compose(h, m, kid(0));
        }
        break;
      }
      case Op::kExp:
        out = kid(0) == Curvature::kUnknown
                  ? Curvature::kUnknown
                  : compose(Curvature::kConvex, Monotone::kIncreasing, kid(0));
        break;
      case Op::kLog: {
        const Interval g = kid_range(0);
        if (!g.may_be_undefined && g.lower > 0.0) {
          out = compose(Curvature::kConcave, Monotone::kIncreasing, kid(0));
        }
        break;
      }
      case Op::kSqrt: {
        const Interval g = kid_range(0);
        if (!g.may_be_undefined && g.lower >= 0.0) {
          out = compose(Curvature::kConcave, Monotone::kIncreasing, kid(0));
        }
        break;
      }
    }
    c[static_cast<std::size_t>(id)] = out;
  }
  return c[static_cast<std::size_t>(root)];
}

int ExpressionGraph::degree(ExprId root) const {
  if (!contains(root)) return -1;
  std::vector<int> d(size(), 0);
  for (const ExprId id : tape(root)) {
    const Node& n = node(id);
    const auto kid = [&](std::size_t k) { return d[static_cast<std::size_t>(n.children[k])]; };
    int out = -1;
    switch (n.op) {
      case Op::kConstant: out = 0; break;
      case Op::kVariable: out = 1; break;
      case Op::kSum:
        out = 0;
        for (std::size_t k = 0; k < n.children.size(); ++k) {
          if (kid(k) < 0) {
            out = -1;
            break;
          }
          out = std::max(out, kid(k));
        }
        break;
      case Op::kNegate: out = kid(0); break;
      case Op::kProduct: out = (kid(0) < 0 || kid(1) < 0) ? -1 : kid(0) + kid(1); break;
      case Op::kDivide: out = kid(1) == 0 ? kid(0) : -1; break;
      case Op::kPower:
        out = (kid(0) >= 0 && is_integer(n.value) && n.value >= 0.0)
                  ? (kid(0) == 0 ? 0 : kid(0) * static_cast<int>(n.value))
                  : (kid(0) == 0 ? 0 : -1);
        break;
      case Op::kExp:
      case Op::kLog:
      case Op::kSqrt: out = kid(0) == 0 ? 0 : -1; break;
    }
    d[static_cast<std::size_t>(id)] = out;
  }
  return d[static_cast<std::size_t>(root)];
}

std::vector<std::string> ExpressionGraph::domain_risks(ExprId root,
                                                       const std::vector<double>& lower,
                                                       const std::vector<double>& upper) const {
  std::vector<std::string> risks;
  if (!contains(root)) {
    risks.emplace_back("the expression is not a node of this graph");
    return risks;
  }
  const std::vector<Interval> r = ranges(root, lower, upper);
  for (const ExprId id : tape(root)) {
    const Node& n = node(id);
    if (n.children.empty()) continue;
    const Interval a = r[static_cast<std::size_t>(n.children[0])];
    const std::string arg = to_string(n.children[0]);
    switch (n.op) {
      case Op::kLog:
        if (a.lower <= 0.0) {
          risks.push_back(
              fmt::format("log({}) needs a positive argument; over the bounds it "
                          "ranges down to {:g}",
                          arg, a.lower));
        }
        break;
      case Op::kSqrt:
        if (a.lower < 0.0) {
          risks.push_back(
              fmt::format("sqrt({}) needs a non-negative argument; over the bounds "
                          "it ranges down to {:g}",
                          arg, a.lower));
        }
        break;
      case Op::kDivide: {
        const Interval b = r[static_cast<std::size_t>(n.children[1])];
        if (b.lower <= 0.0 && b.upper >= 0.0) {
          risks.push_back(
              fmt::format("the denominator {} can be zero over the bounds "
                          "([{:g}, {:g}])",
                          to_string(n.children[1]), b.lower, b.upper));
        }
        break;
      }
      case Op::kPower:
        if (!power_defined(n.value, Interval{a.lower, a.upper, false})) {
          risks.push_back(fmt::format("{}^{:g} is undefined for part of [{:g}, {:g}]", arg,
                                      n.value, a.lower, a.upper));
        }
        break;
      default: break;
    }
  }
  return risks;
}

}  // namespace sankhya::nlp

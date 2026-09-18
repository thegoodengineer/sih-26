// SPDX-License-Identifier: Apache-2.0
// SANKHYA - nonlinear expressions: representation, evaluation, derivatives, convexity (#296).
//
// WHAT THIS IS. The representation a nonlinear objective or constraint needs before any
// engine can be asked to solve it: a directed acyclic graph of operations over the model's
// columns, evaluated without ever returning NaN as a value, differentiated exactly, and
// classified for convexity by rules that are written down. It is a MODEL layer. It does not
// solve anything, and nothing in the LP / MILP / QP / MIQP engines reads it; those keep the
// Model they have, which stays frozen.
//
// OWNERSHIP. A graph owns every node in one arena, and a node is named by its index. Children
// are created before their parent, so a child's index is always smaller: the graph cannot hold
// a cycle, a reference cannot dangle while the graph lives, and the arena order IS a
// topological order, which is what evaluation walks. Nodes are immutable once created.
//
// SHARING. Construction is hash-consed: asking for an operation that already exists returns
// the existing node. `x1 + x2` built twice is one node, and `y = x1 + x2; y*y + 3*y` refers
// to it twice without copying it. Commutative operations sort their children first, so
// `x + y` and `y + x` are the same node as well.
//
// DERIVATIVES are exact. The gradient is reverse-mode automatic differentiation over the
// graph; the Hessian is forward-over-reverse (a second-order adjoint), one sweep per column
// the expression actually contains, returned as a sparse lower triangle. Griewank and
// Walther, "Evaluating Derivatives", 2nd ed., SIAM 2008, chapters 3 and 5. Finite differences
// exist only as a CHECK on those, and are named as approximations wherever they appear.
//
// CONVEXITY follows the composition rules of disciplined convex programming (Grant, Boyd and
// Ye, "Disciplined convex programming", in Global Optimization, Springer 2006): an atom's
// curvature, its monotonicity over the RANGE its argument can take, and the curvature of the
// argument. Ranges come from interval arithmetic over the column bounds. Anything the rules
// cannot establish is `kUnknown`, never guessed convex.
//
// THREAD SAFETY. Building a graph is single-threaded. Every evaluation function is const and
// writes only into caller-owned vectors, so one graph may be evaluated from several threads at
// once, each with its own output.

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya::nlp {

/// The operations a node can perform. The set is what can be evaluated, differentiated twice
/// and classified reliably; anything else is out of scope for this layer rather than half-
/// supported.
enum class Op : std::uint8_t {
  kConstant,  ///< `value`
  kVariable,  ///< column `variable`
  kSum,       ///< sum of any number of children
  kProduct,   ///< children[0] * children[1]
  kNegate,    ///< -children[0]
  kDivide,    ///< children[0] / children[1]
  kPower,     ///< children[0] ^ `value`, a constant exponent
  kExp,
  kLog,  ///< natural logarithm
  kSqrt,
};

[[nodiscard]] const char* to_string(Op op) noexcept;

/// A node's name: its index in the graph's arena.
using ExprId = std::int32_t;
inline constexpr ExprId kNoExpr = -1;

struct Node {
  Op op = Op::kConstant;
  double value = 0.0;   ///< the constant, or the exponent of a kPower
  Index variable = -1;  ///< the column of a kVariable
  std::vector<ExprId> children;
};

/// A closed interval of the real line, possibly unbounded. Used to decide where an operation
/// is defined and which way a function is monotone over the values its argument can take.
struct Interval {
  double lower = -std::numeric_limits<double>::infinity();
  double upper = std::numeric_limits<double>::infinity();
  /// True when the argument could land where the operation is undefined, so interval
  /// arithmetic above it means nothing.
  bool may_be_undefined = false;
};

/// Curvature by the disciplined-convex-programming rules. `kUnknown` means the rules could
/// not establish one, not that the expression is nonconvex.
enum class Curvature : std::uint8_t { kConstant, kAffine, kConvex, kConcave, kUnknown };
[[nodiscard]] const char* to_string(Curvature curvature) noexcept;

/// Why an evaluation produced no number.
enum class EvalError : std::uint8_t {
  kNone,
  kDomain,     ///< log of a non-positive, sqrt of a negative, division by zero, ...
  kNonFinite,  ///< every operation was defined but the result overflowed
  kInvalid,    ///< the expression itself is malformed (see ExpressionGraph::invalid())
};

struct Evaluation {
  double value = 0.0;
  EvalError error = EvalError::kNone;
  ExprId failed_at = kNoExpr;  ///< the first node that could not be evaluated
  std::string message;         ///< what went wrong there, in words
  [[nodiscard]] bool ok() const noexcept { return error == EvalError::kNone; }
};

/// A sparse vector, sorted by index with no repeats.
using SparseEntries = std::vector<std::pair<Index, double>>;

/// One entry of a sparse symmetric matrix held as its lower triangle (row >= col).
struct HessianEntry {
  Index row = 0;
  Index col = 0;
  double value = 0.0;
};

class ExpressionGraph {
 public:
  /// A graph over `num_variables` columns. Variable indices outside [0, num_variables) are
  /// refused at construction rather than read out of bounds later.
  ///
  /// MISUSE IS STICKY, NOT THROWN. Nothing in src/ throws, and a malformed expression must
  /// not become a plausible one. A call given an id that is not in the graph, a column out of
  /// range, or a non-finite constant returns kNoExpr and records the first such message in
  /// invalid(); every operation given kNoExpr returns kNoExpr, and evaluating it reports
  /// EvalError::kInvalid. It is the convention SparseMatrix uses for an overflowed count
  /// (#305): one flag, checked where the model is validated.
  explicit ExpressionGraph(Index num_variables);

  /// Empty while every construction has been well formed; otherwise the first problem.
  [[nodiscard]] const std::string& invalid() const noexcept { return invalid_; }

  // ---- Construction. Every call simplifies what it safely can, folds constants, and returns
  // an existing node when the same operation was built before.
  ExprId constant(double value);
  ExprId variable(Index column);
  ExprId sum(std::vector<ExprId> terms);
  ExprId add(ExprId a, ExprId b);
  ExprId subtract(ExprId a, ExprId b);
  ExprId multiply(ExprId a, ExprId b);
  ExprId divide(ExprId numerator, ExprId denominator);
  ExprId negate(ExprId a);
  ExprId power(ExprId base, double exponent);
  ExprId exp(ExprId a);
  ExprId log(ExprId a);
  ExprId sqrt(ExprId a);

  /// `id` must be a node of this graph (see contains()).
  [[nodiscard]] const Node& node(ExprId id) const;
  [[nodiscard]] bool contains(ExprId id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < nodes_.size();
  }
  [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }
  [[nodiscard]] Index num_variables() const noexcept { return num_variables_; }

  /// The nodes `root` depends on, itself included, in evaluation order.
  [[nodiscard]] std::vector<ExprId> tape(ExprId root) const;

  /// The columns `root` depends on, ascending.
  [[nodiscard]] std::vector<Index> variables_of(ExprId root) const;

  /// Readable form, for diagnostics and the CLI; not a serialization format.
  [[nodiscard]] std::string to_string(ExprId root) const;

  // ---- Evaluation (derivatives.cpp). `x` has num_variables() entries.
  [[nodiscard]] Evaluation evaluate(ExprId root, const std::vector<double>& x) const;

  /// The exact gradient at `x`, sparse. False, with `error` filled, when the value or a
  /// derivative is undefined there - sqrt at 0 has no derivative even though it has a value.
  bool gradient(ExprId root, const std::vector<double>& x, SparseEntries* out,
                Evaluation* error) const;

  /// The exact Hessian at `x`, its lower triangle, sparse. The same failure convention.
  bool hessian(ExprId root, const std::vector<double>& x, std::vector<HessianEntry>* out,
               Evaluation* error) const;

  // ---- Structure (convexity.cpp).
  /// The range of `root` over the box [lower, upper], by interval arithmetic.
  [[nodiscard]] Interval range(ExprId root, const std::vector<double>& lower,
                               const std::vector<double>& upper) const;

  /// Curvature over the box, by the documented composition rules.
  [[nodiscard]] Curvature curvature(ExprId root, const std::vector<double>& lower,
                                    const std::vector<double>& upper) const;

  /// Polynomial degree: 0 constant, 1 affine, 2 quadratic, ... and -1 for anything that is
  /// not a polynomial in the columns (a log, a non-integer power, a division by a variable).
  [[nodiscard]] int degree(ExprId root) const;

  /// Operations whose domain is restricted, with the argument range over the box that would
  /// violate it; empty when every operation is defined everywhere the box allows.
  [[nodiscard]] std::vector<std::string> domain_risks(ExprId root,
                                                      const std::vector<double>& lower,
                                                      const std::vector<double>& upper) const;

 private:
  ExprId intern(Node node);
  [[nodiscard]] bool is_constant(ExprId id, double* value = nullptr) const;
  /// True when the subtree has no restricted-domain operation, so dropping it (0 * f) cannot
  /// remove an error the caller's point would have raised.
  [[nodiscard]] bool defined_everywhere(ExprId id) const;
  /// Every node's range on the tape of `root`, indexed by node id, in one pass.
  [[nodiscard]] std::vector<Interval> ranges(ExprId root, const std::vector<double>& lower,
                                             const std::vector<double>& upper) const;
  /// False, recording why, when `id` cannot be used as an operand.
  bool usable(ExprId id, const char* operation);
  void mark_invalid(std::string message);

  Index num_variables_ = 0;
  std::string invalid_;
  std::vector<Node> nodes_;
  std::unordered_map<std::string, ExprId> interned_;
};

/// Central-difference check of the exact gradient and Hessian at `x` (#296 item 16). A
/// debugging facility, not a derivative source: its numbers are approximations and are only
/// ever compared against the exact ones. Returns the largest relative disagreement, or a
/// negative value when the point is outside the expression's domain.
struct DerivativeCheck {
  double gradient_error = 0.0;
  double hessian_error = 0.0;
  bool evaluated = false;
};
[[nodiscard]] DerivativeCheck check_derivatives(const ExpressionGraph& graph, ExprId root,
                                                const std::vector<double>& x,
                                                double step = 1e-5);

}  // namespace sankhya::nlp

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - conflict analysis for the branch and bound: learning, from a node proved
// infeasible, which of its branching decisions were to blame, and pruning every later node
// that repeats them (#292).
//
// References, written from the literature:
//   Achterberg, "Conflict analysis in mixed integer programming", Discrete Optimization 4(1),
//     2007 - conflicts from infeasible LPs and from propagation, and their use in the tree
//   Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 11 - the same, in SCIP
//   Witzig, Berthold & Heinz, "Experiments with conflict analysis in mixed integer
//     programming", CPAIOR 2017 - the dual-ray (Farkas) proof as the source of an LP conflict
//   Junker, "QuickXplain", AAAI 2004, and the deletion filter it improves on - minimising a
//     conflict by dropping one element at a time and re-checking
//
// WHAT A CONFLICT IS HERE. A set of bound literals, each "x_j <= v" or "x_j >= v" on an
// INTEGER column with an integral v, such that the model's rows, its global column bounds and
// those literals together admit no point. The learned constraint is the bound disjunction
// "at least one literal is false", which for an integer column means x_j >= v + 1 or
// x_j <= v - 1. It is GLOBALLY valid: it was proved from the global bounds, not from the
// node it was found at, so it may prune anywhere in the tree.
//
// WHY IT CANNOT CUT OFF A FEASIBLE POINT. A literal set is stored only after it is checked
// again from scratch, on the global bounds, by one of two sound arguments: bound propagation
// empties a box, or a Farkas multiplier vector - checked here, conservatively, not taken on
// the LP's word - shows the rows cannot be met inside the propagated box. Neither argument
// is weaker than the one that pruned the original node. Removing a stored conflict only
// ever weakens pruning, which is why the database may forget.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/types.hpp"

namespace sankhya::mip {

/// One bound decision: x_column <= value (is_upper) or x_column >= value.
struct ConflictLiteral {
  Index column = -1;
  bool is_upper = false;
  double value = 0.0;

  friend bool operator<(const ConflictLiteral& a, const ConflictLiteral& b) {
    if (a.column != b.column) return a.column < b.column;
    if (a.is_upper != b.is_upper) return a.is_upper < b.is_upper;
    return a.value < b.value;
  }
  friend bool operator==(const ConflictLiteral& a, const ConflictLiteral& b) {
    return a.column == b.column && a.is_upper == b.is_upper && a.value == b.value;
  }
};

/// What proved the node infeasible.
enum class ConflictSource { kPropagation, kLp };

[[nodiscard]] const char* to_string(ConflictSource source) noexcept;

/// One per (column, direction), keeping the tightest, sorted. The canonical form is what
/// duplicate detection compares, so two orders of the same decisions are one conflict.
[[nodiscard]] std::vector<ConflictLiteral> canonical(std::vector<ConflictLiteral> literals);

/// Does the box [lower, upper] make the literal true, i.e. is every point of the box on the
/// literal's side? `tolerance` is the integrality tolerance: bounds are integral here.
[[nodiscard]] inline bool literal_holds(const ConflictLiteral& literal,
                                        const std::vector<double>& lower,
                                        const std::vector<double>& upper, double tolerance) {
  const auto u = static_cast<std::size_t>(literal.column);
  return literal.is_upper ? upper[u] <= literal.value + tolerance
                          : lower[u] >= literal.value - tolerance;
}

/// Is the literal false everywhere in the box? Then the conflict it belongs to cannot fire.
[[nodiscard]] inline bool literal_refuted(const ConflictLiteral& literal,
                                          const std::vector<double>& lower,
                                          const std::vector<double>& upper, double tolerance) {
  const auto u = static_cast<std::size_t>(literal.column);
  return literal.is_upper ? lower[u] > literal.value + tolerance
                          : upper[u] < literal.value - tolerance;
}

/// A conflict against a box: every literal holds (the box is infeasible), all but one hold
/// (that one must be false, which is a bound), or neither.
struct ConflictVerdict {
  bool infeasible = false;
  /// Index into the conflict's literals of the one that must be false, or -1.
  int implied = -1;
};

[[nodiscard]] ConflictVerdict check_conflict(const std::vector<ConflictLiteral>& literals,
                                             const std::vector<double>& lower,
                                             const std::vector<double>& upper,
                                             double tolerance);

/// The bound an implied literal's negation gives on an integer column: x <= v false means
/// x >= v + 1, x >= v false means x <= v - 1.
[[nodiscard]] inline ConflictLiteral negation(const ConflictLiteral& literal) {
  return literal.is_upper ? ConflictLiteral{literal.column, false, literal.value + 1.0}
                          : ConflictLiteral{literal.column, true, literal.value - 1.0};
}

/// The rows aggregated by Farkas multipliers y: the proof says  d'x >= required  for every
/// point meeting the rows, where d = A'y. It is a contradiction on a box when the most d'x
/// can reach inside the box is below `required` by more than rounding.
struct FarkasProof {
  std::vector<double> d;
  double required = 0.0;
  double weight =
      0.0;  ///< sum |y_i| * max(1, |row bound|): the size a rounding error scales with
  bool usable = false;  ///< false when y leans on a row bound the row does not have
};

[[nodiscard]] FarkasProof aggregate_farkas(const Model& model, const std::vector<double>& y);

/// Does the proof contradict the box? CONSERVATIVE by construction: every column's
/// contribution is its true maximum over the box, however small its coefficient - a column
/// with a nonzero coefficient and an infinite bound on the side it leans on makes the proof
/// fail rather than being rounded away - and the margin must beat a tolerance that grows
/// with the size of the aggregation.
[[nodiscard]] bool farkas_contradicts(const FarkasProof& proof,
                                      const std::vector<double>& lower,
                                      const std::vector<double>& upper);

/// The deletion filter. `verify` must say whether a literal set is still proved infeasible.
/// Literals are tried in order and dropped when the rest still verifies; `keep_last` skips
/// the final literal (the node's own branching decision, which its feasible parent shows is
/// needed). At most `budget` calls to `verify`. Returns the surviving literals, in order.
[[nodiscard]] std::vector<ConflictLiteral> minimize_conflict(
    std::vector<ConflictLiteral> literals, bool keep_last, int budget,
    const std::function<bool(const std::vector<ConflictLiteral>&)>& verify, int* checks_used);

/// The learned conflicts, with their activity, a capacity and deterministic eviction.
class ConflictStore {
 public:
  struct Entry {
    std::vector<ConflictLiteral> literals;  ///< canonical
    ConflictSource source = ConflictSource::kPropagation;
    std::int64_t id = 0;         ///< insertion order; stable across runs
    std::int64_t created = 0;    ///< the node count when learned
    std::int64_t last_used = 0;  ///< the node count when it last pruned or tightened
    std::int64_t uses = 0;       ///< prunings plus tightenings
  };

  ConflictStore() = default;
  explicit ConflictStore(std::size_t capacity) : capacity_(capacity) {}

  /// Store a canonical conflict. Returns false for an exact duplicate of a stored one.
  /// A full store first evicts the least useful tenth: fewest uses, then longest unused,
  /// then oldest - a total order, so the same run evicts the same conflicts.
  bool add(std::vector<ConflictLiteral> literals, ConflictSource source, std::int64_t node);

  [[nodiscard]] std::size_t size() const { return entries_.size(); }
  [[nodiscard]] std::size_t capacity() const { return capacity_; }
  [[nodiscard]] std::vector<Entry>& entries() { return entries_; }
  [[nodiscard]] const std::vector<Entry>& entries() const { return entries_; }
  [[nodiscard]] std::int64_t evicted() const { return evicted_; }
  [[nodiscard]] std::int64_t duplicates() const { return duplicates_; }

 private:
  void evict();

  std::size_t capacity_ = 0;
  std::vector<Entry> entries_;
  std::set<std::vector<ConflictLiteral>> known_;
  std::int64_t next_id_ = 0;
  std::int64_t evicted_ = 0;
  std::int64_t duplicates_ = 0;
};

/// Running totals, for the log line and conflict_out.
struct ConflictStats {
  std::int64_t detected = 0;   ///< infeasible nodes analysed
  std::int64_t learned = 0;    ///< stored
  std::int64_t rejected = 0;   ///< the check from the global bounds did not prove it
  std::int64_t too_long = 0;   ///< proved, but longer than conflict_max_size
  std::int64_t minimized = 0;  ///< stored with fewer literals than the node had decisions
  std::int64_t checks = 0;     ///< verification calls, the analysis's unit of work
  std::int64_t nodes_pruned = 0;
  std::int64_t tightenings = 0;
  double seconds = 0.0;
  std::vector<std::int64_t> sizes;  ///< literal count of every learned conflict
};

/// The store and the statistics as JSON, for conflict_out. Literals are in the indices of
/// the model the search ran on (presolved, if presolve ran).
[[nodiscard]] std::string conflicts_to_json(const ConflictStore& store,
                                            const ConflictStats& stats);

}  // namespace sankhya::mip

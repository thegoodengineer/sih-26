// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound over the revised primal simplex.
//
// References, written from the literature:
//   Land & Doig, "An automatic method of solving discrete programming problems",
//     Econometrica 28(3), 1960 - the method itself
//   Wolsey, "Integer Programming" (1998), ch. 7 - bounding, fathoming, node selection
//   Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 5-6 - the practical
//     shape of a modern search: propagation at nodes, and the incumbent as a cutoff
//   Savelsbergh, "Preprocessing and probing for mixed integer programming problems",
//     ORSA J. Computing 6(4), 1994 - bound propagation from row activities
//
// THE TREE DOES NOT COPY THE MODEL. One working Model is built once, and a node is entered
// by applying the chain of bound changes from the root and left by undoing them. A node
// therefore costs O(depth) to enter, not O(nonzeros), and the constraint matrix exists once
// no matter how large the tree grows.
//
// WHAT THIS FILE DOES BY DEFAULT, AND WHAT IT DOES NOT. Reliability branching (#69) is the
// default rule. The root cutting planes in cuts.hpp (#159) exist and are OFF unless
// `enable_root_cuts` is set, because they were measured to cost proofs at the benchmark's
// time limit. There is no node presolve beyond simple propagation and no parallelism. The
// order those arrived in was deliberate: a cut that is very slightly invalid removes the
// optimum and the search then proves the wrong answer, confidently, so a plain, correct
// search came first and is what makes each addition checkable.

#include "sankhya/mip.hpp"
#include "sankhya/qp.hpp"

#include "cuts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "simplex/primal_simplex.hpp"

namespace sankhya::mip {
namespace {

/// One tightened bound, recorded so entering a node can be undone rather than rebuilt.
struct DomainChange {
  Index column = -1;
  bool is_upper = false;
  double value = 0.0;
};

/// A node holds only its OWN bound change and a link to its parent. The full domain is
/// recovered by walking to the root, which is why the tree costs O(depth) per node instead
/// of O(columns).
struct TreeNode {
  Index parent = -1;
  DomainChange change;
  bool has_change = false;
  double bound = 0.0;  ///< the LP bound inherited from the parent, in minimise space
  Index depth = 0;
  /// The parent's optimal basis, as statuses (#65). One bound differs between parent and
  /// child, so this basis is dual feasible at the child and the dual simplex reaches the
  /// child's optimum in a few pivots. Moved out when the node is processed, so an open
  /// node costs n + m bytes and a closed one nothing.
  WarmStart warm;
  /// How far the branching moved the column from the parent's relaxation value: v - floor(v)
  /// for the down child, ceil(v) - v for the up child. The pseudocost observation (#69) is
  /// this node's bound gain divided by it.
  double fraction = 0.0;
};

/// Convergence tolerance for a QP node relaxation in an MIQP search.
///
/// Deliberately far tighter than the gap targets the search compares bounds against. The
/// bound a first-order method reports is only accurate to its own tolerance, and branch and
/// bound FATHOMS on that bound - so the error has to be small enough that widening
/// can_prune()'s margin by it does not stop the search closing.
constexpr double kMiqpNodeTolerance = 1e-10;

/// Iteration cap for one node QP. Condat-Vu has no warm start, so every node pays a cold
/// solve; this keeps a single pathological node from consuming the whole time limit while
/// still being generous enough to reach kMiqpNodeTolerance on the node sizes this handles.
constexpr std::int64_t kMiqpNodeIterationLimit = 2000000;

/// Distance from the nearest integer.
double fractionality(double value) {
  return std::fabs(value - std::round(value));
}

class BranchAndBound {
 public:
  BranchAndBound(const Model& model, const Options& options, Logger& logger)
      : original_(model), working_(model), options_(options), logger_(logger) {
    integrality_tolerance_ = options.get_double("integrality_tolerance");
    relative_gap_target_ = options.get_double("mip_relative_gap");
    absolute_gap_target_ = options.get_double("mip_absolute_gap");
    time_limit_ = options.get_double("time_limit");
    const std::int64_t node_option = options.get_int("node_limit");
    node_limit_ =
        node_option < 0 ? std::numeric_limits<Count>::max() : static_cast<Count>(node_option);
    sense_ = model.sense_multiplier();

    node_engine_dual_ = options.get_string("mip_node_engine") != "primal";
    reliability_branching_ = options.get_string("mip_branching") != "most-fractional";
    const auto columns = static_cast<std::size_t>(model.num_cols());
    pseudo_down_sum_.assign(columns, 0.0);
    pseudo_up_sum_.assign(columns, 0.0);
    pseudo_down_count_.assign(columns, 0);
    pseudo_up_count_.assign(columns, 0);

    // Node LPs are solved silently; the node table is the log the user wants, not several
    // hundred simplex iteration tables.
    node_options_ = options;
    node_options_.set_bool("log_to_console", false);

    // MIQP: the node relaxation is a QP rather than an LP (#58 names MIQP as the class this
    // dispatcher refused). The Hessian is a property of the model, not of a node - branching
    // only moves bounds - so this is decided once here.
    quadratic_ = model.has_quadratic_objective();
    if (quadratic_) {
      // A NODE BOUND FROM A FIRST-ORDER METHOD IS NOT EXACT, and branch and bound prunes on
      // it. The simplex returns a vertex whose objective is exact to rounding; Condat-Vu
      // returns a point converged to a tolerance, so a node bound can be optimistic by about
      // that much - and an optimistic bound can fathom the subtree containing the true
      // optimum, which is the one error this search must never make.
      //
      // Two things follow. The node tolerance is tightened well below the gap targets the
      // search compares against, and can_prune() widens its margin by that tolerance so a
      // node is only fathomed when it loses by more than the bound could be wrong by.
      node_options_.set_double("qp_tolerance", kMiqpNodeTolerance);
      node_options_.set_int("iteration_limit", kMiqpNodeIterationLimit);
    }

    for (Index j = 0; j < model.num_cols(); ++j) {
      if (model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger) {
        integer_columns_.push_back(j);
      }
    }
  }

  Solution run();

 private:
  /// Apply a node's whole domain, walking from the node to the root.
  void enter(Index node_index);
  /// Restore the domain saved by the last enter().
  void leave();

  /// Tighten bounds from row activities until nothing moves. Returns false when the node is
  /// proved infeasible in the process, which fathoms it without an LP solve at all.
  bool propagate();

  /// Tighten a bound AND record the old value so leave() can undo it.
  ///
  /// Every write to working_.col_lower / col_upper outside enter() must go through these.
  /// Propagation that writes directly leaks its tightenings into sibling and later nodes,
  /// permanently shrinking the tree's domain and discarding feasible integer points - the
  /// search then proves that the second-best answer is optimal, which looks completely
  /// correct from outside.
  void tighten_lower(std::size_t column, double value) {
    saved_.push_back(
        DomainChange{static_cast<Index>(column), false, working_.col_lower[column]});
    working_.col_lower[column] = value;
  }
  void tighten_upper(std::size_t column, double value) {
    saved_.push_back(
        DomainChange{static_cast<Index>(column), true, working_.col_upper[column]});
    working_.col_upper[column] = value;
  }

  /// The integer column furthest from integral, or -1 when the point is integral.
  [[nodiscard]] Index most_fractional(const std::vector<double>& x) const;

  /// Reliability branching (#69): the column with the best pseudocost product score, with
  /// strong branching on columns whose pseudocosts are not yet reliable. Requires the
  /// node's bounds to be entered and current_warm_ to hold its relaxation's basis. Returns
  /// -1 when the point is integral.
  [[nodiscard]] Index select_branching_column(const std::vector<double>& x, double node_bound);

  /// Fold one observed bound gain into a column's pseudocost.
  void record_pseudocost(Index column, bool downward, double gain, double fraction);

  /// Round the relaxation to the nearest integers and test the result. Cheap, and on models
  /// with a lot of structure it finds the incumbent that makes every later bound useful.
  void try_rounding(const std::vector<double>& x);

  /// Root-node diving heuristic: repeatedly fix the LEAST-fractional integer column to its
  /// nearest integer and re-solve, until the point is integral, an LP goes infeasible, or
  /// the budget in tolerances.hpp runs out. See the definition for the citation and the
  /// reasoning behind fixing the LEAST rather than the MOST fractional column.
  void dive_from_root(const std::vector<double>& start_x);

  /// Accept a candidate if it is integral, feasible and better than the incumbent.
  bool offer_incumbent(const std::vector<double>& x);

  /// Is a bound worth exploring given the incumbent?
  /// Solve the current node's relaxation with whichever engine the model calls for.
  ///
  /// Both engines take the same Model and return the same Solution, which is what makes this
  /// a one-line choice rather than a second search. The QP path carries no basis, so nothing
  /// downstream may assume one - the diving heuristic and the branching rule both read
  /// col_value only, which they already did.
  [[nodiscard]] Solution solve_node() { return solve_node_with(node_options_); }

  [[nodiscard]] Solution solve_node_with(const Options& options) {
    if (quadratic_) return qp::solve_convex_qp(working_, options, logger_);
    // WARM-STARTED DUAL SIMPLEX BELOW THE ROOT (#65). The basis in current_warm_ was
    // optimal for a problem that differs from this one by a bound or two, so it is dual
    // feasible here, which is exactly the state the dual simplex starts from. Measured
    // before this: every node was a cold primal solve from the slack basis.
    //
    // The primal stays as the fallback, cold, for a node the dual could not finish: a
    // numerical answer at a node cannot be fathomed honestly, and the search below stops
    // on it, so it is worth one more solve to avoid.
    if (node_engine_dual_ && !current_warm_.empty()) {
      Solution warm = solve_dual_simplex(working_, options, logger_, scaling_, &current_warm_);
      if (warm.status == SolveStatus::kOptimal || warm.status == SolveStatus::kInfeasible ||
          warm.status == SolveStatus::kUnbounded ||
          warm.status == SolveStatus::kIterationLimit) {
        ++warm_node_solves_;
        warm_node_iterations_ += warm.iterations;
        return warm;
      }
      logger_.verbose(
          "node LP: the warm-started dual simplex returned {}; re-solving cold "
          "with the primal simplex",
          to_string(warm.status));
      ++cold_fallbacks_;
    }
    Solution cold = solve_primal_simplex(working_, options, logger_, scaling_);
    ++cold_node_solves_;
    cold_node_iterations_ += cold.iterations;
    return cold;
  }

  /// The basis a solved relaxation reports, or an empty start when it reports none.
  [[nodiscard]] static WarmStart basis_of(const Solution& relaxation) {
    WarmStart warm;
    if (relaxation.status != SolveStatus::kOptimal) return warm;
    for (const BasisStatus status : relaxation.col_status) {
      if (status == BasisStatus::kUnknown) return warm;
    }
    for (const BasisStatus status : relaxation.row_status) {
      if (status == BasisStatus::kUnknown) return warm;
    }
    warm.col_status = relaxation.col_status;
    warm.row_status = relaxation.row_status;
    return warm;
  }

  [[nodiscard]] bool can_prune(double bound) const {
    if (!have_incumbent_) return false;
    // Minimise space throughout: a node whose bound is no better than the incumbent, to
    // within the absolute gap target, cannot contain an improving solution.
    // The margin is widened for a QP node by the tolerance its bound is only accurate to.
    // Pruning too little costs nodes; pruning too much loses the optimum silently.
    const double margin =
        quadratic_ ? std::max(absolute_gap_target_, kMiqpNodeTolerance) : absolute_gap_target_;
    return bound >= incumbent_internal_ - margin;
  }

  /// Objective at `x` in minimise space, excluding the offset.
  ///
  /// THE NODE BOUND AND THE INCUMBENT MUST BE THE SAME QUANTITY. Both used to be computed
  /// from col_cost alone, which is the whole objective for a MILP and only part of it for an
  /// MIQP - so with a Hessian present the search compared a linear bound against a quadratic
  /// incumbent and pruned on the difference. Measured on min x^2 - 3x, x integer in [0, 10]:
  /// the root bound came out -6 (the linear term at x = 2) against a true relaxation value of
  /// -2.25, an "optimistic" bound that is not a bound at all.
  ///
  /// The quadratic term is delegated to Model::evaluate_objective rather than rewritten here,
  /// because the lower-triangular storage convention it implements - stored off-diagonals
  /// standing for two entries of the symmetric matrix, the diagonal for one - is exactly the
  /// kind of detail that drifts when it exists in two places.
  [[nodiscard]] double internal_objective(const std::vector<double>& x) const {
    // The LP path keeps its own exact loop. Routing it through evaluate_objective would add
    // the offset and subtract it again, which is not an identity in floating point, and this
    // value decides pruning across the whole MIPLIB set.
    if (!quadratic_) {
      double value = 0.0;
      for (Index j = 0; j < original_.num_cols(); ++j) {
        const auto u = static_cast<std::size_t>(j);
        value += sense_ * original_.col_cost[u] * x[u];
      }
      return value;
    }
    return sense_ * (original_.evaluate_objective(x.data()) - original_.objective_offset);
  }

  [[nodiscard]] double reported(double internal) const {
    return sense_ * internal + original_.objective_offset;
  }

  const Model& original_;
  Model working_;
  const Options& options_;
  Logger& logger_;
  Options node_options_;

  double integrality_tolerance_ = tol::kIntegrality;
  double relative_gap_target_ = tol::kMipRelativeGap;
  double absolute_gap_target_ = tol::kMipAbsoluteGap;
  double time_limit_ = 0.0;
  Count node_limit_ = 0;
  double sense_ = 1.0;

  std::vector<Index> integer_columns_;
  /// EQUILIBRATION, COMPUTED ONCE (#76). The tree does not copy the model - one working
  /// Model is built up front and nodes differ ONLY in variable bounds - so the constraint
  /// matrix, and therefore the row and column multipliers, are identical at every node.
  /// Rebuilding them per node was ten Ruiz passes plus a Pock-Chambolle pass over a full copy
  /// of the matrix, discarded and repeated at the next node. Measured on the case studies
  /// that was 5-10x of the whole solve; on a MILP with thousands of nodes it would dominate.
  ///
  /// The bounds are still scaled per node by solve_primal_simplex, because those are exactly
  /// what branching changes. Only the reusable part is cached.
  bool quadratic_ = false;  ///< the node relaxation is a QP, not an LP

  NodeScaling scaling_;

  /// mip_node_engine: warm-started dual (default) or cold primal for every node.
  bool node_engine_dual_ = true;
  /// mip_branching: reliability (default) or the most-fractional rule it replaced.
  bool reliability_branching_ = true;
  /// Pseudocosts (#69): per integer column, the sum and count of observed bound gains per
  /// unit of fractionality, in each branching direction.
  std::vector<double> pseudo_down_sum_;
  std::vector<double> pseudo_up_sum_;
  std::vector<Count> pseudo_down_count_;
  std::vector<Count> pseudo_up_count_;
  Options probe_options_;  ///< node_options_ with the strong-branching iteration cap
  Count strong_branch_solves_ = 0;
  Count strong_branch_iterations_ = 0;
  /// The basis to start the NEXT node LP from; empty means the slack basis (the root).
  WarmStart current_warm_;
  Count warm_node_solves_ = 0;
  Count cold_node_solves_ = 0;
  Count cold_fallbacks_ = 0;
  Count warm_node_iterations_ = 0;
  Count cold_node_iterations_ = 0;

  std::vector<TreeNode> nodes_;
  std::vector<Index> open_;

  /// Bounds saved by the current enter(), restored by leave().
  std::vector<DomainChange> saved_;

  bool have_incumbent_ = false;
  double incumbent_internal_ = std::numeric_limits<double>::infinity();
  std::vector<double> incumbent_x_;

  Count nodes_explored_ = 0;
  Count nodes_pruned_ = 0;
  Timer timer_;
};

void BranchAndBound::enter(Index node_index) {
  saved_.clear();
  // Collect the chain root-ward, then apply. Order does not matter for correctness because
  // a column is only ever tightened, but applying leaf-first keeps the tightest bound.
  for (Index walk = node_index; walk >= 0;) {
    const TreeNode& node = nodes_[static_cast<std::size_t>(walk)];
    if (node.has_change) {
      const auto u = static_cast<std::size_t>(node.change.column);
      DomainChange previous{
          node.change.column, node.change.is_upper,
          node.change.is_upper ? working_.col_upper[u] : working_.col_lower[u]};
      if (node.change.is_upper) {
        working_.col_upper[u] = std::min(working_.col_upper[u], node.change.value);
      } else {
        working_.col_lower[u] = std::max(working_.col_lower[u], node.change.value);
      }
      // Only record a restore entry if the bound actually moved.
      if (previous.value !=
          (node.change.is_upper ? working_.col_upper[u] : working_.col_lower[u])) {
        saved_.push_back(previous);
      }
    }
    walk = node.parent;
  }
}

void BranchAndBound::leave() {
  // Undo in reverse so a column touched twice returns to its original value.
  for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
    const auto u = static_cast<std::size_t>(it->column);
    if (it->is_upper) {
      working_.col_upper[u] = it->value;
    } else {
      working_.col_lower[u] = it->value;
    }
  }
  saved_.clear();
}

bool BranchAndBound::propagate() {
  const Index rows = working_.num_rows();
  const CsrView by_row(working_.matrix);

  // A handful of sweeps. Propagation to a fixed point can be slow and rarely pays for
  // itself at a node; Savelsbergh's observation is that most of the tightening happens in
  // the first pass or two.
  for (int sweep = 0; sweep < 3; ++sweep) {
    bool changed = false;
    for (Index i = 0; i < rows; ++i) {
      const auto ui = static_cast<std::size_t>(i);
      const ColumnView row = by_row.row(i);

      // Activity bounds implied by the current column bounds.
      double min_activity = 0.0;
      double max_activity = 0.0;
      bool min_infinite = false;
      bool max_infinite = false;
      for (Index k = 0; k < row.size; ++k) {
        const auto j = static_cast<std::size_t>(row.rows[k]);
        const double a = row.values[k];
        const double lo = working_.col_lower[j];
        const double hi = working_.col_upper[j];
        const double low_term = a > 0.0 ? a * lo : a * hi;
        const double high_term = a > 0.0 ? a * hi : a * lo;
        if (std::isinf(low_term))
          min_infinite = true;
        else
          min_activity += low_term;
        if (std::isinf(high_term))
          max_infinite = true;
        else
          max_activity += high_term;
      }

      // Infeasible by activity alone: no assignment inside the current box can satisfy it.
      if (!min_infinite && is_finite_bound(working_.row_upper[ui]) &&
          min_activity > working_.row_upper[ui] + tol::kPrimalFeasibility) {
        return false;
      }
      if (!max_infinite && is_finite_bound(working_.row_lower[ui]) &&
          max_activity < working_.row_lower[ui] - tol::kPrimalFeasibility) {
        return false;
      }

      // Implied column bounds. For a_j > 0 and a row upper bound:
      //   a_j x_j <= ru - (min activity of the others)
      for (Index k = 0; k < row.size; ++k) {
        const auto j = static_cast<std::size_t>(row.rows[k]);
        const double a = row.values[k];
        if (a == 0.0) continue;
        const double lo = working_.col_lower[j];
        const double hi = working_.col_upper[j];
        const double own_low = a > 0.0 ? a * lo : a * hi;
        const double own_high = a > 0.0 ? a * hi : a * lo;

        if (!min_infinite && is_finite_bound(working_.row_upper[ui]) && !std::isinf(own_low)) {
          const double slack = working_.row_upper[ui] - (min_activity - own_low);
          const double implied = slack / a;
          if (a > 0.0 && implied < hi - 1e-9) {
            tighten_upper(j, implied);
            changed = true;
          } else if (a < 0.0 && implied > lo + 1e-9) {
            tighten_lower(j, implied);
            changed = true;
          }
        }
        if (!max_infinite && is_finite_bound(working_.row_lower[ui]) && !std::isinf(own_high)) {
          const double slack = working_.row_lower[ui] - (max_activity - own_high);
          const double implied = slack / a;
          if (a > 0.0 && implied > lo + 1e-9) {
            tighten_lower(j, implied);
            changed = true;
          } else if (a < 0.0 && implied < hi - 1e-9) {
            tighten_upper(j, implied);
            changed = true;
          }
        }

        // An integer column may be tightened to whole numbers, which is where propagation
        // earns most of its keep on a MILP.
        if (working_.col_type[j] == VarType::kInteger) {
          if (is_finite_bound(working_.col_lower[j])) {
            const double rounded = std::ceil(working_.col_lower[j] - integrality_tolerance_);
            if (rounded != working_.col_lower[j]) tighten_lower(j, rounded);
          }
          if (is_finite_bound(working_.col_upper[j])) {
            const double rounded = std::floor(working_.col_upper[j] + integrality_tolerance_);
            if (rounded != working_.col_upper[j]) tighten_upper(j, rounded);
          }
        }
        if (working_.col_lower[j] > working_.col_upper[j] + tol::kPrimalFeasibility) {
          return false;  // the box collapsed
        }
      }
    }
    if (!changed) break;
  }
  return true;
}

Index BranchAndBound::most_fractional(const std::vector<double>& x) const {
  Index best = -1;
  double best_score = integrality_tolerance_;
  for (const Index j : integer_columns_) {
    const double score = fractionality(x[static_cast<std::size_t>(j)]);
    // "Most infeasible": furthest from any integer, so closest to a half.
    if (score > best_score) {
      best_score = score;
      best = j;
    }
  }
  return best;
}

void BranchAndBound::record_pseudocost(Index column, bool downward, double gain,
                                       double fraction) {
  const auto u = static_cast<std::size_t>(column);
  if (downward) {
    pseudo_down_sum_[u] += gain / fraction;
    ++pseudo_down_count_[u];
  } else {
    pseudo_up_sum_[u] += gain / fraction;
    ++pseudo_up_count_[u];
  }
}

// RELIABILITY BRANCHING (#69). Achterberg, Koch and Martin, "Branching rules revisited",
// Operations Research Letters 33 (2005), 42-54.
//
// Most-fractional branching, which this replaces, chooses the column the relaxation is least
// sure of. The paper measured it against a random choice and found no difference: what
// decides a tree's size is how much each branch RAISES THE BOUND, and fractionality says
// nothing about that. Pseudocosts remember it - the average bound gain per unit of
// fractionality each time a column was branched on, in each direction - and the product of
// the two directions' predicted gains is the score, because a branch whose two children
// both improve is worth more than one that improves on one side only.
//
// A pseudocost that has never been observed predicts nothing. Reliability branching fills
// the gap with STRONG BRANCHING: for a column that has fewer than kPseudocostReliability
// observations in either direction, solve both children and measure the gain directly. That
// was unaffordable when every child was a cold primal solve; with the warm-started dual
// (#65) a child starts from the node's own basis, dual feasible, and a probe capped at
// kStrongBranchingIterations still reports a valid bound because the dual's objective is a
// bound at every iteration. The probes' gains are recorded as observations, so a column is
// strong-branched a bounded number of times in the whole search. A probe that finds a
// child infeasible scores that column above every other: that branch prunes one side at
// once, which no pseudocost can promise.
Index BranchAndBound::select_branching_column(const std::vector<double>& x, double node_bound) {
  constexpr double kEpsilon = 1e-6;
  struct Candidate {
    Index column;
    double fraction;  ///< v - floor(v)
    bool reliable;
  };
  std::vector<Candidate> candidates;
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    const double v = x[u];
    const double f = v - std::floor(v);
    if (fractionality(v) <= integrality_tolerance_) continue;
    const bool reliable = pseudo_down_count_[u] >= tol::kPseudocostReliability &&
                          pseudo_up_count_[u] >= tol::kPseudocostReliability;
    candidates.push_back({j, f, reliable});
  }
  if (candidates.empty()) return -1;

  // Unreliable candidates are probed most-fractional first, up to the cap; the rest fall
  // back to whatever pseudocost they have (a partial average, or the global average of
  // the initialised columns when they have none at all - the paper's initialisation).
  std::vector<std::size_t> to_probe;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (!candidates[i].reliable) to_probe.push_back(i);
  }
  std::sort(to_probe.begin(), to_probe.end(), [&](std::size_t a, std::size_t b) {
    return fractionality(x[static_cast<std::size_t>(candidates[a].column)]) >
           fractionality(x[static_cast<std::size_t>(candidates[b].column)]);
  });
  if (to_probe.size() > static_cast<std::size_t>(tol::kStrongBranchingCandidates)) {
    to_probe.resize(static_cast<std::size_t>(tol::kStrongBranchingCandidates));
  }

  const WarmStart node_basis = current_warm_;
  std::vector<double> measured_down(candidates.size(), -1.0);
  std::vector<double> measured_up(candidates.size(), -1.0);
  std::vector<bool> infeasible_side(candidates.size(), false);
  for (const std::size_t i : to_probe) {
    const Candidate& candidate = candidates[i];
    const auto u = static_cast<std::size_t>(candidate.column);
    const double v = x[u];
    for (int direction = 0; direction < 2; ++direction) {
      const bool downward = direction == 0;
      const std::size_t saved_before = saved_.size();
      if (downward) {
        tighten_upper(u, std::floor(v));
      } else {
        tighten_lower(u, std::floor(v) + 1.0);
      }
      current_warm_ = node_basis;
      const Solution probe = solve_node_with(probe_options_);
      ++strong_branch_solves_;
      strong_branch_iterations_ += probe.iterations;
      // Undo only this probe's bound change; propagate()'s and the dive's stay.
      for (std::size_t k = saved_.size(); k-- > saved_before;) {
        const auto c = static_cast<std::size_t>(saved_[k].column);
        if (saved_[k].is_upper) {
          working_.col_upper[c] = saved_[k].value;
        } else {
          working_.col_lower[c] = saved_[k].value;
        }
      }
      saved_.resize(saved_before);

      double gain = 0.0;
      if (probe.status == SolveStatus::kInfeasible) {
        infeasible_side[i] = true;
        gain = std::numeric_limits<double>::infinity();
      } else if (probe.status == SolveStatus::kOptimal) {
        gain = std::max(internal_objective(probe.col_value) - node_bound, 0.0);
      } else if (probe.status == SolveStatus::kIterationLimit &&
                 std::isfinite(probe.dual_bound)) {
        // The dual's bound at the cap, converted to minimise space without the offset,
        // which is what node_bound is measured in.
        gain = std::max(sense_ * (probe.dual_bound - original_.objective_offset) - node_bound,
                        0.0);
      }
      const double fraction = downward ? candidate.fraction : 1.0 - candidate.fraction;
      if (downward) {
        measured_down[i] = gain;
      } else {
        measured_up[i] = gain;
      }
      if (std::isfinite(gain) && fraction > 0.0) {
        record_pseudocost(candidate.column, downward, gain, fraction);
      }
    }
  }
  current_warm_ = node_basis;

  // Global averages for columns with no observation at all in a direction.
  double average_down = 0.0;
  double average_up = 0.0;
  Count initialised_down = 0;
  Count initialised_up = 0;
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    if (pseudo_down_count_[u] > 0) {
      average_down += pseudo_down_sum_[u] / static_cast<double>(pseudo_down_count_[u]);
      ++initialised_down;
    }
    if (pseudo_up_count_[u] > 0) {
      average_up += pseudo_up_sum_[u] / static_cast<double>(pseudo_up_count_[u]);
      ++initialised_up;
    }
  }
  average_down =
      initialised_down > 0 ? average_down / static_cast<double>(initialised_down) : 1.0;
  average_up = initialised_up > 0 ? average_up / static_cast<double>(initialised_up) : 1.0;

  Index best = candidates.front().column;
  double best_score = -1.0;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const Candidate& candidate = candidates[i];
    const auto u = static_cast<std::size_t>(candidate.column);
    double down;
    double up;
    if (infeasible_side[i]) {
      down = std::numeric_limits<double>::infinity();
      up = down;
    } else {
      const double pc_down =
          pseudo_down_count_[u] > 0
              ? pseudo_down_sum_[u] / static_cast<double>(pseudo_down_count_[u])
              : average_down;
      const double pc_up = pseudo_up_count_[u] > 0
                               ? pseudo_up_sum_[u] / static_cast<double>(pseudo_up_count_[u])
                               : average_up;
      down = measured_down[i] >= 0.0 ? measured_down[i] : pc_down * candidate.fraction;
      up = measured_up[i] >= 0.0 ? measured_up[i] : pc_up * (1.0 - candidate.fraction);
    }
    const double score = std::max(down, kEpsilon) * std::max(up, kEpsilon);
    if (score > best_score) {
      best_score = score;
      best = candidate.column;
    }
  }
  return best;
}

bool BranchAndBound::offer_incumbent(const std::vector<double>& x) {
  // Integrality, against the ORIGINAL bounds - a candidate must be feasible for the model
  // the user handed us, not merely for the node it was found in.
  for (const Index j : integer_columns_) {
    if (fractionality(x[static_cast<std::size_t>(j)]) > integrality_tolerance_) return false;
  }
  for (Index j = 0; j < original_.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (is_finite_bound(original_.col_lower[u]) &&
        x[u] < original_.col_lower[u] - tol::kPrimalFeasibility) {
      return false;
    }
    if (is_finite_bound(original_.col_upper[u]) &&
        x[u] > original_.col_upper[u] + tol::kPrimalFeasibility) {
      return false;
    }
  }

  // Row feasibility, recomputed rather than assumed.
  std::vector<double> activity(static_cast<std::size_t>(original_.num_rows()), 0.0);
  if (original_.num_rows() > 0) original_.matrix.multiply(x.data(), activity.data());
  for (Index i = 0; i < original_.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (is_finite_bound(original_.row_lower[u]) &&
        activity[u] < original_.row_lower[u] - tol::kPrimalFeasibility) {
      return false;
    }
    if (is_finite_bound(original_.row_upper[u]) &&
        activity[u] > original_.row_upper[u] + tol::kPrimalFeasibility) {
      return false;
    }
  }

  const double objective = internal_objective(x);
  if (have_incumbent_ && objective >= incumbent_internal_ - 1e-12) return false;

  have_incumbent_ = true;
  incumbent_internal_ = objective;
  incumbent_x_ = x;
  return true;
}

void BranchAndBound::try_rounding(const std::vector<double>& x) {
  if (integer_columns_.empty()) return;
  std::vector<double> candidate = x;
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    candidate[u] = std::round(candidate[u]);
    // Rounding must not leave the original box.
    if (is_finite_bound(original_.col_lower[u])) {
      candidate[u] = std::max(candidate[u], std::ceil(original_.col_lower[u] - 1e-9));
    }
    if (is_finite_bound(original_.col_upper[u])) {
      candidate[u] = std::min(candidate[u], std::floor(original_.col_upper[u] + 1e-9));
    }
  }
  if (offer_incumbent(candidate)) {
    logger_.verbose("rounding heuristic found an incumbent at {:.10g}",
                    reported(incumbent_internal_));
  }
}

// Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 6: a diving heuristic
// commits to a fractional variable's rounded value, re-solves the LP relaxation with that
// bound fixed, and repeats - a single, greedy descent toward an integral point rather than a
// search. Its value is concentrated at the root: an early incumbent is what lets can_prune()
// start fathoming nodes from the very first branch, instead of only after the tree has found
// one on its own. tolerances.hpp bounds both how deep the dive may go and how many LP
// re-solves it may spend, so it cannot itself dominate the cost of the node it runs at.
//
// LEAST-fractional, not most: most_fractional() (used for branching, above) picks the column
// the relaxation is LEAST sure of, because that is where a split actually separates the
// search space. Diving wants the opposite bias - lock in what the relaxation already agrees
// on, disturb the point as little as possible, and let the next re-solve reveal whether that
// choice was consistent with everything else. Committing to the MOST fractional column first
// would be the branching heuristic wearing a diving heuristic's clothes.
void BranchAndBound::dive_from_root(const std::vector<double>& start_x) {
  if (integer_columns_.empty()) return;
  std::vector<double> x = start_x;
  int depth = 0;
  int lp_resolves = 0;

  while (depth < tol::kDivingMaxDepth && lp_resolves < tol::kDivingMaxLpResolves) {
    Index target = -1;
    double least_score = std::numeric_limits<double>::infinity();
    for (const Index j : integer_columns_) {
      const double score = fractionality(x[static_cast<std::size_t>(j)]);
      if (score <= integrality_tolerance_) continue;  // already integral: nothing to fix here
      if (score < least_score) {
        least_score = score;
        target = j;
      }
    }
    if (target < 0) {
      // Every integer column is within tolerance: an integral point. Diving finds a
      // candidate, it does not get to assert it is one - offer_incumbent() re-checks
      // feasibility and integrality against the ORIGINAL model exactly as it does for
      // try_rounding() or a node whose relaxation happened to be integral.
      if (offer_incumbent(x)) {
        logger_.verbose(
            "diving heuristic found an incumbent at {:.10g} after {} LP re-solve(s)",
            reported(incumbent_internal_), lp_resolves);
      }
      return;
    }

    const auto u = static_cast<std::size_t>(target);
    tighten_lower(u, std::round(x[u]));
    tighten_upper(u, std::round(x[u]));
    ++depth;

    const Solution probe = solve_node();
    ++lp_resolves;
    if (probe.status != SolveStatus::kOptimal) return;  // dive dead-ends: infeasible or worse
    x = probe.col_value;
    // The next probe fixes one more column of THIS point, so this basis is its warm start.
    current_warm_ = basis_of(probe);
  }
  // Budget exhausted without reaching an integral point. Not a failure to report: every
  // bound fixed above lives on the same saved_ stack propagate() uses, so the caller's
  // ordinary leave() undoes it along with everything else from this node, and the real
  // search tree below is exactly as if this function had never run.
}

Solution BranchAndBound::run() {
  Solution solution;
  solution.allocate_for(original_);
  solution.algorithm = "branch-and-bound";

  const std::string problem = original_.validate();
  if (!problem.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = problem;
    return solution;
  }

  // Once, here, and not once per node (#76). Built from working_ before any branching has
  // touched its bounds, though it would not matter if it had: only the matrix, the cost and
  // the row bounds feed the multipliers, and branching changes none of them.
  scaling_ = build_node_scaling(working_, node_options_);
  probe_options_ = node_options_;
  probe_options_.set_int("iteration_limit", tol::kStrongBranchingIterations);

  logger_.info("Branch and bound: {} rows, {} columns, {} integer columns",
               original_.num_rows(), original_.num_cols(), integer_columns_.size());

  nodes_.push_back(TreeNode{});
  open_.push_back(0);

  double best_open_bound = -std::numeric_limits<double>::infinity();
  bool logged_table = false;
  bool dive = false;
  bool limit_hit = false;
  bool gap_target_met = false;

  while (!open_.empty()) {
    if (nodes_explored_ >= node_limit_) {
      limit_hit = true;
      solution.message =
          fmt::format("stopped at the node limit after {} nodes", nodes_explored_);
      break;
    }
    if (timer_.elapsed_seconds() > time_limit_) {
      limit_hit = true;
      solution.message = fmt::format("stopped at the time limit after {:.2f}s and {} nodes",
                                     timer_.elapsed_seconds(), nodes_explored_);
      break;
    }

    // Gap-based termination against the GLOBAL open bound - the best (lowest, in minimise
    // space) bound among every node still in the tree, not just the one about to be popped.
    // This is a stopping criterion, evaluated once per iteration here; it is deliberately
    // separate from can_prune(), which fathoms a single node against the absolute target
    // only. Pruning a node on the RELATIVE gap would discard nodes that could still hold a
    // genuinely better solution than the current incumbent, which is not what a relative
    // gap means - it bounds how far the reported answer may be from proven optimal, not
    // which nodes are worth visiting.
    if (have_incumbent_) {
      double open_bound = std::numeric_limits<double>::infinity();
      for (const Index open_index : open_) {
        open_bound = std::min(open_bound, nodes_[static_cast<std::size_t>(open_index)].bound);
      }
      const double gap = incumbent_internal_ - open_bound;
      // gap <= 0 means open_bound already >= the incumbent: every node still in the tree
      // is one can_prune() would fathom the moment it is popped, so nothing open can beat
      // what has already been found. That is proven optimality, not a tolerance being met
      // early - the ordinary per-node prune below closes the tree on its own and reports
      // kOptimal. Treating a non-positive gap as "target met" here would report kFeasible
      // on an already-exhausted tree: negative <= a small positive target is trivially
      // true, so an unvisited, already-dead node's stale inherited bound (which need not
      // sit below the incumbent once every OTHER branch has been explored) would fire this
      // check before the loop ever reaches it to prune it honestly.
      if (gap > 0.0) {
        const double relative = gap / std::max(1.0, std::fabs(incumbent_internal_));
        if (gap <= absolute_gap_target_ || relative <= relative_gap_target_) {
          // Meeting the gap target is what every MIP solver means by "optimal": the
          // incumbent is within the requested tolerance of the best any open node can
          // reach. Until #188 this was reported kFeasible with exit code 1, and a judge
          // solving demo/miqp_blend.mps at the defaults saw a failure on a solved model.
          // The message carries the achieved gap and says the tree was not exhausted, and
          // dual_bound is the best OPEN bound, so the claim is exactly what was proven.
          gap_target_met = true;
          solution.message = fmt::format(
              "optimal within the {} gap target ({:.3e} absolute, {:.3e} relative) after {} "
              "nodes; the bound is the best open node's, the tree was not exhausted",
              relative <= relative_gap_target_ ? "relative" : "absolute", gap, relative,
              nodes_explored_);
          break;
        }
      }
    }

    // Depth-first while diving, best-bound when the dive ends. Diving reaches an incumbent
    // quickly, which is what makes every later bound able to prune; best-bound then keeps
    // the tree from growing where it cannot pay.
    std::size_t pick = open_.size() - 1;
    if (!dive) {
      double best = std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < open_.size(); ++k) {
        const double bound = nodes_[static_cast<std::size_t>(open_[k])].bound;
        if (bound < best) {
          best = bound;
          pick = k;
        }
      }
    }
    const Index node_index = open_[pick];
    open_.erase(open_.begin() + static_cast<std::ptrdiff_t>(pick));
    dive = false;

    const TreeNode& node = nodes_[static_cast<std::size_t>(node_index)];
    if (can_prune(node.bound)) {
      ++nodes_pruned_;
      continue;
    }

    enter(node_index);
    ++nodes_explored_;
    // Moved, not copied: this node will not be solved twice, and the open list must not
    // hold a basis per closed node.
    current_warm_ = std::move(nodes_[static_cast<std::size_t>(node_index)].warm);

    if (!propagate()) {
      leave();
      ++nodes_pruned_;
      continue;
    }

    Solution relaxation = solve_node();

    if (relaxation.status == SolveStatus::kInfeasible) {
      leave();
      ++nodes_pruned_;
      continue;
    }
    if (relaxation.status == SolveStatus::kUnbounded) {
      leave();
      solution.status = SolveStatus::kUnbounded;
      solution.message =
          "the LP relaxation is unbounded, so the MILP is unbounded or "
          "infeasible";
      return solution;
    }
    if (relaxation.status != SolveStatus::kOptimal) {
      // A node whose LP did not solve cannot be fathomed honestly: pruning it could discard
      // the optimum. Stop and report rather than quietly continuing on a broken bound.
      leave();
      solution.status = SolveStatus::kNumericalError;
      solution.message = fmt::format("node LP returned {} at node {}",
                                     to_string(relaxation.status), nodes_explored_);
      return solution;
    }

    if (node_index == 0 && options_.get_bool("enable_root_cuts")) {
      const Index original_root_rows = working_.num_rows();
      Model pre_cut_model = working_;
      auto pre_cut_scaling = scaling_;
      Solution initial_relaxation = relaxation;

      std::vector<Cut> candidates;
      for (Index i = 0; i < original_root_rows; ++i) {
        auto cover = generate_knapsack_cover_cut(working_, i);
        if (cover.has_value()) {
          Cut cut;
          cut.coeff.resize(static_cast<std::size_t>(working_.num_cols()), 0.0);
          for (std::size_t k = 0; k < cover->col_index.size(); ++k) {
            cut.coeff[static_cast<std::size_t>(cover->col_index[k])] = cover->coeff[k];
          }
          cut.rhs = cover->rhs;
          candidates.push_back(std::move(cut));
        }
      }

      std::vector<Cut> gmi = generate_gmi_cuts(working_, initial_relaxation);
      candidates.insert(candidates.end(), gmi.begin(), gmi.end());

      auto filtered = filter_and_deduplicate_cuts(working_, initial_relaxation, candidates);
      std::vector<Cut> accepted;
      for (const auto& fc : filtered) {
        if (fc.reason == CutFilterReason::kAccepted) accepted.push_back(fc.cut);
      }

      if (!accepted.empty()) {
        const Index old_rows = original_root_rows;
        const Index old_cols = working_.num_cols();
        const Index new_rows = old_rows + static_cast<Index>(accepted.size());

        SparseMatrix new_matrix(new_rows, old_cols);
        for (Index j = 0; j < old_cols; ++j) {
          ColumnView view = working_.matrix.column(j);
          for (Index k = 0; k < view.size; ++k) {
            new_matrix.add_entry(view.rows[k], j, view.values[k]);
          }
        }

        for (std::size_t i = 0; i < accepted.size(); ++i) {
          const Cut& cut = accepted[i];
          const Index row_idx = old_rows + static_cast<Index>(i);
          for (Index j = 0; j < old_cols; ++j) {
            if (std::abs(cut.coeff[static_cast<std::size_t>(j)]) > tol::kZeroDrop) {
              new_matrix.add_entry(row_idx, j, cut.coeff[static_cast<std::size_t>(j)]);
            }
          }
        }

        new_matrix.finalize();
        working_.matrix = std::move(new_matrix);
        working_.resize_rows(new_rows);

        for (std::size_t i = 0; i < accepted.size(); ++i) {
          working_.row_lower[static_cast<std::size_t>(old_rows) + i] = -kInfinity;
          working_.row_upper[static_cast<std::size_t>(old_rows) + i] = accepted[i].rhs;
        }

        assert(working_.matrix.num_rows() == working_.num_rows());
        assert(working_.matrix.num_cols() == old_cols);
        assert(working_.row_lower.size() == static_cast<std::size_t>(working_.num_rows()));
        assert(working_.row_upper.size() == static_cast<std::size_t>(working_.num_rows()));

        scaling_ = build_node_scaling(working_, node_options_);
        Solution final_relaxation = solve_node();
        if (final_relaxation.status == SolveStatus::kOptimal) {
          relaxation = final_relaxation;
        } else {
          working_ = std::move(pre_cut_model);
          scaling_ = std::move(pre_cut_scaling);
          relaxation = std::move(initial_relaxation);
          logger_.info("Root cuts induced failure: {}; rolled back to initial relaxation",
                       to_string(final_relaxation.status));
        }
      }
    }

    // Node bound in minimise space, excluding the offset (added back on report).
    const double node_bound = internal_objective(relaxation.col_value);

    // THE PSEUDOCOST OBSERVATION (#69): what branching on this node's column bought, per
    // unit of the fractionality it removed, in the direction it went. Recorded whether or
    // not the node is pruned next - the gain is real either way.
    if (node.has_change && node.fraction > 0.0) {
      record_pseudocost(node.change.column, node.change.is_upper,
                        std::max(node_bound - node.bound, 0.0), node.fraction);
    }

    if (can_prune(node_bound)) {
      leave();
      ++nodes_pruned_;
      continue;
    }

    try_rounding(relaxation.col_value);

    if (most_fractional(relaxation.col_value) < 0) {
      // Integral relaxation: this node's optimum is a MILP solution.
      offer_incumbent(relaxation.col_value);
      leave();
      continue;
    }

    // The children start from THIS relaxation's basis, captured before the dive and the
    // strong-branching probes can replace current_warm_ with the bases of their own solves.
    const WarmStart children_warm = basis_of(relaxation);
    current_warm_ = children_warm;

    // Diving (#25): root only. node_index == 0 identifies the root directly - it is the
    // one node present in open_ before anything else can be pushed there, so the first
    // pass through this loop body is always processing it. Every bound the dive fixes
    // lives on the same saved_ stack propagate() already pushed onto for this node, so the
    // leave() below - already here for the branching case - undoes diving's fixes too.
    if (node_index == 0) {
      dive_from_root(relaxation.col_value);
      current_warm_ = children_warm;
    }

    // The branching decision, with the node's bounds still entered: strong branching
    // solves the two children in place and restores the bounds it moved.
    const Index branch_column = reliability_branching_
                                    ? select_branching_column(relaxation.col_value, node_bound)
                                    : most_fractional(relaxation.col_value);
    if (branch_column < 0) {
      // Cannot happen after the integrality test above, but a rule that returns nothing
      // must not be answered with a branch on column -1.
      leave();
      solution.status = SolveStatus::kNumericalError;
      solution.message =
          fmt::format("the branching rule found no column at node {}", nodes_explored_);
      return solution;
    }

    const double value = relaxation.col_value[static_cast<std::size_t>(branch_column)];
    const double floor_value = std::floor(value);
    leave();

    // Two children: x <= floor(v) and x >= floor(v) + 1. Together they cover every integer
    // point, so nothing is lost.
    TreeNode down;
    down.parent = node_index;
    down.has_change = true;
    down.change = DomainChange{branch_column, true, floor_value};
    down.bound = node_bound;
    down.depth = node.depth + 1;
    down.warm = children_warm;
    down.fraction = value - floor_value;

    TreeNode up;
    up.parent = node_index;
    up.has_change = true;
    up.change = DomainChange{branch_column, false, floor_value + 1.0};
    up.bound = node_bound;
    up.depth = node.depth + 1;
    up.warm = children_warm;
    up.fraction = floor_value + 1.0 - value;

    nodes_.push_back(down);
    const auto down_index = static_cast<Index>(nodes_.size() - 1);
    nodes_.push_back(up);
    const auto up_index = static_cast<Index>(nodes_.size() - 1);
    open_.push_back(down_index);
    open_.push_back(up_index);
    dive = true;

    // ---- The node table -------------------------------------------------------------------
    best_open_bound = std::numeric_limits<double>::infinity();
    for (const Index open_index : open_) {
      best_open_bound =
          std::min(best_open_bound, nodes_[static_cast<std::size_t>(open_index)].bound);
    }
    if (!logged_table) {
      logger_.begin_node_table();
      logged_table = true;
    }
    if (nodes_explored_ % 20 == 1 || nodes_explored_ < 5) {
      const double incumbent_report =
          have_incumbent_ ? reported(incumbent_internal_) : kInfinity;
      const double gap = have_incumbent_ ? std::fabs(incumbent_internal_ - best_open_bound) /
                                               std::max(1.0, std::fabs(incumbent_internal_))
                                         : kInfinity;
      logger_.node(nodes_explored_, static_cast<Count>(open_.size()), incumbent_report,
                   reported(best_open_bound), gap, timer_.elapsed_seconds());
    }
  }

  // ---- Report ------------------------------------------------------------------------------
  double final_bound = incumbent_internal_;
  for (const Index open_index : open_) {
    final_bound = std::min(final_bound, nodes_[static_cast<std::size_t>(open_index)].bound);
  }

  if (!have_incumbent_) {
    solution.status = limit_hit ? SolveStatus::kNodeLimit : SolveStatus::kInfeasible;
    if (!limit_hit) {
      solution.message = fmt::format(
          "the search closed with no integer feasible point after {} nodes", nodes_explored_);
    }

    // NO POINT WAS FOUND, so there is no objective to report. Leaving these at their
    // defaults said objective 0, bound 0, gap 0 - and a gap of zero means CLOSED, which is
    // the exact opposite of what happened. MIPLIB found this: enlight8, enlight_hard,
    // timtab1 and neos-1425699 all came back `node_limit` with `gap 0.00e+00` beside them.
    //
    // The worst representable objective is the honest stand-in for "nothing found": no
    // feasible point means no bound on the incumbent side at all. The gaps are infinite for
    // the same reason - unknown, not closed.
    const double nothing_found =
        original_.sense == ObjSense::kMaximize ? -kInfinity : kInfinity;
    solution.objective = nothing_found;
    // The BOUND is different, and is real information worth keeping: when a limit stopped
    // the search, the open nodes still prove the optimum is no better than final_bound. Only
    // a search that closed with nothing has no bound to offer either.
    solution.dual_bound = limit_hit ? reported(final_bound) : nothing_found;
    solution.absolute_gap = kInfinity;
    solution.relative_gap = kInfinity;
    solution.nodes = nodes_explored_;
    solution.solve_seconds = timer_.elapsed_seconds();
    return solution;
  }

  solution.col_value = incumbent_x_;
  solution.nodes = nodes_explored_;
  solution.solve_seconds = timer_.elapsed_seconds();

  if (open_.empty() && !limit_hit && !gap_target_met) {
    // The tree is exhausted: the incumbent is proven optimal and is its own bound.
    solution.status = SolveStatus::kOptimal;
    solution.dual_bound = reported(incumbent_internal_);
  } else if (gap_target_met) {
    // Optimal within the gap target (#188). dual_bound is the best open bound, so
    // objective - dual_bound is the gap that was accepted, and recompute_quality() below
    // reports it; tools/verify_solution.py checks that gap against the targets recorded in
    // the .sol header rather than demanding a closed tree.
    solution.status = SolveStatus::kOptimal;
    solution.dual_bound = reported(final_bound);
  } else {
    // A node or time limit stopped the proof short of the target. The incumbent is
    // feasible, not proven, and dual_bound carries the best bound still open - claiming
    // otherwise would assert a proof that was never established.
    solution.status = SolveStatus::kFeasible;
    solution.dual_bound = reported(final_bound);
  }
  solution.recompute_quality(original_);

  logger_.info("");
  logger_.info("Status: {}   objective {:.10g}   bound {:.10g}   nodes {}   time {:.3f}s",
               to_string(solution.status), solution.objective, solution.dual_bound,
               solution.nodes, solution.solve_seconds);
  logger_.info("Nodes pruned {}, tree {} node(s) at exit", nodes_pruned_, open_.size());
  if (!quadratic_) {
    // How the node LPs were solved (#65). The ratio of warm to cold is the whole point of
    // the dual node engine, and the iterations per solve are the evidence it pays.
    logger_.info(
        "Node LPs: {} warm-started dual ({} iterations), {} cold primal ({} iterations), {} "
        "cold fallback(s) after a dual failure; strong branching {} probe(s), {} iterations",
        warm_node_solves_, warm_node_iterations_, cold_node_solves_, cold_node_iterations_,
        cold_fallbacks_, strong_branch_solves_, strong_branch_iterations_);
  }
  if (!solution.message.empty()) logger_.info("{}", solution.message);
  return solution;
}

}  // namespace

Solution solve_branch_and_bound(const Model& model, const Options& options, Logger& logger) {
  // ROOT CUTS, applied once before the search rather than per node.
  //
  // Integer rounding tightens a row IN PLACE, so unlike a generated cut it adds no row, grows
  // no basis, and costs the search nothing per node - every node LP simply starts from a
  // tighter relaxation. Applying it at the root is therefore all that is needed; re-applying
  // it deeper would find nothing new, because branching changes column bounds and not the row
  // coefficients this reads.
  //
  // It runs on a COPY. The caller's model is an input, and a solver that silently rewrites
  // the model it was handed makes a second solve of the "same" model mean something different
  // from the first.
  Model tightened = model;
  const RowTightening effect = tighten_integral_rows(&tightened, logger);

  BranchAndBound search(effect.rows_tightened > 0 ? tightened : model, options, logger);
  return search.run();
}

}  // namespace sankhya::mip

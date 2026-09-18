// SPDX-License-Identifier: Apache-2.0
// SANKHYA - internal state shared by branch_and_bound.cpp and branch_and_bound_node.cpp
// (issue #262 split this out of one 1,464-line file so neither half stays over the
// ~600-line rule; see ENGINEERING_RULES.md). Not a public header - nothing outside
// src/mip/ includes this.
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
#pragma once

#include "sankhya/mip.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/solve_control.hpp"

#include "checkpoint.hpp"
#include "cuts.hpp"
#include "heuristics.hpp"
#include "solution_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../core/resource_limits.hpp"
#include "../core/stop_controller.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "simplex/primal_simplex.hpp"

namespace sankhya::mip {

/// One tightened bound, recorded so entering a node can be undone rather than rebuilt.
struct DomainChange {
  Index column = -1;
  bool is_upper = false;
  double value = 0.0;
};

/// A node holds only its OWN bound change and a link to its parent. The full domain is
/// recovered by walking to the root, which is why the tree costs O(depth) per node instead
/// of O(columns).
/// Which open node the search takes next (#293).
///
/// Node selection changes the ORDER the tree is explored in and nothing else: the same nodes
/// exist, the same bounds prune them, and the optimum is the optimum under every policy. What
/// it does change is when the first incumbent arrives, how fast the global bound moves, and
/// how many nodes stay open at once - which is why one fixed policy suits no model.
///
/// Reference: Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 6, for the
/// best-estimate rule and the dive-then-best-bound hybrid.
enum class NodeSelection {
  /// Dive to a leaf, then best-bound. An early incumbent is what makes every later bound able
  /// to prune, and best-bound afterwards keeps the tree from growing where it cannot pay.
  kHybrid,
  /// Always the smallest bound. Proves optimality in the fewest nodes and keeps the widest
  /// tree open, so the memory is the price.
  kBestBound,
  /// Always the deepest node. Narrow tree, fast to a first incumbent, and the global bound
  /// barely moves until the search backtracks.
  kDepthFirst,
  /// Smallest pseudocost estimate of where the node's subtree will end up, which is a guess
  /// at where a GOOD incumbent is rather than at where the bound is.
  kBestEstimate,
};

[[nodiscard]] const char* to_string(NodeSelection selection) noexcept;

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
  /// Where the pseudocosts expect this subtree's integer answer to land (#293), computed from
  /// the parent's relaxation when the node is created. Only kBestEstimate reads it; it equals
  /// `bound` until the pseudocosts have seen anything.
  double estimate = 0.0;
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
inline double fractionality(double value) {
  return std::fabs(value - std::round(value));
}

class BranchAndBound {
 public:
  BranchAndBound(const Model& model, const Options& options, Logger& logger,
                 SolveControl* control)
      : original_(model),
        working_(model),
        options_(options),
        logger_(logger),
        control_(control) {
    integrality_tolerance_ = options.get_double("integrality_tolerance");
    relative_gap_target_ = options.get_double("mip_relative_gap");
    absolute_gap_target_ = options.get_double("mip_absolute_gap");
    // One interpretation of every limit, shared with every other engine (#289).
    limits_ = ResourceLimits(options, logger);
    time_limit_ = options.get_double("time_limit");
    sense_ = model.sense_multiplier();

    node_engine_dual_ = options.get_string("mip_node_engine") != "primal";
    const std::string selection = options.get_string("mip_node_selection");
    if (selection == "best-bound") {
      node_selection_ = NodeSelection::kBestBound;
    } else if (selection == "depth-first") {
      node_selection_ = NodeSelection::kDepthFirst;
    } else if (selection == "best-estimate") {
      node_selection_ = NodeSelection::kBestEstimate;
    } else {
      node_selection_ = NodeSelection::kHybrid;
    }
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

    // The solution pool (#225). Built after integer_columns_, because an assignment is what
    // makes two points the same plan.
    pool_ =
        SolutionPool(integer_columns_, static_cast<std::size_t>(options.get_int("pool_size")),
                     options.get_bool("pool_diversity"));
    pool_gap_ = options.get_double("pool_gap");
    // A complete search with nowhere to keep what it finds would enumerate for nothing.
    pool_complete_ = options.get_bool("pool_complete") && pool_.enabled();
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

  /// Take the next open node under the configured policy (#293), removing it from `open_`.
  /// `diving` is the hybrid's signal that the previous node just produced children.
  [[nodiscard]] Index take_next_open_node(bool diving);

  /// Where the pseudocosts expect a node branched from this relaxation to end up: the node's
  /// own bound plus, for every column still fractional, the cheaper of the two directions
  /// (Achterberg 2007, sec. 6.1). Returns `bound` unchanged when nothing is fractional.
  [[nodiscard]] double estimate_from(const std::vector<double>& x, double bound) const;

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

  // ---- Primal heuristics (#290), in branch_and_bound_heuristics.cpp ---------------------
  void init_heuristics();
  /// Offer a heuristic's candidate and count it against that heuristic.
  bool offer_from(std::size_t slot, const std::vector<double>& x);
  /// Rounding (every node), lock rounding (every node), repair (root) and RINS (scheduled).
  void run_node_heuristics(Index node_index, const Solution& relaxation);
  /// The root dive, counted.
  void run_root_dive(const std::vector<double>& x);
  /// The feasibility pump at the root, only when nothing else found an incumbent.
  void run_root_pump(const Solution& relaxation);
  void report_heuristics();
  // ---- Checkpoint and resume (#287), in branch_and_bound_checkpoint.cpp -----------------
  /// The search as it stands between nodes.
  [[nodiscard]] TreeCheckpoint make_checkpoint() const;
  /// Write it to checkpoint_path_ when one is set; a failed write is a warning.
  void save_checkpoint();
  /// Load, validate and rebuild the open nodes and the incumbent. Empty on success, the
  /// reason it was refused otherwise.
  [[nodiscard]] std::string restore_checkpoint(const std::string& path);

  /// pool_complete (#225): an integral relaxation closes a node for the OPTIMUM, not for the
  /// pool - the node's region can still hold the second-best assignment. Partition the rest
  /// of the region around the relaxation's point, every unfixed integer column at once (see
  /// the definition). Returns false, with the node's bounds still entered, when every integer
  /// column is fixed and the node really is a single assignment.
  bool split_integral_node(Index node_index, const Solution& relaxation);

  /// Is a bound worth exploring given the incumbent?
  /// Solve the current node's relaxation with whichever engine the model calls for.
  ///
  /// Both engines take the same Model and return the same Solution, which is what makes this
  /// a one-line choice rather than a second search. The QP path carries no basis, so nothing
  /// downstream may assume one - the diving heuristic and the branching rule both read
  /// col_value only, which they already did.
  [[nodiscard]] Solution solve_node() { return solve_node_with(node_options_); }

  [[nodiscard]] Solution solve_node_with(const Options& options) {
    if (quadratic_) return qp::solve_convex_qp(working_, options, logger_, control_);
    // WARM-STARTED DUAL SIMPLEX BELOW THE ROOT (#65). The basis in current_warm_ was
    // optimal for a problem that differs from this one by a bound or two, so it is dual
    // feasible here, which is exactly the state the dual simplex starts from. Measured
    // before this: every node was a cold primal solve from the slack basis.
    //
    // The primal stays as the fallback, cold, for a node the dual could not finish: a
    // numerical answer at a node cannot be fathomed honestly, and the search below stops
    // on it, so it is worth one more solve to avoid.
    if (node_engine_dual_ && !current_warm_.empty()) {
      Solution warm =
          solve_dual_simplex(working_, options, logger_, scaling_, control_, &current_warm_);
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
    Solution cold = solve_primal_simplex(working_, options, logger_, scaling_, control_);
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
    // Minimise space throughout: a node whose bound is no better than the incumbent, to
    // within the absolute gap target, cannot contain an improving solution.
    // The margin is widened for a QP node by the tolerance its bound is only accurate to.
    // Pruning too little costs nodes; pruning too much loses the optimum silently.
    const double margin =
        quadratic_ ? std::max(absolute_gap_target_, kMiqpNodeTolerance) : absolute_gap_target_;
    if (pool_complete_) return bound >= pool_cutoff() - margin;
    if (!have_incumbent_) return false;
    return bound >= incumbent_internal_ - margin;
  }

  /// What a node must beat to matter when the search is filling the pool (#225): the worst
  /// member of a full pool, and never worse than pool_gap past the incumbent. +infinity while
  /// neither applies, which means nothing is pruned - the price of a complete pool. The
  /// incumbent-relative limit only tightens as the incumbent improves, so a node pruned on it
  /// stays prunable.
  [[nodiscard]] double pool_cutoff() const {
    double cutoff = pool_.cutoff();
    if (have_incumbent_ && pool_gap_ < kNoPoolGap) {
      cutoff = std::min(cutoff, incumbent_internal_ +
                                    pool_gap_ * std::max(1.0, std::fabs(incumbent_internal_)));
    }
    return cutoff;
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

  // ---- Cut rounds (branch_and_bound_cuts.cpp, #221) ------------------------------------

  /// The root round: cover, Gomory and MIR candidates, filtered, appended, the root
  /// re-solved; rolled back if the re-solve fails. Replaces `relaxation` on success.
  void root_cut_round(Solution* relaxation);
  /// A round at a node of depth <= tree_cut_depth_: MIR cuts on the global bounds,
  /// appended for the whole tree, the node re-solved from its own basis.
  void tree_cut_round(Index depth, Solution* relaxation);
  /// Append `accepted` as rows of working_ and register them in the pool.
  void append_cut_rows(const std::vector<Cut>& accepted);
  /// Bring every stored basis to `rows` row statuses: a new row's logical is basic.
  void resize_warm_starts(Index rows);
  [[nodiscard]] bool is_pooled_duplicate(const Cut& cut) const;
  /// Clique and {0,1/2} candidates (#358) at `relaxation`, appended to `candidates`, each
  /// family behind its own option.
  void add_combinatorial_cuts(const Solution& relaxation, std::vector<Cut>* candidates);
  /// Count node solves in which each cut row was slack; free a row slack for too long.
  void age_cut_rows(const Solution& relaxation);

  /// The root bounds and the cut counts onto the answer (#221).
  void report_root(Solution* solution) const {
    solution->cuts_applied = root_cuts_applied_ + tree_cuts_applied_;
    if (std::isnan(root_bound_internal_)) return;
    solution->root_bound = reported(root_bound_internal_);
    solution->root_bound_after_cuts = reported(root_bound_after_cuts_internal_);
  }

  const Model& original_;
  Model working_;
  const Options& options_;
  Logger& logger_;
  SolveControl* control_;
  Options node_options_;

  double integrality_tolerance_ = tol::kIntegrality;
  double relative_gap_target_ = tol::kMipRelativeGap;
  double absolute_gap_target_ = tol::kMipAbsoluteGap;
  ResourceLimits limits_;
  double time_limit_ = 0.0;
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
  /// mip_node_selection (#293). Order only: every policy explores the same tree.
  NodeSelection node_selection_ = NodeSelection::kHybrid;
  /// How many nodes each rule chose, for the report at the end of the search.
  Count selected_by_dive_ = 0;
  Count selected_by_policy_ = 0;
  Index deepest_node_ = 0;
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

  // Primal heuristics (#290).
  std::vector<HeuristicStats> heuristic_stats_;
  Locks locks_;
  bool heuristics_on_ = true;
  Count rins_frequency_ = 0;
  Count rins_nodes_ = 0;
  int pump_rounds_ = 0;
  Count clique_cuts_generated_ = 0;     ///< #358, before the filter
  Count zero_half_cuts_generated_ = 0;  ///< #358, before the filter
  std::string checkpoint_path_;
  Count checkpoint_nodes_ = 0;
  Count last_checkpoint_at_ = -1;
  Count checkpoints_written_ = 0;

  bool have_incumbent_ = false;
  double incumbent_internal_ = std::numeric_limits<double>::infinity();
  std::vector<double> incumbent_x_;

  /// Every integer-feasible point offer_incumbent() found feasible, not only the improving
  /// ones (#225).
  SolutionPool pool_;
  double pool_gap_ = std::numeric_limits<double>::max();
  bool pool_complete_ = false;
  /// pool_gap at or above this is the option's keep-everything default.
  static constexpr double kNoPoolGap = 1e300;

  Count nodes_explored_ = 0;
  /// Root relaxation objective (internal, minimise space) before and after the root cut
  /// round; NaN until the root LP solved. Reported through Solution::root_bound (#221).
  double root_bound_internal_ = std::numeric_limits<double>::quiet_NaN();
  double root_bound_after_cuts_internal_ = std::numeric_limits<double>::quiet_NaN();
  Count root_cuts_applied_ = 0;
  /// Cut rows below the root (#221): the option-driven depth cap and per-round row cap,
  /// the column bounds every tree cut is built on (valid everywhere), the pool of rows
  /// appended so far starting at working_ row first_cut_row_, and their ageing state.
  Index tree_cut_depth_ = 0;
  Index tree_cut_rows_per_round_ = 20;
  std::vector<double> global_lower_;
  std::vector<double> global_upper_;
  Index first_cut_row_ = -1;
  std::vector<Cut> pool_cuts_;
  std::vector<Count> cut_row_slack_;
  std::vector<bool> cut_row_free_;
  Count tree_cuts_applied_ = 0;
  Count tree_cut_rounds_ = 0;
  Count cut_rows_aged_out_ = 0;
  /// Node solves a cut row may sit slack before it is freed (Achterberg 2007, sec. 8.10
  /// uses a comparable age).
  static constexpr Count kCutRowAgeLimit = 50;
  Count nodes_pruned_ = 0;
  Timer timer_;
};

}  // namespace sankhya::mip

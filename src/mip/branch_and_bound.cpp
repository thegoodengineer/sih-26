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
// WHAT THIS PHASE DELIBERATELY DOES NOT DO. No cutting planes, no pseudocost or reliability
// branching, no node presolve beyond simple propagation, no parallelism. Those are Phase 7,
// and they are also where MILP correctness bugs hide: a cut that is very slightly invalid
// removes the optimum and the search then proves the wrong answer, confidently. Getting a
// plain, correct search working first is what makes those additions checkable.

#include "sankhya/mip.hpp"

#include <algorithm>
#include <cmath>
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
};

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

    // Node LPs are solved silently; the node table is the log the user wants, not several
    // hundred simplex iteration tables.
    node_options_ = options;
    node_options_.set_bool("log_to_console", false);

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

  /// Round the relaxation to the nearest integers and test the result. Cheap, and on models
  /// with a lot of structure it finds the incumbent that makes every later bound useful.
  void try_rounding(const std::vector<double>& x);

  /// Accept a candidate if it is integral, feasible and better than the incumbent.
  bool offer_incumbent(const std::vector<double>& x);

  /// Is a bound worth exploring given the incumbent?
  [[nodiscard]] bool can_prune(double bound) const {
    if (!have_incumbent_) return false;
    // Minimise space throughout: a node whose bound is no better than the incumbent, to
    // within the absolute gap target, cannot contain an improving solution.
    return bound >= incumbent_internal_ - absolute_gap_target_;
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

  double objective = 0.0;
  for (Index j = 0; j < original_.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    objective += sense_ * original_.col_cost[u] * x[u];
  }
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

  logger_.info("Branch and bound: {} rows, {} columns, {} integer columns",
               original_.num_rows(), original_.num_cols(), integer_columns_.size());

  nodes_.push_back(TreeNode{});
  open_.push_back(0);

  double best_open_bound = -std::numeric_limits<double>::infinity();
  bool logged_table = false;
  bool dive = false;
  bool limit_hit = false;

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
      const double relative = gap / std::max(1.0, std::fabs(incumbent_internal_));
      if (gap <= absolute_gap_target_ || relative <= relative_gap_target_) {
        limit_hit = true;
        solution.message = fmt::format(
            "stopped on a {} gap target ({:.3e} absolute, {:.3e} relative) after {} nodes",
            relative <= relative_gap_target_ ? "relative" : "absolute", gap, relative,
            nodes_explored_);
        break;
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

    if (!propagate()) {
      leave();
      ++nodes_pruned_;
      continue;
    }

    const Solution relaxation = solve_primal_simplex(working_, node_options_, logger_);

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

    // Node bound in minimise space, excluding the offset (added back on report).
    double node_bound = 0.0;
    for (Index j = 0; j < original_.num_cols(); ++j) {
      const auto u = static_cast<std::size_t>(j);
      node_bound += sense_ * original_.col_cost[u] * relaxation.col_value[u];
    }

    if (can_prune(node_bound)) {
      leave();
      ++nodes_pruned_;
      continue;
    }

    try_rounding(relaxation.col_value);

    const Index branch_column = most_fractional(relaxation.col_value);
    if (branch_column < 0) {
      // Integral relaxation: this node's optimum is a MILP solution.
      offer_incumbent(relaxation.col_value);
      leave();
      continue;
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

    TreeNode up;
    up.parent = node_index;
    up.has_change = true;
    up.change = DomainChange{branch_column, false, floor_value + 1.0};
    up.bound = node_bound;
    up.depth = node.depth + 1;

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
    solution.nodes = nodes_explored_;
    solution.solve_seconds = timer_.elapsed_seconds();
    return solution;
  }

  solution.col_value = incumbent_x_;
  solution.nodes = nodes_explored_;
  solution.solve_seconds = timer_.elapsed_seconds();

  if (open_.empty() && !limit_hit) {
    // The tree is exhausted: the incumbent is proven optimal and is its own bound.
    solution.status = SolveStatus::kOptimal;
    solution.dual_bound = reported(incumbent_internal_);
  } else {
    // A limit stopped the proof. The incumbent is feasible, not proven, and dual_bound
    // carries the best bound still open - claiming otherwise would assert a proof that was
    // never established.
    solution.status = SolveStatus::kFeasible;
    solution.dual_bound = reported(final_bound);
  }
  solution.recompute_quality(original_);

  logger_.info("");
  logger_.info("Status: {}   objective {:.10g}   bound {:.10g}   nodes {}   time {:.3f}s",
               to_string(solution.status), solution.objective, solution.dual_bound,
               solution.nodes, solution.solve_seconds);
  logger_.info("Nodes pruned {}, tree {} node(s) at exit", nodes_pruned_, open_.size());
  if (!solution.message.empty()) logger_.info("{}", solution.message);
  return solution;
}

}  // namespace

Solution solve_branch_and_bound(const Model& model, const Options& options, Logger& logger) {
  BranchAndBound search(model, options, logger);
  return search.run();
}

}  // namespace sankhya::mip

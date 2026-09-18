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
#include "sankhya/solve_control.hpp"

#include "cuts.hpp"
#include "mir_cuts.hpp"
#include "solution_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../core/stop_controller.hpp"
#include "../util/profiler.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "simplex/primal_simplex.hpp"

#include "branch_and_bound_internal.hpp"

namespace sankhya::mip {

Solution BranchAndBound::run() {
  init_heuristics();
  global_lower_ = working_.col_lower;
  global_upper_ = working_.col_upper;
  if (options_.get_bool("enable_root_cuts")) {
    tree_cut_depth_ = static_cast<Index>(options_.get_int("tree_cut_depth"));
    tree_cut_rows_per_round_ = static_cast<Index>(options_.get_int("tree_cut_rows_per_round"));
  }
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

  // THE ROOT STARTS WITH NO BOUND PROVED (#289). TreeNode::bound is 0.0 by default, which
  // is a placeholder for "inherited from the parent" and the root has no parent. A search
  // stopped before it evaluated the root then reported a dual bound of 0: true by luck on a
  // model whose optimum is positive, and a false claim on one whose optimum is negative.
  // Minus infinity in minimise space is what "nothing is proved yet" actually is, and it
  // prunes nothing, which is also what an unevaluated root should do.
  TreeNode root;
  root.bound = -std::numeric_limits<double>::infinity();
  nodes_.push_back(root);
  open_.push_back(0);

  // RESUME (#287): replace the fresh root with the open nodes of a saved search. Everything
  // is validated before a single node is touched, and a checkpoint that does not belong to
  // this model, or this build's format, is refused - never loaded on a best-effort basis.
  checkpoint_path_ = options_.get_string("checkpoint");
  checkpoint_nodes_ = options_.get_int("checkpoint_nodes");
  if (const std::string resume = options_.get_string("resume"); !resume.empty()) {
    if (const std::string refused = restore_checkpoint(resume); !refused.empty()) {
      solution.status = SolveStatus::kNotSolved;
      solution.message = "resume refused: " + refused;
      logger_.warning("{}", solution.message);
      return solution;
    }
  }

  double best_open_bound = -std::numeric_limits<double>::infinity();
  bool logged_table = false;
  bool dive = false;
  bool limit_hit = false;
  bool gap_target_met = false;
  double open_bound = -std::numeric_limits<double>::infinity();

  StopController stop(control_, timer_, limits_);
  SolveStatus stop_status;
  Solution best_available_point;

  while (!open_.empty()) {
    // PERIODIC CHECKPOINT (#287), between nodes: no node is entered, so every bound in the
    // working model is the root's and the open list is the whole of the search.
    if (checkpoint_nodes_ > 0 && nodes_explored_ > 0 &&
        nodes_explored_ % checkpoint_nodes_ == 0 && nodes_explored_ != last_checkpoint_at_) {
      save_checkpoint();
    }
    // Both counters and the clock are checked here, in the documented order: a tree that is
    // out of time and out of nodes at the same node reports the time limit (#289).
    if (const LimitReason why = limits_.exhausted(timer_.elapsed_seconds(), 0,
                                                  static_cast<std::int64_t>(nodes_explored_));
        why != LimitReason::kNone) {
      limit_hit = true;
      solution.status = status_for(why);
      // The status alone cannot say this once an incumbent turns it into kFeasible (#289).
      solution.stopped_by = why;
      solution.message = limits_.describe(why, timer_.elapsed_seconds(), 0,
                                          static_cast<std::int64_t>(nodes_explored_));
      break;
    }

    // ALGORITHMIC OPEN BOUND: compute unconditionally when there is an incumbent so that
    // the gap-target stopping condition below always sees a fresh value every iteration.
    // The progress callback lambda reuses this when reporting and does NOT re-scan.
    if (have_incumbent_) {
      open_bound = std::numeric_limits<double>::infinity();
      for (const Index open_index : open_) {
        open_bound = std::min(open_bound, nodes_[static_cast<std::size_t>(open_index)].bound);
      }
    }

    if (stop.should_stop(
            [&]() {
              Progress p;
              p.phase = Progress::Phase::kTree;
              p.iterations = 0;  // Not tracking simplex iterations across the tree currently
              p.nodes = nodes_explored_;
              p.open_nodes = static_cast<std::int64_t>(open_.size());
              p.objective = have_incumbent_ ? reported(incumbent_internal_)
                                            : std::numeric_limits<double>::infinity();
              // When an incumbent exists, open_bound was computed above this call; reuse it.
              // When no incumbent exists yet, scan now for accurate reporting only.
              double reporting_bound = open_bound;
              if (!have_incumbent_) {
                reporting_bound = std::numeric_limits<double>::infinity();
                for (const Index open_index : open_) {
                  reporting_bound = std::min(
                      reporting_bound, nodes_[static_cast<std::size_t>(open_index)].bound);
                }
              }
              p.best_bound =
                  (original_.sense == ObjSense::kMaximize) ? -reporting_bound : reporting_bound;
              p.gap = have_incumbent_ ? (incumbent_internal_ - open_bound)
                                      : std::numeric_limits<double>::infinity();
              return p;
            },
            &stop_status)) {
      limit_hit = true;
      solution.status = stop_status;
      solution.stopped_by =
          stop_status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt;
      solution.message = limits_.describe(
          stop_status == SolveStatus::kTimeLimit ? LimitReason::kTime : LimitReason::kInterrupt,
          timer_.elapsed_seconds(), 0, static_cast<std::int64_t>(nodes_explored_));
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
    // Not while filling the pool: meeting the gap target proves the incumbent, not that the
    // pool holds the best alternatives, and pool_complete promises the second.
    if (have_incumbent_ && !pool_complete_) {
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

    // Which node to take next is the configured policy's decision (#293), and only the
    // order it decides: the tree, the bounds and the incumbent test are the same either way.
    const Index node_index = take_next_open_node(dive);
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

    Solution relaxation = [&] {
      ProfileScope timed(logger_.profiler(), "node LP", ProfileMode::kDetailed);
      return solve_node();
    }();

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
    if (relaxation.status == SolveStatus::kInterrupted ||
        relaxation.status == SolveStatus::kTimeLimit) {
      leave();
      limit_hit = true;
      solution.status = relaxation.status;
      solution.stopped_by = relaxation.status == SolveStatus::kTimeLimit
                                ? LimitReason::kTime
                                : LimitReason::kInterrupt;
      best_available_point = std::move(relaxation);
      break;
    }
    if (relaxation.status != SolveStatus::kOptimal) {
      // A node whose LP did not solve cannot be fathomed honestly: pruning it could discard
      // the optimum. Stop and report rather than quietly continuing on a broken bound.
      leave();
      // WHY IT COULD NOT BE SOLVED DECIDES WHAT THIS IS (#289). A node LP that ran out of
      // ITERATIONS is a resource limit that reached the tree through the node engine, not a
      // numerical failure: `--option iteration_limit=5` on a MILP used to come back
      // numerical_error, which says the solver broke when what happened is that it was told
      // to stop. Anything else - a singular basis, an unbounded node, a status no node
      // should return - stays the numerical failure it is.
      const bool out_of_iterations = relaxation.status == SolveStatus::kIterationLimit;
      limit_hit = out_of_iterations;
      solution.status =
          out_of_iterations ? SolveStatus::kIterationLimit : SolveStatus::kNumericalError;
      if (out_of_iterations) solution.stopped_by = LimitReason::kIterations;
      solution.message =
          out_of_iterations
              ? fmt::format(
                    "the node LP at node {} stopped at the iteration limit, so the tree "
                    "cannot go on: a node whose bound is unknown cannot be fathomed without "
                    "risking the optimum ({})",
                    nodes_explored_, relaxation.message)
              : fmt::format("node LP returned {} at node {}", to_string(relaxation.status),
                            nodes_explored_);
      if (!out_of_iterations) return solution;
      break;
    }

    best_available_point = relaxation;

    if (node_index == 0) {
      root_bound_internal_ = internal_objective(relaxation.col_value);
      root_bound_after_cuts_internal_ = root_bound_internal_;
    }
    if (node_index == 0 && options_.get_bool("enable_root_cuts")) root_cut_round(&relaxation);
    // Cuts below the root (#221): shallow nodes only, on the global bounds, kept for the
    // whole tree. The node's bound is taken after the round, so a cut that moved it
    // counts for pruning and for the pseudocosts alike.
    if (node_index != 0 && node.depth <= tree_cut_depth_ && !quadratic_) {
      tree_cut_round(node.depth, &relaxation);
    }
    age_cut_rows(relaxation);

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

    // Primal heuristics (#290): rounding every node as before, and the ones #290 added,
    // on their own schedules and budgets. They propose; offer_incumbent() decides.
    {
      ProfileScope timed(logger_.profiler(), "heuristics", ProfileMode::kDetailed);
      run_node_heuristics(node_index, relaxation);
    }

    if (most_fractional(relaxation.col_value) < 0) {
      // Integral relaxation: this node's optimum is a MILP solution.
      offer_incumbent(relaxation.col_value);
      if (pool_complete_ && split_integral_node(node_index, relaxation)) continue;
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
      ProfileScope timed(logger_.profiler(), "heuristics", ProfileMode::kDetailed);
      run_root_dive(relaxation.col_value);
      current_warm_ = children_warm;
      // The feasibility pump only when rounding, repair and the dive all came back empty:
      // its value is an incumbent where there is none, and it costs LP solves.
      run_root_pump(relaxation);
    }

    // The branching decision, with the node's bounds still entered: strong branching
    // solves the two children in place and restores the bounds it moved.
    const Index branch_column = [&] {
      // Strong branching's probe LPs are inside this, which is the point: it is the cost of
      // the branching decision.
      ProfileScope timed(logger_.profiler(), "branching", ProfileMode::kDetailed);
      return reliability_branching_ ? select_branching_column(relaxation.col_value, node_bound)
                                    : most_fractional(relaxation.col_value);
    }();
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
    // Both children inherit the parent's estimate: it is a property of the relaxation they
    // were branched from, and the branched column's own contribution is the one term the
    // branch is about to settle.
    const double child_estimate = estimate_from(relaxation.col_value, node_bound);

    TreeNode down;
    down.parent = node_index;
    down.has_change = true;
    down.change = DomainChange{branch_column, true, floor_value};
    down.bound = node_bound;
    down.depth = node.depth + 1;
    down.warm = children_warm;
    down.fraction = value - floor_value;
    down.estimate = child_estimate;

    TreeNode up;
    up.parent = node_index;
    up.has_change = true;
    up.change = DomainChange{branch_column, false, floor_value + 1.0};
    up.bound = node_bound;
    up.depth = node.depth + 1;
    up.warm = children_warm;
    up.fraction = floor_value + 1.0 - value;
    up.estimate = child_estimate;

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

  // A search stopped by a limit is exactly the one worth resuming (#287).
  if (limit_hit && !open_.empty()) save_checkpoint();

  // ---- Report ------------------------------------------------------------------------------
  double final_bound = incumbent_internal_;
  for (const Index open_index : open_) {
    final_bound = std::min(final_bound, nodes_[static_cast<std::size_t>(open_index)].bound);
  }

  if (!have_incumbent_) {
    if (!limit_hit) {
      solution.status = SolveStatus::kInfeasible;
      solution.message = fmt::format(
          "the search closed with no integer feasible point after {} nodes", nodes_explored_);
    }

    const double nothing_found =
        original_.sense == ObjSense::kMaximize ? -kInfinity : kInfinity;

    if (limit_hit && claims_a_point(solution.status) &&
        !best_available_point.col_value.empty()) {
      solution.col_value = std::move(best_available_point.col_value);
      solution.row_activity = std::move(best_available_point.row_activity);
      solution.row_dual = std::move(best_available_point.row_dual);
      solution.col_dual = std::move(best_available_point.col_dual);
      solution.objective = best_available_point.objective;
      solution.primal_infeasibility = best_available_point.primal_infeasibility;
      solution.primal_infeasibility_scaled = best_available_point.primal_infeasibility_scaled;
      solution.dual_infeasibility = best_available_point.dual_infeasibility;
      solution.dual_infeasibility_scaled = best_available_point.dual_infeasibility_scaled;
      solution.integrality_violation = best_available_point.integrality_violation;
      solution.iterations = warm_node_iterations_ + cold_node_iterations_;
      // A LIMIT WITHOUT AN INCUMBENT STILL SHOWS A POINT, and says what it is. `objective`
      // on a MILP has always meant the incumbent's value; a reader who finds a value here
      // must not take a fractional relaxation for an integer solution.
      solution.message += fmt::format(
          "; no integer feasible point was found, so the point reported is the last LP "
          "relaxation, fractional by {:.3e}",
          solution.integrality_violation);
    } else {
      // NO POINT WAS FOUND, so there is no objective to report. Leaving these at their
      // defaults said objective 0, bound 0, gap 0 - and a gap of zero means CLOSED, which is
      // the exact opposite of what happened. MIPLIB found this: enlight8, enlight_hard,
      // timtab1 and neos-1425699 all came back `node_limit` with `gap 0.00e+00` beside them.
      //
      // The worst representable objective is the honest stand-in for "nothing found": no
      // feasible point means no bound on the incumbent side at all. The gaps are infinite for
      // the same reason - unknown, not closed.
      solution.objective = nothing_found;
    }
    // The BOUND is different, and is real information worth keeping: when a limit stopped
    // the search, the open nodes still prove the optimum is no better than final_bound. Only
    // a search that closed with nothing has no bound to offer either.
    solution.dual_bound = limit_hit ? reported(final_bound) : nothing_found;
    solution.absolute_gap = kInfinity;
    solution.relative_gap = kInfinity;
    solution.nodes = nodes_explored_;
    solution.solve_seconds = timer_.elapsed_seconds();
    report_root(&solution);
    return solution;
  }

  solution.col_value = incumbent_x_;
  solution.nodes = nodes_explored_;
  solution.solve_seconds = timer_.elapsed_seconds();
  report_root(&solution);

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
  if (tree_cut_rounds_ > 0) {
    logger_.info("Tree cuts: {} rounds below the root, {} rows added, {} aged out",
                 tree_cut_rounds_, tree_cuts_applied_, cut_rows_aged_out_);
  }

  if (pool_.enabled()) {
    for (SolutionPool::Entry& entry :
         pool_.finish(incumbent_internal_, incumbent_x_, pool_gap_)) {
      Solution::PoolEntry member;
      member.objective = original_.evaluate_objective(entry.x.data());
      member.col_value = std::move(entry.x);
      solution.pool.push_back(std::move(member));
    }
    // pool[0] IS the solution: the same vector, and the objective as recompute_quality wrote
    // it, so a reader comparing the two never meets a last-bit difference.
    solution.pool.front().objective = solution.objective;
    logger_.info("Solution pool: {} plan(s), objectives {:.10g} to {:.10g}{}",
                 solution.pool.size(), solution.pool.front().objective,
                 solution.pool.back().objective, pool_complete_ ? " (complete)" : "");
  }

  logger_.info("");
  logger_.info("Status: {}   objective {:.10g}   bound {:.10g}   nodes {}   time {:.3f}s",
               to_string(solution.status), solution.objective, solution.dual_bound,
               solution.nodes, solution.solve_seconds);
  logger_.info("Nodes pruned {}, tree {} node(s) at exit", nodes_pruned_, open_.size());
  report_heuristics();
  if (clique_cuts_generated_ + zero_half_cuts_generated_ > 0) {
    logger_.info(
        "Combinatorial cut candidates (#358): {} clique, {} zero-half, before the filter",
        clique_cuts_generated_, zero_half_cuts_generated_);
  }
  if (Profiler* profiler = logger_.profiler(); profiler != nullptr) {
    profiler->count("nodes pruned", static_cast<std::int64_t>(nodes_pruned_));
    profiler->count("warm-started node LPs", static_cast<std::int64_t>(warm_node_solves_));
    profiler->count("cold node LPs", static_cast<std::int64_t>(cold_node_solves_));
  }
  logger_.info(
      "Node selection {}: {} node(s) taken deepest-first, {} by the policy, deepest node at "
      "depth {}",
      to_string(node_selection_), selected_by_dive_, selected_by_policy_, deepest_node_);
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

Solution solve_branch_and_bound(const Model& model, const Options& options, Logger& logger,
                                SolveControl* control) {
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

  BranchAndBound search(effect.rows_tightened > 0 ? tightened : model, options, logger,
                        control);
  return search.run();
}

}  // namespace sankhya::mip

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: node domain management, propagation, the branching rule
// (reliability / pseudocosts / strong branching), and the rounding and diving heuristics.
// Split out of branch_and_bound.cpp by issue #262, which keeps the tree driver run() and
// the ~600-line rule off both halves; see branch_and_bound_internal.hpp for the shared
// class declaration and ENGINEERING_RULES.md for the rule.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "../core/stop_controller.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "simplex/primal_simplex.hpp"

namespace sankhya::mip {

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
  conflict_pruned_ = false;
  const Index rows = working_.num_rows();
  const CsrView by_row(working_.matrix);

  // EVERY INTEGER COLUMN IS ROUNDED, AND EVERY BOX IS CHECKED, BEFORE ANY ROW IS READ (#328).
  // The row sweeps below round an integer column only while walking a row it appears in, and
  // check a box for collapse in the same place. A column that appears in no row - an
  // objective-only integer column, legal MPS - was therefore never rounded and never checked:
  // x1 in [0.5, 2.5] stayed fractional at the root, the down child x1 <= floor(0.5) = 0 left
  // it at [0.5, 0], and that crossed box went to the node LP, came back at x1 = 0.5 and was
  // branched on again, forever. The search reported the root bound until its node limit. Both
  // steps are cheap and neither depends on the rows, so they happen once per node, here.
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    if (is_finite_bound(working_.col_lower[u])) {
      const double rounded = std::ceil(working_.col_lower[u] - integrality_tolerance_);
      if (rounded != working_.col_lower[u]) tighten_lower(u, rounded);
    }
    if (is_finite_bound(working_.col_upper[u])) {
      const double rounded = std::floor(working_.col_upper[u] + integrality_tolerance_);
      if (rounded != working_.col_upper[u]) tighten_upper(u, rounded);
    }
  }
  for (Index j = 0; j < working_.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (working_.col_lower[u] > working_.col_upper[u] + tol::kPrimalFeasibility) {
      return false;  // the box is empty, so the node is: no LP is asked to solve it
    }
  }

  // A handful of sweeps. Propagation to a fixed point can be slow and rarely pays for
  // itself at a node; Savelsbergh's observation is that most of the tightening happens in
  // the first pass or two.
  for (int sweep = 0; sweep < 3; ++sweep) {
    bool changed = false;
    // Learned conflicts (#292) first: they are cheap, and a bound one implies feeds the rows.
    if (!conflicts_.entries().empty() && !propagate_conflicts(&changed)) return false;
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

const char* to_string(NodeSelection selection) noexcept {
  switch (selection) {
    case NodeSelection::kHybrid: return "hybrid";
    case NodeSelection::kBestBound: return "best-bound";
    case NodeSelection::kDepthFirst: return "depth-first";
    case NodeSelection::kBestEstimate: return "best-estimate";
  }
  return "unknown";
}

// NODE SELECTION (#293). Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 6.
//
// Every policy here removes one node from `open_` and changes nothing else. That is the whole
// safety argument: the set of open nodes, the bounds that prune them and the incumbent test
// are untouched, so a policy can make the search slower or faster but cannot make it wrong.
//
// TIES ARE BROKEN BY NODE INDEX, always, and that is not cosmetic. Nodes are created in a
// deterministic order, so the smallest index is the oldest node; without it, two runs of the
// same model could take different nodes whenever two bounds compared equal - which on a
// degenerate MILP is most of the tree - and a search that explores a different tree each time
// cannot be debugged, benchmarked or reproduced.
Index BranchAndBound::take_next_open_node(bool diving) {
  std::size_t pick = open_.size() - 1;  // the newest node: depth-first, and the hybrid's dive

  const auto choose_smallest = [&](auto key) {
    double best = std::numeric_limits<double>::infinity();
    Index best_node = std::numeric_limits<Index>::max();
    for (std::size_t k = 0; k < open_.size(); ++k) {
      const Index candidate = open_[k];
      const double value = key(nodes_[static_cast<std::size_t>(candidate)]);
      if (value < best || (value == best && candidate < best_node)) {
        best = value;
        best_node = candidate;
        pick = k;
      }
    }
  };

  switch (node_selection_) {
    case NodeSelection::kDepthFirst: break;  // pick is already the newest
    case NodeSelection::kHybrid:
      if (!diving) choose_smallest([](const TreeNode& node) { return node.bound; });
      break;
    case NodeSelection::kBestBound:
      choose_smallest([](const TreeNode& node) { return node.bound; });
      break;
    case NodeSelection::kBestEstimate:
      choose_smallest([](const TreeNode& node) { return node.estimate; });
      break;
  }

  if (node_selection_ == NodeSelection::kDepthFirst ||
      (node_selection_ == NodeSelection::kHybrid && diving)) {
    ++selected_by_dive_;
  } else {
    ++selected_by_policy_;
  }

  const Index node_index = open_[pick];
  open_.erase(open_.begin() + static_cast<std::ptrdiff_t>(pick));
  deepest_node_ = std::max(deepest_node_, nodes_[static_cast<std::size_t>(node_index)].depth);
  return node_index;
}

// The best-estimate key (Achterberg 2007, sec. 6.1): the node's bound plus, for every column
// still fractional in the relaxation it was branched from, the cheaper of the two directions
// as the pseudocosts price them. It is a guess at where the subtree's integer answer lands,
// which is a different question from the bound - and a better one when what you want is a
// good incumbent early rather than a proof.
//
// Before any pseudocost has been observed the sum is zero and the estimate is the bound, so
// the policy degrades to best-bound at the root rather than to noise.
double BranchAndBound::estimate_from(const std::vector<double>& x, double bound) const {
  double estimate = bound;
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    const double value = x[u];
    const double fraction = value - std::floor(value);
    if (fraction <= integrality_tolerance_ || fraction >= 1.0 - integrality_tolerance_) {
      continue;
    }
    const double down = pseudo_down_count_[u] > 0
                            ? pseudo_down_sum_[u] / static_cast<double>(pseudo_down_count_[u])
                            : 0.0;
    const double up = pseudo_up_count_[u] > 0
                          ? pseudo_up_sum_[u] / static_cast<double>(pseudo_up_count_[u])
                          : 0.0;
    estimate += std::min(down * fraction, up * (1.0 - fraction));
  }
  return estimate;
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
  // Feasible, so it is a plan whether or not it improves: the pool keeps it (#225).
  pool_.offer(objective, x);
  if (have_incumbent_ && objective >= incumbent_internal_ - 1e-12) return false;

  have_incumbent_ = true;
  incumbent_internal_ = objective;
  incumbent_x_ = x;
  return true;
}

bool BranchAndBound::split_integral_node(Index node_index, const Solution& relaxation) {
  // Danna, Fenelon, Gu & Wunderling (IPCO 2007) continue the tree past integral leaves.
  //
  // THE PARTITION. Let j1..jk be the integer columns this node has not fixed and v the
  // relaxation's (integral) point. The node's integer points split, exactly and without
  // overlap, into
  //
  //   x_j1 <= v1-1        x_j1 >= v1+1
  //   x_j1 == v1, x_j2 <= v2-1        x_j1 == v1, x_j2 >= v2+1
  //   ...
  //   x_j1..x_j(k-1) == v, x_jk <= vk-1        ... x_jk >= vk+1
  //   x_j1..x_jk == v        <- the single assignment of the point just found
  //
  // The last piece is never opened: its relaxation optimum is the point already offered,
  // because that point was optimal over the whole node and lies inside the piece. Splitting
  // one column at a time instead - x <= v-1, x == v, x >= v+1, recursing on the middle -
  // re-solves that same point once per unfixed column before it becomes a leaf (Chirag's
  // review of #258); here all 2k side children are created at once and the point costs no
  // further LP.
  //
  // Propagation can leave a fractional bound on an integer column; the integers inside it
  // are what count.
  struct Free {
    Index column;
    double value;
    double lower;
    double upper;
  };
  std::vector<Free> free;
  for (const Index j : integer_columns_) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = std::ceil(working_.col_lower[u] - integrality_tolerance_);
    const double hi = std::floor(working_.col_upper[u] + integrality_tolerance_);
    if (hi > lo) free.push_back({j, std::round(relaxation.col_value[u]), lo, hi});
  }
  if (free.empty()) return false;

  const Index depth = nodes_[static_cast<std::size_t>(node_index)].depth + 1;
  const double bound = internal_objective(relaxation.col_value);
  const WarmStart warm = basis_of(relaxation);
  leave();

  const auto add = [&](Index from, DomainChange change, bool open) {
    TreeNode child;
    child.parent = from;
    child.has_change = true;
    child.change = change;
    child.bound = bound;
    child.depth = depth;
    if (open) child.warm = warm;  // a link is never solved and needs no basis
    nodes_.push_back(std::move(child));
    const auto index = static_cast<Index>(nodes_.size() - 1);
    if (open) open_.push_back(index);
    return index;
  };
  // `fixed` is the tail of a chain of links fixing the columns handled so far; enter() walks
  // parents whether or not they were ever opened, so a child hung off it inherits every fix.
  Index fixed = node_index;
  for (const Free& f : free) {
    if (f.value - 1.0 >= f.lower) add(fixed, DomainChange{f.column, true, f.value - 1.0}, true);
    if (f.value + 1.0 <= f.upper)
      add(fixed, DomainChange{f.column, false, f.value + 1.0}, true);
    fixed = add(fixed, DomainChange{f.column, false, f.value}, false);
    fixed = add(fixed, DomainChange{f.column, true, f.value}, false);
  }
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

}  // namespace sankhya::mip

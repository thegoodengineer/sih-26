// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the cut rounds of branch and cut (#221): the root round, the rounds at shallow
// nodes, and the management of the cut rows they append to the working model.
//
// WHERE A CUT LIVES. Every cut here is a row of `working_`, appended once and kept for the
// rest of the search, so it must be valid for the WHOLE tree and not just for the subtree
// of the node that found it. The root round has that for free. A node round gets it by
// building its MIR cuts on the model's GLOBAL column bounds (the bounds the search started
// with, `global_lower_` / `global_upper_`), not on the node's tightened ones: bound
// substitution against a bound that holds everywhere yields an inequality that holds
// everywhere, whichever node's LP point chose the divisor. The cut is still separated at
// the node's own point, which is what makes it useful there.
//
// WHY THE WARM STARTS SURVIVE. Appending a row to an LP whose basis is optimal, and making
// the new row's logical basic, leaves a basis that is dual feasible (the new logical has
// zero cost) and differs from primal feasibility by the one row that is violated. That is
// the dual simplex's starting state, so every stored basis - the current node's and every
// open node's - is extended with one kBasic entry per new row and stays usable; without
// that every open node would fall back to a cold solve (#65).
//
// AGEING. A cut row whose logical has been basic (the row slack) for kCutRowAgeLimit
// consecutive node solves is not doing anything. Removing the row would invalidate every
// stored basis; making it FREE (both bounds infinite) keeps the structure, keeps every
// basis valid because a free row's logical is basic anyway, and stops the row from ever
// pivoting again. That is the pool this version has: bounded by the depth cap and the
// per-round row cap, aged out in place.
//
// Marchand & Wolsey, "Aggregation and mixed integer rounding to solve MIPs", Operations
// Research 49 (2001); Achterberg, "Constraint integer programming", PhD thesis, TU Berlin
// (2007), ch. 8 for the depth-limited separation and the age-based pool.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "branch_and_bound_internal.hpp"
#include "mir_cuts.hpp"

namespace sankhya::mip {

namespace {

/// Two cuts are the same row when, scaled so the largest coefficient is 1, every entry
/// agrees to this relative tolerance.
constexpr double kSameCutTolerance = 1e-9;

bool same_cut(const Cut& a, const Cut& b) {
  if (a.coeff.size() != b.coeff.size()) return false;
  double scale_a = 0.0;
  double scale_b = 0.0;
  for (std::size_t j = 0; j < a.coeff.size(); ++j) {
    scale_a = std::max(scale_a, std::fabs(a.coeff[j]));
    scale_b = std::max(scale_b, std::fabs(b.coeff[j]));
  }
  if (scale_a == 0.0 || scale_b == 0.0) return scale_a == scale_b;
  for (std::size_t j = 0; j < a.coeff.size(); ++j) {
    if (std::fabs(a.coeff[j] / scale_a - b.coeff[j] / scale_b) > kSameCutTolerance) {
      return false;
    }
  }
  return std::fabs(a.rhs / scale_a - b.rhs / scale_b) <=
         kSameCutTolerance * std::max(1.0, std::fabs(a.rhs / scale_a));
}

}  // namespace

void BranchAndBound::append_cut_rows(const std::vector<Cut>& accepted) {
  const Index old_rows = working_.num_rows();
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
      if (std::fabs(cut.coeff[static_cast<std::size_t>(j)]) > tol::kZeroDrop) {
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

  if (first_cut_row_ < 0) first_cut_row_ = old_rows;
  for (const Cut& cut : accepted) {
    pool_cuts_.push_back(cut);
    cut_row_slack_.push_back(0);
    cut_row_free_.push_back(false);
  }
  scaling_ = build_node_scaling(working_, node_options_);
}

void BranchAndBound::resize_warm_starts(Index rows) {
  const auto extend = [rows](WarmStart& warm) {
    if (warm.empty()) return;
    warm.row_status.resize(static_cast<std::size_t>(rows), BasisStatus::kBasic);
  };
  extend(current_warm_);
  for (const Index open : open_) extend(nodes_[static_cast<std::size_t>(open)].warm);
}

bool BranchAndBound::is_pooled_duplicate(const Cut& cut) const {
  return std::any_of(pool_cuts_.begin(), pool_cuts_.end(),
                     [&cut](const Cut& pooled) { return same_cut(pooled, cut); });
}

void BranchAndBound::root_cut_round(Solution* relaxation) {
  const Index original_root_rows = working_.num_rows();
  Model pre_cut_model = working_;
  auto pre_cut_scaling = scaling_;
  Solution initial_relaxation = *relaxation;

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
  // MIR cuts from the model's own rows (#221): built from original coefficients rather
  // than tableau rows, so they carry none of the Gomory cuts' numerical fragility.
  if (options_.get_bool("enable_mir_cuts")) {
    std::vector<Cut> mir = generate_mir_cuts(working_, initial_relaxation);
    candidates.insert(candidates.end(), mir.begin(), mir.end());
  }

  auto filtered = filter_and_deduplicate_cuts(working_, initial_relaxation, candidates);
  std::vector<Cut> accepted;
  for (const auto& fc : filtered) {
    if (fc.reason == CutFilterReason::kAccepted) accepted.push_back(fc.cut);
  }
  if (accepted.empty()) return;

  append_cut_rows(accepted);
  resize_warm_starts(working_.num_rows());
  Solution final_relaxation = solve_node();
  if (final_relaxation.status == SolveStatus::kOptimal) {
    *relaxation = std::move(final_relaxation);
    root_cuts_applied_ = static_cast<Count>(accepted.size());
    root_bound_after_cuts_internal_ = internal_objective(relaxation->col_value);
    return;
  }
  logger_.info("Root cuts induced failure: {}; rolled back to initial relaxation",
               to_string(final_relaxation.status));
  working_ = std::move(pre_cut_model);
  scaling_ = std::move(pre_cut_scaling);
  *relaxation = std::move(initial_relaxation);
  pool_cuts_.resize(pool_cuts_.size() - accepted.size());
  cut_row_slack_.resize(pool_cuts_.size());
  cut_row_free_.resize(pool_cuts_.size());
  if (pool_cuts_.empty()) first_cut_row_ = -1;
  resize_warm_starts(working_.num_rows());
}

void BranchAndBound::tree_cut_round(Index depth, Solution* relaxation) {
  if (most_fractional(relaxation->col_value) < 0) return;
  std::vector<Cut> candidates =
      generate_mir_cuts(working_, *relaxation, global_lower_, global_upper_);
  if (candidates.empty()) return;
  auto filtered = filter_and_deduplicate_cuts(working_, *relaxation, candidates);
  std::vector<Cut> accepted;
  for (auto& fc : filtered) {
    if (fc.reason != CutFilterReason::kAccepted) continue;
    if (is_pooled_duplicate(fc.cut)) continue;
    accepted.push_back(std::move(fc.cut));
    if (static_cast<Index>(accepted.size()) >= tree_cut_rows_per_round_) break;
  }
  if (accepted.empty()) return;
  ++tree_cut_rounds_;

  Model pre_cut_model = working_;
  auto pre_cut_scaling = scaling_;
  const std::size_t pool_before = pool_cuts_.size();
  append_cut_rows(accepted);
  resize_warm_starts(working_.num_rows());
  // Restart from this node's own optimal basis, extended: dual feasible, one pivot per
  // violated cut away from the new optimum in the usual case.
  current_warm_ = basis_of(*relaxation);
  current_warm_.row_status.resize(static_cast<std::size_t>(working_.num_rows()),
                                  BasisStatus::kBasic);
  Solution after = solve_node();
  if (after.status == SolveStatus::kOptimal) {
    tree_cuts_applied_ += static_cast<Count>(accepted.size());
    *relaxation = std::move(after);
    return;
  }
  // A cut that broke the node LP is not worth keeping; the search continues on the node's
  // pre-cut relaxation, which was optimal. (An infeasible re-solve would mean the cuts cut
  // the node off entirely, which valid cuts cannot do to a node with a feasible relaxation
  // unless numerically; treated the same way.)
  logger_.verbose("tree cuts at depth {} induced {}; rolled back", depth,
                  to_string(after.status));
  working_ = std::move(pre_cut_model);
  scaling_ = std::move(pre_cut_scaling);
  pool_cuts_.resize(pool_before);
  cut_row_slack_.resize(pool_before);
  cut_row_free_.resize(pool_before);
  if (pool_cuts_.empty()) first_cut_row_ = -1;
  resize_warm_starts(working_.num_rows());
  current_warm_ = basis_of(*relaxation);
}

void BranchAndBound::age_cut_rows(const Solution& relaxation) {
  if (first_cut_row_ < 0) return;
  if (relaxation.row_status.size() != static_cast<std::size_t>(working_.num_rows())) return;
  for (std::size_t k = 0; k < pool_cuts_.size(); ++k) {
    if (cut_row_free_[k]) continue;
    const auto row = static_cast<std::size_t>(first_cut_row_) + k;
    if (relaxation.row_status[row] == BasisStatus::kBasic) {
      if (++cut_row_slack_[k] >= kCutRowAgeLimit) {
        working_.row_lower[row] = -kInfinity;
        working_.row_upper[row] = kInfinity;
        cut_row_free_[k] = true;
        ++cut_rows_aged_out_;
      }
    } else {
      cut_row_slack_[k] = 0;
    }
  }
}

}  // namespace sankhya::mip

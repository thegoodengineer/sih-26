// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: saving the search and resuming it (#287). The file format and
// its guarantees are in checkpoint.hpp; this is what goes into it and how it comes back.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>

namespace sankhya::mip {

TreeCheckpoint BranchAndBound::make_checkpoint() const {
  TreeCheckpoint c;
  c.model_fingerprint = fmt::format("{:016x}", original_.fingerprint());
  c.problem_class = quadratic_ ? "miqp" : "milp";
  c.num_rows = original_.num_rows();
  c.num_cols = original_.num_cols();
  c.integrality_tolerance = integrality_tolerance_;
  c.have_incumbent = have_incumbent_;
  if (have_incumbent_) c.incumbent_x = incumbent_x_;
  c.nodes_explored = static_cast<Count>(nodes_explored_);
  c.nodes_pruned = nodes_pruned_;
  c.pseudo_down_sum = pseudo_down_sum_;
  c.pseudo_up_sum = pseudo_up_sum_;
  c.pseudo_down_count = pseudo_down_count_;
  c.pseudo_up_count = pseudo_up_count_;
  for (const Index open : open_) {
    const TreeNode& leaf = nodes_[static_cast<std::size_t>(open)];
    CheckpointNode n;
    n.bound = leaf.bound;
    n.depth = leaf.depth;
    n.estimate = leaf.estimate;
    // The node's domain is the chain of single changes from it up to the root.
    for (Index walk = open; walk >= 0;) {
      const TreeNode& node = nodes_[static_cast<std::size_t>(walk)];
      if (node.has_change) {
        n.domain.push_back(
            CheckpointChange{node.change.column, node.change.is_upper, node.change.value});
      }
      walk = node.parent;
    }
    c.open.push_back(std::move(n));
  }
  return c;
}

void BranchAndBound::save_checkpoint() {
  if (checkpoint_path_.empty()) return;
  std::string error;
  last_checkpoint_at_ = static_cast<Count>(nodes_explored_);
  if (!write_checkpoint(checkpoint_path_, make_checkpoint(), &error)) {
    // A diagnostic that could not be written must not end the search it describes.
    logger_.warning("checkpoint: {}", error);
    return;
  }
  ++checkpoints_written_;
  logger_.info("Checkpoint written to {}: {} open nodes after {} nodes explored",
               checkpoint_path_, open_.size(), nodes_explored_);
}

std::string BranchAndBound::restore_checkpoint(const std::string& path) {
  TreeCheckpoint c;
  std::string error;
  if (!read_checkpoint(path, &c, &error)) return error;

  // ---- Is it THIS search's checkpoint? Every check before anything is changed. ------------
  const std::string fingerprint = fmt::format("{:016x}", original_.fingerprint());
  if (c.model_fingerprint != fingerprint) {
    return fmt::format(
        "the checkpoint was written for model {} and this model is {} - a "
        "different model, or the same one under different presolve settings",
        c.model_fingerprint, fingerprint);
  }
  const std::string cls = quadratic_ ? "miqp" : "milp";
  if (c.problem_class != cls || c.num_rows != original_.num_rows() ||
      c.num_cols != original_.num_cols()) {
    return fmt::format("the checkpoint is a {} of {} x {} and this is a {} of {} x {}",
                       c.problem_class, c.num_rows, c.num_cols, cls, original_.num_rows(),
                       original_.num_cols());
  }
  if (c.integrality_tolerance != integrality_tolerance_) {
    return fmt::format(
        "the checkpoint was written with integrality_tolerance {:g} and this "
        "solve uses {:g}; a node the first search pruned as integral might not "
        "be integral here",
        c.integrality_tolerance, integrality_tolerance_);
  }
  const auto n = static_cast<std::size_t>(original_.num_cols());
  const bool sized = c.pseudo_down_sum.size() == n && c.pseudo_up_sum.size() == n &&
                     c.pseudo_down_count.size() == n && c.pseudo_up_count.size() == n &&
                     (!c.have_incumbent || c.incumbent_x.size() == n);
  if (!sized) return "the checkpoint's per-column arrays do not match the model's columns";
  for (const CheckpointNode& node : c.open) {
    for (const CheckpointChange& change : node.domain) {
      if (change.column < 0 || change.column >= original_.num_cols() ||
          !std::isfinite(change.value)) {
        return fmt::format(
            "an open node branches on column {} with value {}, which this model "
            "does not have",
            change.column, change.value);
      }
    }
  }

  // ---- Rebuild. The fresh root stays as node 0, processed; each open node becomes a chain
  // of single changes hung from it, and only the chain's leaf is open.
  open_.clear();
  for (const CheckpointNode& saved : c.open) {
    Index parent = 0;
    for (std::size_t k = 0; k < saved.domain.size(); ++k) {
      TreeNode link;
      link.parent = parent;
      link.has_change = true;
      link.change =
          DomainChange{saved.domain[k].column, saved.domain[k].is_upper, saved.domain[k].value};
      link.bound = saved.bound;
      link.depth = static_cast<Index>(k + 1);
      link.estimate = saved.estimate;
      nodes_.push_back(std::move(link));
      parent = static_cast<Index>(nodes_.size() - 1);
    }
    if (parent == 0) {
      // An open node with no branching at all is the root itself: re-open it.
      nodes_[0].bound = saved.bound;
    }
    nodes_[static_cast<std::size_t>(parent)].depth = saved.depth;
    open_.push_back(parent);
  }
  pseudo_down_sum_ = c.pseudo_down_sum;
  pseudo_up_sum_ = c.pseudo_up_sum;
  pseudo_down_count_ = c.pseudo_down_count;
  pseudo_up_count_ = c.pseudo_up_count;
  nodes_explored_ = static_cast<decltype(nodes_explored_)>(c.nodes_explored);
  nodes_pruned_ = c.nodes_pruned;

  // THE INCUMBENT IS RE-VERIFIED, NOT TRUSTED. offer_incumbent() checks integrality, the
  // bounds and every row of the model and recomputes the objective itself; a saved point that
  // fails is dropped with a warning and the search continues without it.
  if (c.have_incumbent && !offer_incumbent(c.incumbent_x)) {
    logger_.warning(
        "resume: the saved incumbent did not pass the model's checks and was "
        "dropped");
  }
  logger_.info("Resumed from {}: {} open nodes, {} nodes already explored, {}", path,
               open_.size(), nodes_explored_,
               have_incumbent_ ? fmt::format("incumbent {:.10g}", reported(incumbent_internal_))
                               : std::string("no incumbent yet"));
  return {};
}

}  // namespace sankhya::mip

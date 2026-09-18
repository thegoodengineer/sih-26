// SPDX-License-Identifier: Apache-2.0
// SANKHYA - conflict analysis inside the branch and bound (#292): learning from a node proved
// infeasible, and using what was learned in every later node's propagation.
//
// The pure parts - the literal, the store, the Farkas proof, the deletion filter - live in
// conflict.cpp and are tested on their own. This file owns WHEN: which nodes are analysed,
// the work each analysis may spend, and the statistics. See conflict.hpp for the references
// and for why a stored conflict cannot remove a feasible point.

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../util/profiler.hpp"
#include "branch_and_bound_internal.hpp"
#include "conflict.hpp"

namespace sankhya::mip {

void BranchAndBound::init_conflicts() {
  conflicts_enabled_ = options_.get_bool("conflict_analysis");
  conflict_minimize_ = options_.get_bool("conflict_minimize");
  conflict_max_size_ = static_cast<std::size_t>(options_.get_int("conflict_max_size"));
  conflicts_ = ConflictStore(
      conflicts_enabled_ ? static_cast<std::size_t>(options_.get_int("conflict_max")) : 0);
}

bool BranchAndBound::propagate_conflicts(bool* changed) {
  for (ConflictStore::Entry& entry : conflicts_.entries()) {
    const ConflictVerdict verdict = check_conflict(entry.literals, working_.col_lower,
                                                   working_.col_upper, integrality_tolerance_);
    if (!verdict.infeasible && verdict.implied < 0) continue;
    if (!analysing_) {
      ++entry.uses;
      entry.last_used = static_cast<std::int64_t>(nodes_explored_);
    }
    if (verdict.infeasible) {
      if (!analysing_) conflict_pruned_ = true;
      return false;
    }
    // Every other literal holds, so this one must be false: an integer bound one step past it.
    const ConflictLiteral bound =
        negation(entry.literals[static_cast<std::size_t>(verdict.implied)]);
    const auto u = static_cast<std::size_t>(bound.column);
    if (bound.is_upper) {
      if (bound.value < working_.col_upper[u]) tighten_upper(u, bound.value);
    } else if (bound.value > working_.col_lower[u]) {
      tighten_lower(u, bound.value);
    }
    if (!analysing_) ++conflict_stats_.tightenings;
    *changed = true;
    if (working_.col_lower[u] > working_.col_upper[u] + tol::kPrimalFeasibility) return false;
  }
  return true;
}

void BranchAndBound::analyze_conflict(Index node_index, ConflictSource source,
                                      const std::vector<double>* farkas) {
  if (!conflicts_enabled_ || node_index == 0) return;
  // THE ANALYSIS HAS A BUDGET IN WORK, NOT IN SECONDS, so a rerun learns the same conflicts
  // (#288): verification calls in total may not run far ahead of the nodes the search has
  // explored. A search that finds infeasible nodes cheaply should not spend its time here.
  if (conflict_stats_.checks > kConflictChecksPerNode * nodes_explored_ + kConflictChecksBase) {
    return;
  }
  ProfileScope timed(logger_.profiler(), "conflict analysis", ProfileMode::kDetailed);
  const Timer clock;
  ++conflict_stats_.detected;

  // The node's decisions, root first, the tightest per (column, direction) kept at the
  // position of its LAST occurrence, so the node's own decision stays last.
  std::vector<ConflictLiteral> chain;
  for (Index walk = node_index; walk >= 0;) {
    const TreeNode& node = nodes_[static_cast<std::size_t>(walk)];
    if (node.has_change) {
      chain.push_back(
          ConflictLiteral{node.change.column, node.change.is_upper, node.change.value});
    }
    walk = node.parent;
  }
  std::reverse(chain.begin(), chain.end());
  {
    std::map<std::pair<Index, bool>, std::size_t> last;
    for (std::size_t k = 0; k < chain.size(); ++k) {
      last[{chain[k].column, chain[k].is_upper}] = k;
    }
    std::vector<ConflictLiteral> unique;
    for (std::size_t k = 0; k < chain.size(); ++k) {
      if (last[{chain[k].column, chain[k].is_upper}] != k) continue;
      ConflictLiteral literal = chain[k];
      for (const ConflictLiteral& other : chain) {
        if (other.column != literal.column || other.is_upper != literal.is_upper) continue;
        literal.value = literal.is_upper ? std::min(literal.value, other.value)
                                         : std::max(literal.value, other.value);
      }
      unique.push_back(literal);
    }
    chain = std::move(unique);
  }
  if (chain.empty()) return;

  // The check every stored conflict passes, run from the GLOBAL bounds (the caller has left
  // the node): the literals, then propagation - which may itself prove the box empty - then
  // the Farkas proof, if there is one, on the box propagation left.
  FarkasProof proof;
  analysing_ = true;
  const auto verify = [&](const std::vector<ConflictLiteral>& literals) {
    ++conflict_stats_.checks;
    saved_.clear();
    for (const ConflictLiteral& literal : literals) {
      const auto u = static_cast<std::size_t>(literal.column);
      if (literal.is_upper) {
        if (literal.value < working_.col_upper[u]) tighten_upper(u, literal.value);
      } else if (literal.value > working_.col_lower[u]) {
        tighten_lower(u, literal.value);
      }
    }
    bool proved = !propagate();
    if (!proved && proof.usable) {
      proved = farkas_contradicts(proof, working_.col_lower, working_.col_upper);
    }
    leave();
    return proved;
  };

  bool proved = false;
  if (source == ConflictSource::kLp) {
    if (farkas != nullptr && !farkas->empty()) {
      // Engines differ in the sign they report the multipliers with; the proof is checked
      // either way round, and only a proof that holds is used.
      proof = aggregate_farkas(working_, *farkas);
      proved = verify(chain);
      if (!proved) {
        std::vector<double> flipped(*farkas);
        for (double& value : flipped) value = -value;
        proof = aggregate_farkas(working_, flipped);
        proved = verify(chain);
      }
    }
  } else {
    proved = verify(chain);
  }
  if (!proved) {
    analysing_ = false;
    ++conflict_stats_.rejected;
    conflict_stats_.seconds += clock.elapsed_seconds();
    return;
  }

  std::vector<ConflictLiteral> start = chain;
  if (proof.usable) {
    // The literals the proof leans on directly. Often enough on its own; when propagation
    // needed the others, the full chain is the start instead.
    std::vector<ConflictLiteral> leaned;
    for (const ConflictLiteral& literal : chain) {
      const double dj = proof.d[static_cast<std::size_t>(literal.column)];
      if ((literal.is_upper && dj > 0.0) || (!literal.is_upper && dj < 0.0)) {
        leaned.push_back(literal);
      }
    }
    if (!leaned.empty() && leaned.size() < chain.size() && verify(leaned)) start = leaned;
  }
  std::vector<ConflictLiteral> literals = start;
  if (conflict_minimize_ && literals.size() > 1) {
    // The node's own decision is needed whenever its parent was not infeasible, which the
    // parent's solved LP shows; the filter does not spend a check on it.
    const bool keep_last = literals.back() == chain.back();
    int used = 0;
    literals = minimize_conflict(std::move(literals), keep_last, kConflictMinimizeChecks,
                                 verify, &used);
  }
  analysing_ = false;
  conflict_stats_.seconds += clock.elapsed_seconds();

  if (literals.size() > conflict_max_size_) {
    ++conflict_stats_.too_long;
    return;
  }
  const std::size_t size = literals.size();
  if (conflicts_.add(canonical(std::move(literals)), source,
                     static_cast<std::int64_t>(nodes_explored_))) {
    ++conflict_stats_.learned;
    if (size < chain.size()) ++conflict_stats_.minimized;
    conflict_stats_.sizes.push_back(static_cast<std::int64_t>(size));
  }
}

void BranchAndBound::report_conflicts() {
  if (!conflicts_enabled_) return;
  if (conflict_stats_.detected > 0) {
    std::vector<std::int64_t> sizes = conflict_stats_.sizes;
    std::sort(sizes.begin(), sizes.end());
    double mean = 0.0;
    for (const std::int64_t size : sizes) mean += static_cast<double>(size);
    if (!sizes.empty()) mean /= static_cast<double>(sizes.size());
    logger_.info(
        "Conflicts: {} infeasible node(s) analysed, {} learned ({} minimised), {} not proved "
        "from the global bounds, {} too long, {} duplicate(s), {} evicted, {} held; size "
        "min {} median {} mean {:.1f} max {}; {} node(s) pruned and {} bound(s) tightened by "
        "them; {} check(s), {:.3f}s",
        conflict_stats_.detected, conflict_stats_.learned, conflict_stats_.minimized,
        conflict_stats_.rejected, conflict_stats_.too_long, conflicts_.duplicates(),
        conflicts_.evicted(), conflicts_.size(), sizes.empty() ? 0 : sizes.front(),
        sizes.empty() ? 0 : sizes[sizes.size() / 2], mean, sizes.empty() ? 0 : sizes.back(),
        conflict_stats_.nodes_pruned, conflict_stats_.tightenings, conflict_stats_.checks,
        conflict_stats_.seconds);
  }
  if (Profiler* profiler = logger_.profiler(); profiler != nullptr) {
    profiler->count("conflicts learned", conflict_stats_.learned);
    profiler->count("nodes pruned by conflicts", conflict_stats_.nodes_pruned);
  }
  const std::string path = options_.get_string("conflict_out");
  if (path.empty()) return;
  std::FILE* out = std::fopen(path.c_str(), "wb");
  if (out == nullptr) {
    logger_.warning("conflict_out: cannot open {} for writing", path);
    return;
  }
  const std::string text = conflicts_to_json(conflicts_, conflict_stats_);
  const bool written = std::fwrite(text.data(), 1, text.size(), out) == text.size();
  if (std::fclose(out) != 0 || !written) {
    logger_.warning("conflict_out: writing {} failed", path);
  }
}

}  // namespace sankhya::mip

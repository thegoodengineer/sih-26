// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch-and-bound checkpoints: what is saved, how it is written, how it is refused
// (#287).
//
// WHAT A CHECKPOINT IS. Enough of a search to continue it: the incumbent, every open node's
// domain (the branching bounds from the root down to it) with the bound it inherited, the
// pseudocosts, and the counters. The search that loads it re-verifies the incumbent against
// the model rather than trusting the file, and re-solves each open node's LP from its domain,
// so a checkpoint can make a resumed search SLOWER than the original (bases and root cuts are
// not saved) but never wrong: every saved bound is a valid lower bound on its node, because
// it was proved by that node's parent.
//
// THE FORMAT is JSON, versioned, never a raw struct: {"format", "version", "sankhya":
// {version, commit}, "checksum", "state"}. Doubles round-trip exactly through nlohmann's
// shortest-round-trip printing; an infinity is written as a string, which JSON has no literal
// for. The checksum is FNV-1a over the compact dump of "state", so a truncated or edited file
// is refused before any of it is used.
//
// WRITES ARE ATOMIC. The checkpoint is written to <path>.tmp and moved over <path> only once
// it is complete and closed, so a crash or a full disk mid-write leaves the previous
// checkpoint intact - a failed write never destroys the last good one.

#pragma once

#include <string>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya::mip {

/// One bound change on the path from the root to a node.
struct CheckpointChange {
  Index column = 0;
  bool is_upper = false;
  double value = 0.0;
};

struct CheckpointNode {
  std::vector<CheckpointChange> domain;  ///< root-ward order does not matter: all tighten
  double bound = 0.0;                    ///< minimise space, proved by the node's parent
  Index depth = 0;
  double estimate = 0.0;
};

struct TreeCheckpoint {
  /// Checked against the model on resume. The model the TREE sees - after presolve - so a
  /// resume under different presolve settings is a different model, and is refused.
  std::string model_fingerprint;
  std::string problem_class;  ///< "milp" or "miqp"
  Index num_rows = 0;
  Index num_cols = 0;
  double integrality_tolerance = 0.0;

  bool have_incumbent = false;
  std::vector<double> incumbent_x;

  Count nodes_explored = 0;
  Count nodes_pruned = 0;
  std::vector<double> pseudo_down_sum;
  std::vector<double> pseudo_up_sum;
  std::vector<Count> pseudo_down_count;
  std::vector<Count> pseudo_up_count;

  std::vector<CheckpointNode> open;
};

/// The format this build writes, and the only one it reads.
inline constexpr int kCheckpointVersion = 1;

/// Write atomically: to <path>.tmp, then moved over <path>. False, with `error`, when any
/// step fails - and then <path> is exactly what it was before the call.
bool write_checkpoint(const std::string& path, const TreeCheckpoint& checkpoint,
                      std::string* error);

/// Read and validate the container: parseable, the right format name, this version, and a
/// checksum that matches. The CONTENT (model identity, dimensions) is the caller's to check.
bool read_checkpoint(const std::string& path, TreeCheckpoint* checkpoint, std::string* error);

}  // namespace sankhya::mip

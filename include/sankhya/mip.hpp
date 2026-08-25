// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound for mixed-integer linear programming.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {

/// Solve a MILP by branch and bound over the revised primal simplex.
///
/// Accepts a model with or without integrality; with none it is a single LP solve and says
/// so. Never throws: every failure comes back as a status.
///
/// The returned Solution distinguishes the two outcomes that matter and are easy to
/// conflate: kOptimal means the search CLOSED - the incumbent is proven best - while
/// kFeasible means an incumbent exists but a limit stopped the proof, and dual_bound then
/// carries the best bound still open.
[[nodiscard]] Solution solve_branch_and_bound(const Model& model, const Options& options,
                                              Logger& logger);

}  // namespace sankhya::mip

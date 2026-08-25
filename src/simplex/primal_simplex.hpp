// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bounded-variable revised primal simplex.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {

/// Solve a continuous LP with the revised primal simplex. Integrality is IGNORED: this is
/// the node solver branch-and-cut will call in Phase 5, and it is the caller's job to know
/// whether it wanted a relaxation. solve() in src/core refuses a MILP for exactly this
/// reason rather than quietly returning a fractional point labelled optimal.
[[nodiscard]] Solution solve_primal_simplex(const Model& model, const Options& options,
                                            Logger& logger);

}  // namespace sankhya

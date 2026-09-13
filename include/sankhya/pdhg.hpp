// SPDX-License-Identifier: Apache-2.0
// SANKHYA - restarted PDHG, the first-order LP engine.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::pdhg {

/// Solve an LP with restarted primal-dual hybrid gradient.
///
/// Requires a model with no integrality and no quadratic objective; the solve() dispatcher
/// checks that. Never throws: every failure comes back as a status.
///
/// `control` (#223): optional progress/interrupt channel, polled at the same point the
/// iteration loop already checks its time limit.
[[nodiscard]] Solution solve_pdhg(const Model& model, const Options& options, Logger& logger,
                                  SolveControl* control = nullptr);

}  // namespace sankhya::pdhg

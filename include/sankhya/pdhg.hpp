// SPDX-License-Identifier: Apache-2.0
// SANKHYA - restarted PDHG, the first-order LP engine.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::pdhg {

/// Solve an LP with restarted primal-dual hybrid gradient.
///
/// Requires a model with no integrality and no quadratic objective; the solve() dispatcher
/// checks that. Never throws: every failure comes back as a status.
[[nodiscard]] Solution solve_pdhg(const Model& model, const Options& options, Logger& logger);

}  // namespace sankhya::pdhg

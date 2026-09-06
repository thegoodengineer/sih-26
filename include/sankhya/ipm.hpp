// SPDX-License-Identifier: Apache-2.0
// SANKHYA - primal-dual interior-point method for LP (#56).
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::ipm {

/// Solve a continuous LP with Mehrotra's predictor-corrector method on the normal
/// equations. Integrality is ignored, as by every LP engine; the dispatcher routes MILPs
/// elsewhere. NO BASIS IS PRODUCED: the returned Solution carries primal values, row duals
/// and reduced costs to the tolerance the method converged to, and empty statuses. The
/// simplex remains the node engine for branch and bound for that reason.
[[nodiscard]] Solution solve_ipm(const Model& model, const Options& options, Logger& logger);

}  // namespace sankhya::ipm

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

/// A starting point handed in from outside: a primal point, its row duals and its reduced
/// costs, in the ORIGINAL model's units and sign convention (the ones a Solution carries).
/// The polish of a PDHG answer is the caller (#229); the method scales them as it scales
/// the model, projects the point into its box, floors the slacks and multipliers so the
/// first iterate is interior, and lets the residuals absorb the rest. Vectors of the wrong
/// length are ignored and the solve is cold.
struct WarmStart {
  std::vector<double> col_value;
  std::vector<double> row_dual;
  std::vector<double> col_dual;
};

/// As above, started from `warm` when it is not null. With a warm start the option
/// polish_max_factor_nonzeros is honoured: when the ordering says the factor would exceed
/// it, the solve is declined with status kNotSolved and the reason in the message, before
/// anything expensive is built.
[[nodiscard]] Solution solve_ipm(const Model& model, const Options& options, Logger& logger,
                                 const WarmStart* warm);

}  // namespace sankhya::ipm

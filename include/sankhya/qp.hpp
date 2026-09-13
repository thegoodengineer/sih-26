// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex quadratic programming.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"

namespace sankhya::qp {

/// Solve a convex QP:
///
///     minimize    offset + c'x + 0.5 x' Q x
///     subject to  row_lower <= A x <= row_upper
///                 col_lower <=   x  <= col_upper
///
/// A non-convex Hessian is REFUSED (kModelError), never solved to whatever local point the
/// iteration happens to reach. Convexity is decided before any arithmetic starts; see
/// src/qp/convexity.hpp.
///
/// `control` (#223): optional progress/interrupt channel, polled at the same point the
/// iteration loop already checks its time limit.
[[nodiscard]] Solution solve_convex_qp(const Model& model, const Options& options,
                                       Logger& logger, SolveControl* control = nullptr);

}  // namespace sankhya::qp

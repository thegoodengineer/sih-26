// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a convex NLP engine: smooth convex objective, linear constraints (#226).
//
// WHAT IT SOLVES.  min  f(x) + c'x + 0.5 x'Qx  s.t.  row_lower <= A x <= row_upper,
// col_lower <= x <= col_upper, with f a convex expression from src/nlp (#296) and every column
// continuous. Nonlinear CONSTRAINTS, integer columns and an objective the convexity rules
// cannot prove convex are refused with kNotSolved and the reason - a local method on a
// non-convex problem returns a local point, and labelling that optimal is the one outcome
// worse than refusing.
//
// THE METHOD. Condat-Vu's primal-dual splitting (Condat, JOTA 158, 2013; Vu, Adv. Comput.
// Math. 38, 2013), the algorithm the convex QP engine already runs, with the gradient of the
// smooth part taken from the expression graph's exact reverse-mode AD instead of from c + Qx:
//
//     x+ = proj_box( x - tau (grad f(x) + A' y) ),   y+ = prox( y + sigma A (2 x+ - x) )
//
// A QP's gradient has one global Lipschitz constant, ||Q||. A general convex f does not:
// entropy's curvature is 1/x, unbounded towards the boundary. So L is found by BACKTRACKING -
// the descent lemma f(x+) <= f(x) + grad f(x)'(x+ - x) + L/2 ||x+ - x||^2 is tested at every
// step and L doubled until it holds - and a step that leaves f's DOMAIN counts as a failed
// test too. L only ever INCREASES, so on a problem whose curvature is bounded where the
// iterates go the step sizes become constant after finitely many doublings, and from there
// the iteration is Condat-Vu with fixed steps, for which the convergence proof holds.
//
// OPTIMAL MEANS CHECKED. The iteration stops on its own fixed-point residual; the status is
// decided afterwards by an independent KKT check at the returned point with the exact
// gradient: primal feasibility and the projected-gradient stationarity measure, both to
// `nlp_tolerance`. A point that fails it is reported as kFeasible or at its limit, never as
// optimal on the iteration's word.

#pragma once

#include "nlp/nonlinear_model.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::nlp {

/// Solve a convex NLP with linear constraints. See the file comment for what is refused.
[[nodiscard]] Solution solve_convex_nlp(const NonlinearModel& model, const Options& options,
                                        Logger& logger);

/// The same, with a logger built from `options` (log_to_console).
[[nodiscard]] Solution solve_convex_nlp(const NonlinearModel& model, const Options& options);

}  // namespace sankhya::nlp

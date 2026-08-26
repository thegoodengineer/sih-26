// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the status/measurement reconciliation guard.
//
// Declared in its own header so it can be tested DIRECTLY. The guard's job is to catch an
// engine that claims more than its own returned point supports; testing it only through an
// engine means the test can exist only while some engine is misbehaving, and it silently
// loses its subject the moment that engine is fixed. That is exactly what happened here -
// PDHG now polices itself, so the path through PDHG no longer reaches this code.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {

/// Force the reported status to agree with the measured quality of the point. See the
/// definition in src/core/solve.cpp for the full reasoning.
///
/// `check_dual` is false for a MILP: a branch-and-bound incumbent comes from a node LP whose
/// bounds were tightened by branching, so its reduced costs are dual feasible for that node
/// and generally not for the original model.
void reconcile_status_with_measurement(Solution* solution, const Options& options,
                                       Logger& logger, bool check_dual);

}  // namespace sankhya

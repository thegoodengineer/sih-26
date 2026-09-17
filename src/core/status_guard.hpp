// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the status/measurement reconciliation guard.
//
// Declared in its own header so it can be tested DIRECTLY. The guard's job is to catch an
// engine that claims more than its own returned point supports; testing it only through an
// engine means the test can exist only while some engine is misbehaving, and it silently
// loses its subject the moment that engine is fixed. That is exactly what happened here -
// PDHG now polices itself, so the path through PDHG no longer reaches this code.
#pragma once

#include <new>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/timer.hpp"

namespace sankhya {

/// Force the reported status to agree with the measured quality of the point. See the
/// definition in src/core/solve.cpp for the full reasoning.
///
/// `check_dual` is false for a MILP: a branch-and-bound incumbent comes from a node LP whose
/// bounds were tightened by branching, so its reduced costs are dual feasible for that node
/// and generally not for the original model.
void reconcile_status_with_measurement(Solution* solution, const Options& options,
                                       Logger& logger, bool check_dual);

/// Run an engine and turn an out-of-memory condition into a status (#246).
///
/// Every other failure in this project is a SolveStatus with a message and a stats blob; a
/// std::bad_alloc escaping solve() was the one that was not - the CLI died with the log
/// buffer, the C API's caller got an error code and nothing else, a benchmark runner saw
/// `no_output`. The engines allocate in the places where a model can be too large for the
/// machine (the ordering's quotient graph, the factor, a dense working set), and each has
/// its own budget so that this is the last resort, not the first line. When it fires the
/// unwinding has already released what the engine held, so filling in the Solution here is
/// safe, and the message names the engine so a reader knows which budget to lower.
///
/// Declared here, beside the other guard, so a test can throw through it directly rather
/// than only through an engine that happens to exhaust memory today.
template <typename Body>
Solution run_engine_guarded(Body&& body, std::string_view engine, const Timer& timer,
                            Logger& logger) {
  try {
    return body();
  } catch (const std::bad_alloc&) {
    Solution solution;
    solution.status = SolveStatus::kNumericalError;
    solution.algorithm = std::string(engine);
    solution.message = fmt::format(
        "the solve ran out of memory inside the {} after {:.1f}s; the model is larger than "
        "this machine can hold for that engine - lower its size budget, add memory, or use "
        "another engine",
        engine, timer.elapsed_seconds());
    solution.solve_seconds = timer.elapsed_seconds();
    logger.error("{}", solution.message);
    return solution;
  }
}

}  // namespace sankhya

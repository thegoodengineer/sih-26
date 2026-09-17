// SPDX-License-Identifier: Apache-2.0
// SANKHYA - resource limits, read once and interpreted in one place (#289).

#include "core/resource_limits.hpp"

#include <cmath>
#include <limits>

#include <fmt/format.h>

namespace sankhya {

namespace {
/// What the termination options carry when nothing is to stop the solve on the clock. The
/// registry's default for time_limit, and the value an engine is handed when the budget is
/// not to bind.
constexpr double kNoWallClockLimit = std::numeric_limits<double>::max();
}  // namespace

const char* to_string(LimitReason reason) noexcept {
  switch (reason) {
    case LimitReason::kNone: return "none";
    case LimitReason::kInterrupt: return "user_interrupt";
    case LimitReason::kTime: return "time_limit";
    case LimitReason::kIterations: return "iteration_limit";
    case LimitReason::kNodes: return "node_limit";
  }
  return "none";
}

SolveStatus status_for(LimitReason reason) noexcept {
  switch (reason) {
    case LimitReason::kNone: return SolveStatus::kNotSolved;
    case LimitReason::kInterrupt: return SolveStatus::kInterrupted;
    case LimitReason::kTime: return SolveStatus::kTimeLimit;
    case LimitReason::kIterations: return SolveStatus::kIterationLimit;
    case LimitReason::kNodes: return SolveStatus::kNodeLimit;
  }
  return SolveStatus::kNotSolved;
}

ResourceLimits::ResourceLimits(const Options& options, Logger& logger) {
  const double seconds = options.get_double("time_limit");
  if (std::isnan(seconds) || seconds < 0.0) {
    // The registry refuses this through set_from_string; set_double does not validate, so a
    // program that calls it directly reaches here. Refusing it loudly beats a limit of
    // "negative seconds", which every comparison below would read as already exhausted.
    logger.warning(
        "time_limit={:g} is not a duration; ignoring it and running without a time limit",
        seconds);
  } else if (std::isfinite(seconds) && seconds < kNoWallClockLimit) {
    has_time_limit_ = true;
    time_limit_ = seconds;
  }

  const std::int64_t iterations = options.get_int("iteration_limit");
  if (iterations < -1) {
    logger.warning("iteration_limit={} is not a count; ignoring it (-1 means no limit)",
                   iterations);
  } else {
    iteration_limit_ = iterations;
  }

  const std::int64_t nodes = options.get_int("node_limit");
  if (nodes < -1) {
    logger.warning("node_limit={} is not a count; ignoring it (-1 means no limit)", nodes);
  } else {
    node_limit_ = nodes;
  }
}

bool ResourceLimits::time_exhausted(double elapsed) const noexcept {
  if (!has_time_limit_) return false;
  // Strictly greater, so a limit is not reported at the instant it is reached on a clock
  // whose resolution is coarser than the check; a zero budget is exhausted anyway, which is
  // the case the >= would have been for.
  return time_limit_ <= 0.0 || elapsed > time_limit_;
}

bool ResourceLimits::iterations_exhausted(std::int64_t done) const noexcept {
  return iteration_limit_ >= 0 && done >= iteration_limit_;
}

bool ResourceLimits::nodes_exhausted(std::int64_t done) const noexcept {
  return node_limit_ >= 0 && done >= node_limit_;
}

LimitReason ResourceLimits::exhausted(double elapsed, std::int64_t iterations,
                                      std::int64_t nodes) const noexcept {
  if (time_exhausted(elapsed)) return LimitReason::kTime;
  if (iterations_exhausted(iterations)) return LimitReason::kIterations;
  if (nodes_exhausted(nodes)) return LimitReason::kNodes;
  return LimitReason::kNone;
}

double ResourceLimits::remaining_seconds(double elapsed) const noexcept {
  if (!has_time_limit_) return kNoWallClockLimit;
  const double left = time_limit_ - elapsed;
  return left > 0.0 ? left : 0.0;
}

std::string ResourceLimits::describe(LimitReason reason, double elapsed,
                                     std::int64_t iterations, std::int64_t nodes) const {
  switch (reason) {
    case LimitReason::kNone: return {};
    case LimitReason::kInterrupt:
      return fmt::format("stopped by user interrupt after {:.2f}s, {} iterations, {} nodes",
                         elapsed, iterations, nodes);
    case LimitReason::kTime:
      return fmt::format(
          "stopped at the time limit of {:g}s after {:.2f}s, {} iterations, "
          "{} nodes",
          time_limit_, elapsed, iterations, nodes);
    case LimitReason::kIterations:
      return fmt::format("stopped at the iteration limit of {} after {} iterations, {:.2f}s",
                         iteration_limit_, iterations, elapsed);
    case LimitReason::kNodes:
      return fmt::format("stopped at the node limit of {} after {} nodes, {:.2f}s", node_limit_,
                         nodes, elapsed);
  }
  return {};
}

}  // namespace sankhya

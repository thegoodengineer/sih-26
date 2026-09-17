// SPDX-License-Identifier: Apache-2.0
// SANKHYA - one place that decides what a resource limit means (#289).
//
// THE PROBLEM THIS SOLVES. Every engine used to read time_limit, iteration_limit and
// node_limit for itself and decide on its own what they meant. Measured on main before this
// file existed, on bandm.mps:
//
//   time_limit=0        simplex/dual/PDHG stop at once; the INTERIOR POINT ignores it and
//                       returns optimal after 38 iterations, because its guard was
//                       `time_limit > 0.0` and zero read as "no limit"
//   iteration_limit=0   simplex and dual simplex perform ONE iteration, because both count
//                       the iteration and then compare; PDHG and the interior point
//                       perform none
//   node_limit=0        the MILP path returned numerical_error, because a tree that stopped
//                       before its first node has no incumbent, and the infinite objective
//                       that stands for "nothing found" was caught by the non-finite guard
//
// Three different answers to one option, and one of them reports a resource limit as a
// numerical failure, which is the single thing the issue says must never happen. The engines
// are not wrong individually; there was nowhere for them to agree.
//
// SEMANTICS, decided here and documented under "Resource limits" in docs/ARCHITECTURE.md:
//
//   time_limit        seconds of WALL CLOCK. The default (the no-limit sentinel) and any
//                     non-finite value mean no limit. ZERO MEANS ZERO SECONDS: the solve
//                     stops at its first safe point. Negative is a configuration error,
//                     refused with a warning and treated as no limit.
//   iteration_limit   -1 means no limit. N means AT MOST N iterations, N=0 included.
//   node_limit        -1 means no limit. N means AT MOST N nodes, N=0 included.
//
// PRECEDENCE when more than one is true at the same check, highest first:
//
//   user interrupt  >  time  >  iterations  >  nodes
//
// The interrupt wins because it is the only one a person is waiting on. Time beats the
// counters because it is the limit that protects a caller's own deadline. The order is
// fixed here so that two engines cannot disagree about which reason to report.
//
// SAFE POINTS. A limit is observed at a boundary the engine chooses - between simplex
// iterations, between PDHG iterations, between branch-and-bound nodes, and inside the
// interior point's linear algebra through the same predicate. It cannot interrupt a
// factorization or a kernel already running, so an overrun of one such step is expected and
// is NOT a violated limit.

#pragma once

#include <cstdint>
#include <string>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {

// LimitReason, to_string(LimitReason) and status_for() are declared in sankhya/model.hpp,
// beside SolveStatus and beside the Solution field that carries one. They are defined in
// resource_limits.cpp, with everything else that decides what a limit means.

/// The limits one solve runs under, read from the options once.
///
/// It holds no clock and counts nothing: the engine owns its counters and its timer, and
/// asks here whether what it has reached is past the limit. That keeps this testable without
/// sleeping, which is what the issue asks for, and keeps the hot loop free of anything
/// beyond a comparison.
class ResourceLimits {
 public:
  ResourceLimits() = default;

  /// Read the limits from the options. `logger` is where a refused configuration is
  /// reported; pass a logger with no sink to read them silently.
  ResourceLimits(const Options& options, Logger& logger);

  [[nodiscard]] bool has_time_limit() const noexcept { return has_time_limit_; }
  [[nodiscard]] double time_limit() const noexcept { return time_limit_; }
  [[nodiscard]] std::int64_t iteration_limit() const noexcept { return iteration_limit_; }
  [[nodiscard]] std::int64_t node_limit() const noexcept { return node_limit_; }

  /// True when `elapsed` seconds have used the whole budget. A zero budget is exhausted at
  /// the first check, which is what a caller asking for zero seconds asked for.
  [[nodiscard]] bool time_exhausted(double elapsed) const noexcept;

  /// True when `done` iterations have used the whole budget, i.e. no further iteration may
  /// start. Called at the TOP of an iteration, so a limit of N runs exactly N of them.
  [[nodiscard]] bool iterations_exhausted(std::int64_t done) const noexcept;

  /// The same for branch-and-bound nodes.
  [[nodiscard]] bool nodes_exhausted(std::int64_t done) const noexcept;

  /// The highest-precedence limit that is already exhausted, or kNone. Time beats the
  /// counters, so an engine checking several at one safe point reports them in the order
  /// this project documents rather than in the order its own loop happens to test them.
  /// A user interrupt outranks all of these and is the StopController's to report.
  [[nodiscard]] LimitReason exhausted(double elapsed, std::int64_t iterations,
                                      std::int64_t nodes) const noexcept;

  /// The single line a stopped solve puts in its message, e.g.
  /// "stopped at the iteration limit: 500 iterations of 500, 0.42s elapsed".
  [[nodiscard]] std::string describe(LimitReason reason, double elapsed,
                                     std::int64_t iterations, std::int64_t nodes) const;

  /// The time limit an engine should run under when `elapsed` seconds of the solve are
  /// already spent, so that presolve and a first pass cannot each be given the whole budget.
  /// Returns the no-limit sentinel when there is no limit to share.
  [[nodiscard]] double remaining_seconds(double elapsed) const noexcept;

 private:
  bool has_time_limit_ = false;
  double time_limit_ = 0.0;
  std::int64_t iteration_limit_ = -1;
  std::int64_t node_limit_ = -1;
};

}  // namespace sankhya

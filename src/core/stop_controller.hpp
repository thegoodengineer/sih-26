// SPDX-License-Identifier: Apache-2.0
// SANKHYA - unified stop condition checker.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#include "core/resource_limits.hpp"
#include "sankhya/model.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya {

/// Unified checker for time limits, user interruptions, and progress callbacks.
///
/// Designed to be called at safe points inside the solver engines. The callback's window
/// is kept by the SolveControl it checks (see there for why), so one is built per engine
/// call without the callback firing once per engine call.
///
/// What a limit MEANS is not decided here: ResourceLimits owns that, so every engine that
/// goes through this class agrees with every other one (#289). The precedence between a
/// user interrupt and the clock is the order of the tests below, and it is the order this
/// project documents.
class StopController {
 public:
  StopController(SolveControl* control, const Timer& timer, const ResourceLimits& limits)
      : control_(control), timer_(timer), limits_(limits) {}

  /// Check if the solver should stop.
  /// `get_progress` is a callable returning `Progress`, evaluated only when the callback is
  /// due.
  template <typename F>
  bool should_stop(F&& get_progress, SolveStatus* out_status) {
    LimitReason reason = LimitReason::kNone;
    const bool stop = should_stop(std::forward<F>(get_progress), &reason);
    if (stop) *out_status = status_for(reason);
    return stop;
  }

  /// The same check, reporting WHICH limit fired rather than the status that follows from
  /// it. An engine that has a counter of its own to name in the message wants this one.
  template <typename F>
  bool should_stop(F&& get_progress, LimitReason* out_reason) {
    if (control_ && control_->interruption_requested()) {
      *out_reason = LimitReason::kInterrupt;
      return true;
    }

    const double elapsed = timer_.elapsed_seconds();
    if (limits_.time_exhausted(elapsed)) {
      *out_reason = LimitReason::kTime;
      return true;
    }

    if (control_ && control_->progress_callback) {
      if (control_->callback_due()) {
        Progress p = get_progress();
        p.elapsed_seconds = elapsed;
        if (control_->progress_callback(p) != 0) {
          control_->interrupt();
          *out_reason = LimitReason::kInterrupt;
          return true;
        }
        if (control_->interruption_requested()) {
          *out_reason = LimitReason::kInterrupt;
          return true;
        }
      }
    }

    return false;
  }

  [[nodiscard]] const ResourceLimits& limits() const noexcept { return limits_; }

 private:
  SolveControl* control_;
  const Timer& timer_;
  ResourceLimits limits_;
};

}  // namespace sankhya

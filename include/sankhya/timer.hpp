// SPDX-License-Identifier: Apache-2.0
// SANKHYA - monotonic timing.
//
// steady_clock, never system_clock: a wall-clock adjustment mid-solve must not be able to
// produce a negative elapsed time and trip a time limit. Every reported duration in the
// project comes from this type, so that benchmark CSVs are measured one way only.
#pragma once

#include <chrono>

namespace sankhya {

class Timer {
 public:
  Timer() : start_(Clock::now()) {}

  void reset() noexcept { start_ = Clock::now(); }

  [[nodiscard]] double elapsed_seconds() const noexcept {
    const std::chrono::duration<double> d = Clock::now() - start_;
    return d.count();
  }

  [[nodiscard]] double elapsed_milliseconds() const noexcept {
    return elapsed_seconds() * 1000.0;
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

/// Adds its lifetime to a double on destruction. Used to accumulate per-component timings
/// (factorization, pricing, ratio test) that the numerics report in Phase 9 breaks down.
class ScopedTimer {
 public:
  explicit ScopedTimer(double& accumulator) noexcept : accumulator_(accumulator) {}
  ~ScopedTimer() { accumulator_ += timer_.elapsed_seconds(); }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;
  ScopedTimer(ScopedTimer&&) = delete;
  ScopedTimer& operator=(ScopedTimer&&) = delete;

 private:
  double& accumulator_;
  Timer timer_;
};

}  // namespace sankhya

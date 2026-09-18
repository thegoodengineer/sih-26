// SPDX-License-Identifier: Apache-2.0
// SANKHYA - where a solve's time goes (#285).
//
// A tree of named regions - solve > presolve, solve > engine > factorize - each with its
// inclusive wall-clock time, the time spent in it outside its children (exclusive), and how
// many times it was entered; plus named counters (iterations, nodes, refactorizations). It
// holds no solver logic and measures nothing a Timer does not already measure: it is where the
// measurements are put so that one report can show all of them.
//
// OFF COSTS A POINTER TEST. A ProfileScope built against a null profiler, or one whose mode is
// below what the scope asks for, reads no clock and touches no memory. The engines are
// instrumented unconditionally and pay for it only when `--option profile=basic|detailed` is
// set; the overhead with it off is measured in the PR for #285, not assumed.
//
// WHAT IS NOT HERE. No GPU timing: the CUDA backend is not on main, and a CUDA event timer
// with nothing to time is a feature in name only. No memory statistics: peak resident memory
// is not portably measurable from this binary (the same reason #289 has no memory limit).
//
// THREADS. A Profiler is owned by one solve and written by one thread; scopes nest by a stack
// it keeps. A parallel search gives each worker its own and folds them together with merge(),
// which adds regions matched by their PATH from the root, so the same work done on four
// threads reports as one region with four threads' worth of calls and time.

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sankhya {

enum class ProfileMode : std::uint8_t {
  kOff,       ///< nothing recorded
  kBasic,     ///< the phases of a solve and the counters an engine already keeps
  kDetailed,  ///< also the sub-phases inside an engine's iteration loop
};

/// "off", "basic" or "detailed"; false for anything else.
[[nodiscard]] bool parse_profile_mode(std::string_view text, ProfileMode* out) noexcept;
[[nodiscard]] const char* to_string(ProfileMode mode) noexcept;

class Profiler {
 public:
  struct Region {
    std::string name;
    int parent = -1;  ///< index into regions(); -1 for a top-level region
    double inclusive_seconds = 0.0;
    std::int64_t calls = 0;
    std::vector<int> children;
  };
  struct Counter {
    std::string name;
    std::int64_t value = 0;
  };

  explicit Profiler(ProfileMode mode = ProfileMode::kOff) : mode_(mode) {}

  [[nodiscard]] ProfileMode mode() const noexcept { return mode_; }
  [[nodiscard]] bool records(ProfileMode needed) const noexcept {
    return mode_ != ProfileMode::kOff && static_cast<int>(mode_) >= static_cast<int>(needed);
  }

  /// Open `name` under the region currently open (or at the top), creating it the first time.
  /// Returns its index; pass it to leave().
  int enter(std::string_view name);
  /// Close the innermost region, which must be `region`, adding `seconds` to it.
  void leave(int region, double seconds);

  /// Add a region whose time was measured elsewhere - an engine's own phase accumulators -
  /// as a child of the region currently open. The way existing timers report here without
  /// the engine timing the same thing twice.
  void record(std::string_view name, double seconds, std::int64_t calls = 1);

  void count(std::string_view name, std::int64_t amount = 1);

  [[nodiscard]] const std::vector<Region>& regions() const noexcept { return regions_; }
  [[nodiscard]] const std::vector<Counter>& counters() const noexcept { return counters_; }

  /// Inclusive time less the children's inclusive time. Never below zero: a child measured
  /// by a coarser clock than its parent can exceed it by a tick.
  [[nodiscard]] double exclusive_seconds(int region) const;

  /// Fold another profiler's regions and counters into this one, matching regions by path.
  void merge(const Profiler& other);

  /// The indented table the log prints.
  [[nodiscard]] std::string format_text() const;
  /// The same data as JSON: {"mode", "regions": [{path, parent_path, inclusive_seconds,
  /// exclusive_seconds, calls}], "counters": {name: value}}.
  [[nodiscard]] std::string format_json() const;

 private:
  int child_named(int parent, std::string_view name);
  [[nodiscard]] std::string path(int region) const;

  ProfileMode mode_ = ProfileMode::kOff;
  std::vector<Region> regions_;
  std::vector<int> roots_;
  std::vector<Counter> counters_;
  std::vector<int> open_;
};

/// Times its own lifetime into a region, when the profiler asks for it; otherwise nothing.
class ProfileScope {
 public:
  ProfileScope(Profiler* profiler, std::string_view name,
               ProfileMode needed = ProfileMode::kBasic)
      : profiler_(profiler != nullptr && profiler->records(needed) ? profiler : nullptr) {
    if (profiler_ != nullptr) {
      region_ = profiler_->enter(name);
      start_ = Clock::now();
    }
  }
  ~ProfileScope() {
    if (profiler_ != nullptr) {
      const std::chrono::duration<double> spent = Clock::now() - start_;
      profiler_->leave(region_, spent.count());
    }
  }
  ProfileScope(const ProfileScope&) = delete;
  ProfileScope& operator=(const ProfileScope&) = delete;

 private:
  // A bare time point rather than a Timer: a Timer reads the clock when it is constructed,
  // and a disabled scope must not read it at all.
  using Clock = std::chrono::steady_clock;
  Profiler* profiler_;
  int region_ = -1;
  Clock::time_point start_{};
};

}  // namespace sankhya

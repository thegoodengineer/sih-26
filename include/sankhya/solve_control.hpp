// SPDX-License-Identifier: Apache-2.0
// SANKHYA - progress reporting and cooperative interruption (#223).
//
// A solve on a real planning model runs for minutes. Before this, the only way to stop one
// was to kill the process, which throws away the incumbent branch-and-bound was carrying.
// This is the generalisation of the time-limit deadline that already reaches inside the LDL
// ordering and factorization (#197, include/sankhya/ldl.hpp's ShouldStop): the same idea,
// widened to carry a progress snapshot out to the caller and to accept a stop request FROM
// the caller rather than only from a clock.
//
// TWO INDEPENDENT WAYS TO STOP, sharing one flag:
//   1. A registered callback returns non-zero from inside the solve's own thread.
//   2. Another thread calls interrupt() while the solve is running.
// Both end at the same atomic flag, so every engine has exactly one thing to check.
//
// interrupt() ITSELF IS NOT THROTTLED - it is a single atomic store, and is_interrupted()
// a single atomic load, checked unconditionally on every poll(). What IS throttled is the
// CALLBACK: building a Progress snapshot and invoking user code every iteration would slow
// exactly the tight loops (simplex pivots, PDHG iterations) that #197 was written to keep
// fast, so the callback fires at a bounded rate - by iteration/node count and by wall time -
// and a poll() between callback firings costs one atomic load plus a few integer
// comparisons.
//
// THIS DOES NOT MOVE THE ALGORITHM'S PATH. Exactly like the time-limit deadline it
// generalises, a SolveControl decides only whether an unfinished solve keeps going; a solve
// that runs to completion without ever being asked to stop does the same arithmetic in the
// same order regardless of whether a SolveControl was attached. CLAUDE.md's determinism rule
// (see #172's history in ldl.hpp) is about the PATH a running solve takes, not about whether
// it may be interrupted.
#pragma once

#include <atomic>
#include <functional>
#include <mutex>

#include "sankhya/types.hpp"

namespace sankhya {

/// Which part of a solve produced a Progress snapshot. Mirrors the phases an industrial
/// solver log already names (presolve / simplex or interior point / branch-and-bound tree).
enum class SolvePhase : std::uint8_t { kPresolve, kLp, kTree };

[[nodiscard]] const char* to_string(SolvePhase phase) noexcept;

/// One progress snapshot, handed to the callback. Fields that do not apply to the engine
/// reporting it are left at their default (0.0 / 0) rather than omitted, so a callback
/// written against one engine does not have to special-case every other one - the C API and
/// Python surfaces mirror this shape exactly (sankhya_progress, Progress).
struct Progress {
  SolvePhase phase = SolvePhase::kLp;

  /// Simplex/IPM/PDHG iterations. Zero in the tree phase, where `nodes` is what advances.
  Count iterations = 0;
  /// Branch-and-bound nodes explored. Zero outside the tree phase.
  Count nodes = 0;
  /// Nodes still open (queued, not yet explored). Zero outside the tree phase.
  Count open_nodes = 0;

  /// Current objective: the LP/IPM/PDHG iterate's objective, or the tree's incumbent when
  /// one has been found. Meaningless (left at 0.0) before any point exists.
  double objective = 0.0;
  /// Best proven bound: equal to `objective` for a converging LP, the tree's global dual
  /// bound for the MILP phase.
  double best_bound = 0.0;
  /// Relative gap (objective - best_bound) / |objective|, tree phase only; 0 elsewhere.
  double gap = 0.0;
  /// Wall-clock seconds since the solve that reports this progress started.
  double elapsed_seconds = 0.0;
};

/// Returns true to ask the solve to stop. Called from the solving thread, at the bounded
/// rate described above - never called concurrently with itself.
using ProgressCallback = std::function<bool(const Progress&)>;

/// Shared between a caller and the engine currently solving: the caller's handle to ask a
/// running solve to report progress and, optionally, to stop.
///
/// THREADING. `interrupt()` and `is_interrupted()` are safe to call from any thread at any
/// time - that is the whole point, since the thread running the solve is by definition busy.
/// `set_callback()` and `set_throttle()` are NOT meant to race a concurrent poll() in
/// practice (there is no use case for changing the callback mid-solve), but are made
/// thread-safe against poll() anyway with a small mutex, cheap next to the arithmetic every
/// engine here does per iteration, so an accidental race is a stale callback rather than a
/// data race.
class SolveControl {
 public:
  SolveControl() = default;

  /// Install (or, with an empty std::function, remove) the progress callback. Not owned;
  /// the caller's callable must outlive every poll() call made against this SolveControl.
  void set_callback(ProgressCallback callback) {
    const std::lock_guard<std::mutex> lock(mutex_);
    callback_ = std::move(callback);
  }

  /// Ask a running solve to stop. Thread-safe, and NOT throttled: the next poll() on any
  /// engine thread sees it immediately. Idempotent.
  void interrupt() noexcept { interrupted_.store(true, std::memory_order_relaxed); }

  [[nodiscard]] bool is_interrupted() const noexcept {
    return interrupted_.load(std::memory_order_relaxed);
  }

  /// Override the default rate limit. `every_iterations` and `every_nodes` are the minimum
  /// count of each metric between two callback firings (values <= 0 are clamped to 1);
  /// `min_interval_seconds` is the wall-clock floor between firings regardless of count -
  /// the "at most every 100 ms" from #223. Exists mainly so a test can set
  /// `min_interval_seconds` to 0 and get a rate governed purely by the (deterministic)
  /// iteration/node count, rather than by how fast the machine running it happens to be.
  void set_throttle(Count every_iterations, Count every_nodes,
                     double min_interval_seconds) noexcept {
    every_iterations_ = every_iterations > 0 ? every_iterations : 1;
    every_nodes_ = every_nodes > 0 ? every_nodes : 1;
    min_interval_seconds_ = min_interval_seconds;
  }

  /// Called by an engine wherever it already checks the time-limit deadline. Returns true
  /// exactly when the solve should stop: an external interrupt() (checked unconditionally,
  /// first, so it is never delayed by the callback throttle below) or a callback that just
  /// returned non-zero.
  ///
  /// The callback itself only actually runs when the throttle says it is due; every other
  /// call is one atomic load plus a few comparisons, on the same order of cost as the
  /// time-limit check it sits beside.
  [[nodiscard]] bool poll(const Progress& progress) {
    if (is_interrupted()) return true;

    ProgressCallback callback_copy;
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!callback_) return false;
      const bool tree = progress.phase == SolvePhase::kTree;
      const Count metric = tree ? progress.nodes : progress.iterations;
      const Count every = tree ? every_nodes_ : every_iterations_;
      const bool count_due = !fired_once_ || (metric - last_metric_) >= every;
      const bool time_due =
          !fired_once_ || (progress.elapsed_seconds - last_call_seconds_) >= min_interval_seconds_;
      if (!count_due || !time_due) return false;
      fired_once_ = true;
      last_metric_ = metric;
      last_call_seconds_ = progress.elapsed_seconds;
      callback_copy = callback_;
    }
    const bool stop = callback_copy(progress);
    if (stop) interrupt();
    return stop;
  }

  /// Clear the interrupt flag and the throttle history so a handle can be reused for a
  /// fresh solve. The registered callback and the throttle configuration are left alone -
  /// only the STATE of a previous solve is cleared. A caller that owns one SolveControl per
  /// model (the C API's sankhya_model does, so a callback survives across solves) calls
  /// this at the start of each solve; a caller that constructs a fresh SolveControl per
  /// solve (the CLI does) never needs it.
  void reset() noexcept {
    interrupted_.store(false, std::memory_order_relaxed);
    const std::lock_guard<std::mutex> lock(mutex_);
    fired_once_ = false;
    last_metric_ = 0;
    last_call_seconds_ = 0.0;
  }

  SolveControl(const SolveControl&) = delete;
  SolveControl& operator=(const SolveControl&) = delete;

 private:
  std::atomic<bool> interrupted_{false};

  std::mutex mutex_;  ///< guards everything below
  ProgressCallback callback_;
  Count every_iterations_ = 200;
  Count every_nodes_ = 1;
  double min_interval_seconds_ = 0.1;
  bool fired_once_ = false;
  Count last_metric_ = 0;
  double last_call_seconds_ = 0.0;
};

}  // namespace sankhya

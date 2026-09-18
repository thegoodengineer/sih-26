// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

namespace sankhya {

/// Live metrics reported during a solve.
///
/// Reported approximately every 100ms to the progress callback. The `phase` field
/// identifies which part of the solve is running.
struct Progress {
  enum class Phase : std::uint8_t { kPresolve, kLp, kTree };

  Phase phase = Phase::kLp;
  int64_t iterations = 0;
  int64_t nodes = 0;
  double objective = 0.0;
  double best_bound = 0.0;
  double gap = 0.0;
  double elapsed_seconds = 0.0;
  int64_t open_nodes = 0;
};

/// A callback invoked periodically by the solver.
/// Returning a non-zero value requests an immediate interrupt of the solve.
using ProgressCallback = std::function<int(const Progress&)>;

/// Solve-scoped control object.
///
/// This object is passed down to all solver engines participating in a single solve.
/// It holds the interruption state and the progress callback. Since a solve may scale
/// or transform the model (producing a copy of `Model`), holding this state here ensures
/// all transformations share the same interruption context and `Model` itself remains
/// copyable and movable.
///
/// THE CALLBACK WINDOW LIVES HERE TOO, not in the per-engine checker. Branch-and-bound
/// runs one simplex per node, and each engine call builds its own StopController; a window
/// kept there fired the callback on every node's first check - once per node on a MILP of
/// ten thousand nodes, each a Python call - while this header promised every 100 ms. The
/// window is measured on a steady clock rather than an engine's Timer because every engine
/// has its own.
/// Basis status of a column or row; the definition is in model.hpp, and this opaque
/// declaration is enough for the vectors below (a scoped enum with a fixed underlying type
/// is a complete type once declared).
enum class BasisStatus : std::uint8_t;

class SolveControl {
 public:
  /// A STARTING BASIS (#218): the statuses a previous Solution reported (`col_status`,
  /// `row_status`), for a re-solve after the model was edited. The simplex engines honour
  /// it - the dual simplex when bounds or right-hand sides moved (the old basis is still
  /// dual feasible), the primal simplex (`algorithm=simplex`) when costs moved (still primal
  /// feasible) - and finish in a handful of pivots where a cold solve takes thousands. It
  /// must describe a basis of THIS model: one status per column and per row and exactly
  /// `num_rows()` of them basic, which is what a Solution's statuses are after #341. Presolve
  /// is bypassed on a warm solve, since the statuses name the caller's rows and columns, and
  /// the message says so. A basis that does not seed (wrong lengths, wrong count, singular
  /// beyond repair) is reported and the solve runs cold; other engines ignore it with a note.
  std::vector<BasisStatus> start_col_status;
  std::vector<BasisStatus> start_row_status;
  [[nodiscard]] bool has_starting_basis() const noexcept {
    return !start_col_status.empty() || !start_row_status.empty();
  }

  /// How often the progress callback is invoked, at most: the first check of a solve
  /// always calls it, and after that one call per interval.
  static constexpr std::chrono::milliseconds kCallbackInterval{100};

  /// The user-provided progress callback, invoked approximately every 100ms.
  ProgressCallback progress_callback;

  /// Is a progress callback due now? Records the call when it says yes.
  [[nodiscard]] bool callback_due() noexcept {
    const auto now = std::chrono::steady_clock::now();
    if (callback_seen_ && now - last_callback_ < kCallbackInterval) return false;
    callback_seen_ = true;
    last_callback_ = now;
    return true;
  }

  /// Request an immediate interruption of the solve.
  /// This is safe to call from another thread or a signal handler.
  void interrupt() noexcept { interrupt_requested_.store(true, std::memory_order_relaxed); }

  /// Has an interruption been requested?
  [[nodiscard]] bool interruption_requested() const noexcept {
    return interrupt_requested_.load(std::memory_order_relaxed);
  }

  /// Reset the interruption flag and the callback window. Call before starting a new solve.
  void reset() noexcept {
    interrupt_requested_.store(false, std::memory_order_relaxed);
    callback_seen_ = false;
  }

 private:
  std::atomic<bool> interrupt_requested_{false};
  bool callback_seen_ = false;
  std::chrono::steady_clock::time_point last_callback_{};
};

}  // namespace sankhya

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - what can be said about a model before solving it (#294).
//
// WHAT THIS IS FOR. A planner who has just been handed an MPS file wants to know what is in
// it, whether it is the shape of model this solver is good at, and whether anything about it
// will hurt - all before committing to a solve that may take the rest of the afternoon. Every
// number below is read off the model itself in one pass or two; nothing here solves anything,
// and nothing here is a prediction of how long a solve will take.
//
// WHAT IT IS NOT. Not a second copy of the solver's judgement. The classification comes from
// the same properties `solve()` dispatches on, the convexity verdict from the same test the
// QP engine runs, the reduction preview from presolve itself. A diagnostic that disagreed
// with the solver would be worse than no diagnostic, so nothing here re-derives what another
// part of the project already decides.
//
// WHAT IT REFUSES TO SAY. There is no estimate of solve time, no predicted node count, no
// memory figure beyond the arithmetic of what the model itself occupies. Those are the
// numbers a reader would most like and the ones this project has no honest way to produce
// before the solve; inventing them would make every other line here less trustworthy.
#pragma once

#include <string>
#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::diagnose {

/// How alarming the spread of magnitudes in the model is.
enum class Risk {
  kLow,
  kMedium,
  kHigh,
};

[[nodiscard]] const char* to_string(Risk risk) noexcept;

/// Everything `sankhya diagnose` reports, computed from the model alone.
struct Diagnosis {
  // ---- identity and size ----
  std::string name;
  std::string source_path;
  bool maximize = false;
  Index rows = 0;
  Index columns = 0;
  Index nonzeros = 0;
  Index objective_nonzeros = 0;
  double objective_offset = 0.0;

  // ---- columns ----
  Index continuous_columns = 0;
  Index integer_columns = 0;
  Index binary_columns = 0;  ///< integer with bounds inside [0, 1]
  Index fixed_columns = 0;
  Index free_columns = 0;
  Index boxed_columns = 0;
  Index columns_without_entries = 0;

  // ---- rows ----
  Index equality_rows = 0;
  Index less_than_rows = 0;
  Index greater_than_rows = 0;
  Index range_rows = 0;
  Index free_rows = 0;
  Index empty_rows = 0;
  Index singleton_rows = 0;

  // ---- sparsity ----
  double density_percent = 0.0;
  double average_nonzeros_per_row = 0.0;
  double average_nonzeros_per_column = 0.0;
  Index densest_row_nonzeros = 0;
  Index densest_column_nonzeros = 0;

  // ---- magnitudes ----
  double smallest_coefficient = 0.0;  ///< smallest |a_ij| over the nonzeros, 0 when empty
  double largest_coefficient = 0.0;
  double coefficient_ratio = 0.0;  ///< largest / smallest, 0 when there are no entries
  double largest_bound = 0.0;      ///< largest finite |bound| over rows and columns
  double largest_objective_coefficient = 0.0;
  Risk scaling_risk = Risk::kLow;
  std::string scaling_note;

  // ---- class ----
  std::string problem_class;  ///< LP, MILP, QP, MIQP
  /// For a model with a quadratic objective: the convexity test's own verdict and detail.
  std::string convexity;
  std::string convexity_detail;

  // ---- what presolve would do ----
  bool presolve_ran = false;
  Index presolve_rows_removed = 0;
  Index presolve_columns_removed = 0;
  bool presolve_proved_infeasible = false;
  std::string presolve_message;

  // ---- guidance ----
  std::string recommended_engine;
  std::string engine_reason;
  std::string gpu_note;
  /// Things worth knowing before committing to the solve; empty when there are none.
  std::vector<std::string> warnings;

  /// Bytes the constraint matrix and Hessian occupy in this build's storage. Not a memory
  /// estimate for the solve: a factorization's fill is not known until it is computed.
  std::size_t model_bytes = 0;
};

/// Analyse `model`. Runs presolve when `options` has it enabled, because what presolve would
/// remove is one of the more useful things to know before solving; nothing else here solves.
[[nodiscard]] Diagnosis analyse(const Model& model, const Options& options, Logger& logger);

/// The human-readable report, as `sankhya diagnose` prints it.
[[nodiscard]] std::string format_text(const Diagnosis& diagnosis);

/// The same content as JSON, for a script.
[[nodiscard]] std::string format_json(const Diagnosis& diagnosis);

}  // namespace sankhya::diagnose

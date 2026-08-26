// SPDX-License-Identifier: Apache-2.0
// SANKHYA - presolve reductions and the postsolve stack.
//
// References, written from the literature:
//   Brearley, Mitra & Williams, "Analysis of mathematical programming problems prior to
//     applying the simplex algorithm", Math. Programming 8 (1975) - the original treatment
//     of empty/singleton rows and columns and of row activity bounds
//   Andersen & Andersen, "Presolving in linear programming", Math. Programming 71 (1995) -
//     the reduction set and, more importantly, the postsolve discipline
//   Achterberg et al., "Presolve reductions in mixed integer programming", INFORMS J.
//     Computing 32(2), 2020 - integrality-aware bound rounding
//
// WHAT PRESOLVE IS FOR. Industrial models are written by modelling systems, not by hand, and
// they arrive full of rows that say nothing: variables already fixed by data, constraints
// that cannot bind given the bounds, rows with a single entry that are really just a bound.
// Solving those is wasted work. PS26119 asks for models with "thousands to millions of
// variables"; on those, what is removed before the simplex starts matters more than the
// pivot rule.
//
// THE DANGEROUS HALF IS POSTSOLVE. A reduction that is slightly wrong does not crash - it
// returns a confident, feasible-looking answer to a DIFFERENT problem. That is the same
// failure class CLAUDE.md names as the worst available outcome, alongside reporting a MILP's
// fractional relaxation as optimal. Every reduction here therefore pushes a record onto a
// stack, and postsolve replays that stack in reverse to rebuild a solution to the ORIGINAL
// model. The round-trip is asserted against the exact rational oracle, not assumed.
#pragma once

#include <string>
#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::presolve {

/// What a reduction did, kept so postsolve can undo it.
///
/// One flat struct rather than a variant hierarchy: the set is small, closed, and every
/// member needs the same three or four fields. A reader can see the whole vocabulary of the
/// transformation in one place, which matters more here than type-level tidiness.
struct Record {
  enum class Kind {
    kEmptyRow,      ///< a row with no entries; activity is 0 and its dual is 0
    kRedundantRow,  ///< bounds cannot bind given the column bounds; dual is 0
    kFixedColumn,   ///< lower == upper; the value is known and folded into the row bounds
    kEmptyColumn,  ///< no entries and a finite best value; parked at the bound the cost prefers
    kSingletonRow,  ///< one entry; became a bound on that column, the row is now implied
    kForcingRow,    ///< the row bound is only reachable with every variable at one bound
  };

  Kind kind = Kind::kEmptyRow;
  Index index = -1;          ///< original row or column index this record is about
  double value = 0.0;        ///< the value a removed column takes
  double coefficient = 0.0;  ///< the single entry, for a singleton row
  double row_lower = 0.0;    ///< original bounds, kept so postsolve can price the row
  double row_upper = 0.0;
  Index column = -1;  ///< the column a singleton row constrained
};

/// The reduced problem plus everything needed to get back.
struct Result {
  Model model;                  ///< the reduced model, to hand to an engine
  std::vector<Record> records;  ///< applied in order; postsolve replays in reverse
  std::vector<Index>
      col_to_original;  ///< reduced column j came from original col_to_original[j]
  std::vector<Index> row_to_original;

  Index original_rows = 0;
  Index original_cols = 0;
  Index original_nonzeros = 0;

  /// Set when a reduction proves the model cannot have a solution. No engine is run in that
  /// case: an empty row whose bounds exclude zero is infeasible on its own evidence, and
  /// saying so is both faster and more honest than handing the simplex a model we already
  /// know the answer to.
  bool proved_infeasible = false;

  std::string message;

  [[nodiscard]] Index rows_removed() const { return original_rows - model.num_rows(); }
  [[nodiscard]] Index cols_removed() const { return original_cols - model.num_cols(); }
};

/// Apply reductions until nothing more moves.
///
/// `model` is left untouched; the reduced copy is in the result. Integrality is respected:
/// a bound derived for an integer column is rounded inward, never outward, because a bound
/// that excludes a feasible integer point silently removes the optimum.
[[nodiscard]] Result presolve(const Model& model, const Options& options, Logger& logger);

/// Rebuild a solution to the ORIGINAL model from one for the reduced model.
///
/// The returned Solution is measured against the original model by the caller, so a
/// postsolve bug shows up as a feasibility violation on a model the engine never saw rather
/// than as a plausible number.
[[nodiscard]] Solution postsolve(const Result& result, const Model& original,
                                 const Solution& reduced);

}  // namespace sankhya::presolve

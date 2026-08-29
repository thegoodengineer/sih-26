// SPDX-License-Identifier: Apache-2.0
// SANKHYA - cutting planes for the branch and bound (#23).
//
// THE FAILURE MODE THIS FILE MUST NOT HAVE. A cut that is very slightly invalid removes the
// optimum, and the search then PROVES that the second-best answer is optimal: status
// `optimal`, point integral and feasible, bound equal to objective. Nothing about the output
// looks wrong, and no test that only checks self-consistency can see it.
//
// So every family added here carries two obligations, both in tests/unit/test_cuts.cpp:
//
//   1. a validity test against the EXACT optimum from the rational oracle, checked in exact
//      arithmetic, asserting the cut does not separate it;
//   2. a negative control - a deliberately invalid cut the same harness must reject - because
//      a validity test that has never failed is not evidence that it can fail.
//
// #23 carries the design notes for the Gomory mixed-integer family, which needs the simplex
// tableau and is not here yet.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

/// How many row bounds `tighten_integral_rows` moved.
struct RowTightening {
  Count rows_tightened = 0;
  Count bounds_moved = 0;  ///< a range row can have both sides moved
};

/// Round the bounds of rows whose activity must be integral.
///
/// When every column with a nonzero entry in a row is an INTEGER column and every one of
/// those coefficients is itself an integer, the row activity a'x is an integer at every
/// integer-feasible point. A bound of 7.3 therefore cannot be met more tightly than 8, and
/// the row can be restated as
///
///     ceil(lower) <= a'x <= floor(upper)
///
/// without removing a single integer-feasible point. This is the rank-1 Chvatal-Gomory cut
/// with unit multiplier, in the form that costs nothing: it tightens a row in place rather
/// than adding one, so the matrix does not grow and no basis gets larger.
///
/// It is deliberately the first family implemented. Its validity argument is one sentence
/// and does not depend on the tableau, the basis, or which bound a nonbasic column sits at -
/// which is exactly where the Gomory family's difficulty lives. That makes it the right cut
/// to build the validity harness AROUND, so that harness exists and is proven to work before
/// anything subtler is attempted.
///
/// Modifies `model` in place. Safe to call on an LP: a model with no integer columns has no
/// qualifying row and nothing happens.
RowTightening tighten_integral_rows(Model* model, Logger& logger);

}  // namespace sankhya::mip

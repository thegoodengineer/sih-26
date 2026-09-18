// SPDX-License-Identifier: Apache-2.0
// SANKHYA - clique cuts and {0,1/2}-Chvatal-Gomory cuts (#358).
//
// WHY THESE TWO. #355 measured why MIR does nothing on opt1217 and rlp1: their rows are
// pure-integer with unit coefficients and an integer right-hand side, so every MIR divisor
// leaves nothing to round, and every aggregation needs a bound the rows' signs never supply.
// That structure - covering, packing, partitioning - is outside what MIR separates by
// construction. These two families are the ones built for it.
//
//   CLIQUE (Padberg, "On the facial structure of set packing polyhedra", Math. Programming 5,
//   1973; Atamturk, Nemhauser and Savelsbergh, "Conflict graphs in solving integer
//   programming problems", European J. Oper. Res. 121, 2000). Two binaries CONFLICT when a
//   row forbids both being 1 - their coefficients alone exceed what the row allows once every
//   other column sits at its most favourable bound. A set of pairwise-conflicting binaries is
//   a clique, and at most one of them can be 1: sum over the clique <= 1. Separated greedily
//   at the LP point (the exact problem is maximum-weight clique, NP-hard) and extended to a
//   maximal clique, which only strengthens it.
//
//   {0,1/2}-CHVATAL-GOMORY (Caprara and Fischetti, "{0,1/2}-Chvatal-Gomory cuts", Math.
//   Programming 74, 1996). Rows a_i x <= b_i with integer data over non-negative integer x,
//   combined with multipliers 1/2 and rounded down: floor(sum a_i / 2) x <= floor(sum b_i /
//   2). Valid because floor(u'A) x <= u'A x for x >= 0 and the left side is an integer. The
//   parity argument is exactly what a unit-coefficient covering row needs. Separation is
//   NP-hard too; single rows and pairs of rows sharing a column are tried, which is the
//   bounded neighbourhood the PR states.
//
// VALIDITY IS BY CONSTRUCTION AND CHECKED BY ENUMERATION. tests/unit/test_combinatorial_cuts
// enumerates every integer point of small random models and asserts no generated cut removes
// any of them - the obligation cuts.hpp sets for every family - plus a negative control.

#pragma once

#include <vector>

#include "cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

struct CombinatorialCutStats {
  int conflict_edges = 0;  ///< pairs found in conflict (capped, see the definition)
  bool conflict_graph_capped = false;
  int candidate_row_sets = 0;  ///< row sets tried for {0,1/2}
};

/// Clique cuts violated at `solution.col_value`, from the conflict graph of the binaries.
/// `col_lower` / `col_upper` are the bounds the conflicts are derived under: the GLOBAL ones,
/// so a cut is valid at every node.
[[nodiscard]] std::vector<Cut> generate_clique_cuts(const Model& model,
                                                    const Solution& solution,
                                                    const std::vector<double>& col_lower,
                                                    const std::vector<double>& col_upper,
                                                    CombinatorialCutStats* stats = nullptr);

/// {0,1/2}-CG cuts violated at `solution.col_value`, from the pure-integer rows with integer
/// data, one row or two rows sharing a column at a time.
[[nodiscard]] std::vector<Cut> generate_zero_half_cuts(const Model& model,
                                                       const Solution& solution,
                                                       const std::vector<double>& col_lower,
                                                       const std::vector<double>& col_upper,
                                                       CombinatorialCutStats* stats = nullptr);

}  // namespace sankhya::mip

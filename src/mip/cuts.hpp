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
// Two families live here, both generated once at the root and both OFF by default
// (`enable_root_cuts`; the measurement behind the default is in the option's description):
//
//   - lifted knapsack cover cuts, on rows of the form sum(a_j x_j) <= b over binaries, with
//     exact sequential lifting through a 0/1 knapsack dynamic programme;
//   - Gomory mixed-integer cuts, from the tableau row of each fractional basic integer
//     column, reconstructed from the final basis of the root LP.
//
// References, written from the literature:
//   Gomory, "An algorithm for the mixed integer problem", RAND RM-2597, 1960 - the cut
//   Balas, Ceria, Cornuejols & Natraj, "Gomory cuts revisited", Oper. Res. Letters 19, 1996
//     - the cut as a practical root-node tool, and the numerical care it needs
//   Marchand & Wolsey, "Aggregation and mixed integer rounding to solve MIPs", Oper. Res.
//     49(3), 2001 - the mixed-integer-rounding view of the same inequality
//   Balas, "Facets of the knapsack polytope", Math. Programming 8, 1975; Wolsey, "Faces for
//     a linear inequality in 0-1 variables", Math. Programming 8, 1975 - cover inequalities
//     and their lifting
//   Zemel, "Easily computable facets of the knapsack polytope", Math. Oper. Res. 14, 1989 -
//     sequential lifting
//   Crowder, Johnson & Padberg, "Solving large-scale zero-one linear programming problems",
//     Oper. Res. 31(5), 1983; Gu, Nemhauser & Savelsbergh, "Lifted cover inequalities for
//     0-1 integer programs: computation", INFORMS J. Computing 10(4), 1998 - covers in
//     practice
//   Chvatal, "Edmonds polytopes and a hierarchy of combinatorial problems", Discrete Math.
//     4, 1973 - rounding an integral row's bound (tighten_integral_rows)
//   Achterberg, "Constraint Integer Programming" (thesis, 2007), ch. 8 - keeping a cut pool
//     small: density, coefficient range and violation (filter_and_deduplicate_cuts)
#pragma once

#include <optional>
#include <vector>

#include "la/lu.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

/// How many row bounds `tighten_integral_rows` moved.
struct RowTightening {
  Count rows_tightened = 0;
  Count bounds_moved = 0;  ///< a range row can have both sides moved
};

// =========================================================================================
// Basis Reconstruction and Tableau Generation (Stage 2)
// =========================================================================================

namespace detail {
/// Reconstructed tableau row for testing, so the infrastructure stays strictly internal.
struct ReconstructedTableauRow {
  Index basis_row = -1;
  bool basic_is_structural = false;
  Index basic_index = -1;
  double rhs = 0.0;
  std::vector<double> structural_coefs;
  std::vector<double> logical_coefs;
  std::vector<BasisStatus> structural_status;
  std::vector<BasisStatus> logical_status;
};

/// Try to reconstruct the basis and extract one tableau row. Returns std::nullopt if the
/// statuses are invalid or the basis is singular.
[[nodiscard]] std::optional<ReconstructedTableauRow> get_tableau_for_testing(
    const Model& model, const Solution& solution, Index basis_row);
}  // namespace detail

// =========================================================================================
// Knapsack cover cuts with exact sequential lifting (#23)
// =========================================================================================

/// The result of knapsack cover generation for a single model row.
///
/// The lifted cover inequality over the ORIGINAL model columns is:
///
///     sum(coeff[k] * x[col_index[k]]) <= rhs
///
/// where coeff[k] is the final (base or lifted) coefficient for column col_index[k].
///
/// Invariants guaranteed by the generator:
///   - col_index is sorted ascending (deterministic iteration order).
///   - Every coeff[k] >= 0.
///   - rhs == |C| - 1 where |C| is the cover cardinality (before lifting assigns
///     new coefficients to non-cover variables).
///   - The inequality is valid: no integer-feasible point of the original supported
///     row is separated by it.
struct KnapsackCoverCut {
  std::vector<Index> col_index;  ///< original model column indices, ascending
  std::vector<double> coeff;     ///< per-column coefficient (>= 0)
  double rhs = 0.0;              ///< right-hand side (|C| - 1)
};

/// Try to generate a lifted knapsack cover cut from row `row` of `model`.
///
/// SUPPORTED CLASS: the generator accepts ONLY rows of the form
///
///     sum(a_j * x_j) <= b
///
/// where every a_j > 0, every x_j is kInteger with bounds exactly [0, 1] (binary),
/// b is finite and positive, and no continuous column appears.
///
/// Any row outside that class returns std::nullopt. No cut is ever emitted when an
/// ambiguous or potentially invalid coefficient cannot be established exactly.
///
/// LIFTING: exact sequential lifting is applied (Issue #23 requirement). For each
/// variable outside the cover its lifting coefficient is computed from an exact 0/1
/// knapsack auxiliary problem. Lifting coefficients of zero are valid and are included.
///
/// The function does NOT append the cut to `model`. The caller is responsible for that.
[[nodiscard]] std::optional<KnapsackCoverCut> generate_knapsack_cover_cut(const Model& model,
                                                                          Index row);

// =========================================================================================
// Gomory mixed-integer cuts (Stage 3B)
// =========================================================================================

/// The result of Gomory mixed-integer cut generation for a single tableau row.
///
/// The returned inequality is canonically oriented over the ORIGINAL model columns:
///
///     sum(coeff[j] * x_j) <= rhs
///
/// The generator guarantees that this cut is mathematically valid. No filtering is applied.
struct Cut {
  std::vector<double> coeff;  ///< per-column coefficient (indexed by model.num_cols())
  double rhs = 0.0;           ///< right-hand side
};

/// Try to generate a Gomory mixed-integer cut from a specific basis row.
///
/// Returns std::nullopt if the basic variable is not eligible, the fraction is
/// near-integral, or a numerically ambiguous state is encountered.
[[nodiscard]] std::optional<Cut> generate_gmi_cut(const Model& model, const Solution& solution,
                                                  Index basis_row);

/// Reusable production context for multi-row GMI generation.
/// Performs exactly one basis reconstruction and one SparseLu factorization.
struct RootGmiContext {
  std::vector<unsigned char> slot_is_structural;
  std::vector<Index> slot_original_index;
  std::vector<double> logical_values;
  std::vector<Index> logical_rows;
  std::vector<LuColumn> columns;
  SparseLu lu;
  bool is_valid = false;

  /// Construct the context from the root solution. Factorizes the basis exactly once.
  RootGmiContext(const Model& model, const Solution& solution);

  // Non-copyable and non-movable to prevent LuColumn pointer invalidation
  RootGmiContext(const RootGmiContext&) = delete;
  RootGmiContext& operator=(const RootGmiContext&) = delete;
  RootGmiContext(RootGmiContext&&) = delete;
  RootGmiContext& operator=(RootGmiContext&&) = delete;

  /// Get the tableau row for a specific basis slot without re-factorizing.
  [[nodiscard]] std::optional<detail::ReconstructedTableauRow> tableau_row(
      const Model& model, const Solution& solution, Index basis_row) const;
};

/// Production multi-row GMI generator.
/// Iterates over all fractional basic structural integer variables, generating one cut per
/// eligible row. Returns candidates in deterministic basis-slot order.
[[nodiscard]] std::vector<Cut> generate_gmi_cuts(const Model& model, const Solution& solution);

// =========================================================================================
// Cut Filtering and Deduplication (Stage 4C)
// =========================================================================================

/// Reason why a cut candidate was rejected (or accepted) during filtering.
enum class CutFilterReason {
  kAccepted,
  kNonfinite,
  kEmptySupport,
  kTooDense,
  kCoefficientRatio,
  kInsufficientViolation,
  kDuplicate
};

/// A candidate cut paired with its acceptance/rejection status.
struct FilteredCut {
  Cut cut;
  CutFilterReason reason;
};

/// Validates, filters, and deduplicates a set of candidate cuts.
///
/// Enforces density, coefficient-ratio, and root-LP violation policies, and rejects numerical
/// duplicates. Returns the complete set of candidates (accepted and rejected) without modifying
/// their original mathematical representation.
[[nodiscard]] std::vector<FilteredCut> filter_and_deduplicate_cuts(
    const Model& model, const Solution& root_solution, const std::vector<Cut>& candidates);

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

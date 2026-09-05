// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the frozen model and solution interface.
//
// FROZEN INTERFACE (CLAUDE.md). Model is what every reader produces and every engine
// consumes. Solution is what every engine produces. solve() is the single seam where the
// simplex, PDHG, IPM, QP and branch-and-cut engines plug in. Changing anything in this
// file breaks work in three directories at once, so it does not change without an explicit
// decision recorded in the commit message.
//
// Two design choices here exist purely so that later phases do not force a rewrite:
//
//  1. ROWS CARRY TWO-SIDED BOUNDS. A row is  row_lower[i] <= a_i . x <= row_upper[i].
//     Equality is lower == upper; a <= row has lower = -inf; a >= row has upper = +inf; a
//     free (MPS "N") row that is not the objective has both infinite. This is exactly the
//     shape the MPS RANGES section produces, so the reader never has to invent slack
//     variables, and the dual simplex bound-flipping ratio test in Phase 6 gets the
//     two-sided form it needs for free.
//
//  2. THE QUADRATIC OBJECTIVE AND INTEGRALITY MARKERS ARE PRESENT FROM DAY ONE, even
//     though Phase 2 solves only LPs. An LP simply has an empty hessian and all-continuous
//     columns. This is what "modular and extensible to MIQP/NLP/MINLP" costs at this stage:
//     a few unused fields, versus a model-format migration later.
//
// Objective (before ObjSense is applied):
//     objective_offset  +  c . x  +  0.5 * x^T Q x
// Q is symmetric and stored LOWER-TRIANGULAR INCLUDING THE DIAGONAL. Only the stored half
// is kept; the 0.5 factor and the symmetry are applied by whoever evaluates it. This is
// the QPS convention, so a QPLIB/QPS reader maps onto it without a transformation.
#pragma once

#include <string>
#include <vector>

#include "sankhya/options.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya {

/// Direction of optimization. Stored on the model rather than folded into the cost vector
/// so that reported duals and reduced costs keep the sign convention of the original file.
enum class ObjSense { kMinimize, kMaximize };

/// Per-column variable class. Binary is not a separate kind: it is kInteger with bounds
/// [0, 1], which is what the MPS BV bound type produces and what branching expects.
enum class VarType : std::uint8_t { kContinuous, kInteger };

/// Basis status of a column or row. Reported by the simplex, consumed by warm starting and
/// by branch-and-cut. First-order engines (PDHG) leave everything kUnknown - they produce
/// no basis, and that is a documented limitation, not a defect.
enum class BasisStatus : std::uint8_t {
  kUnknown,
  kBasic,
  kAtLower,
  kAtUpper,
  kNonbasicFree,  // free variable held at zero
  kFixed          // lower == upper
};

/// Terminal state of a solve. Every engine must set exactly one of these.
enum class SolveStatus : std::uint8_t {
  kNotSolved,
  kOptimal,
  kInfeasible,
  kUnbounded,
  /// Detected as "not both feasible and bounded" without separating the two cases. Some
  /// first-order methods legitimately stop here; reporting it honestly beats guessing.
  kInfeasibleOrUnbounded,
  /// A feasible point exists and is reported, but optimality was not proven (MIP gap open,
  /// or a limit hit with an incumbent in hand).
  kFeasible,
  kIterationLimit,
  kTimeLimit,
  kNodeLimit,
  kNumericalError,
  kModelError
};

/// Human-readable name for a status, for logs and the JSON result blob.
[[nodiscard]] const char* to_string(SolveStatus status) noexcept;
[[nodiscard]] const char* to_string(BasisStatus status) noexcept;
[[nodiscard]] const char* to_string(VarType type) noexcept;

// =========================================================================================
// Model
// =========================================================================================

/// A linear, mixed-integer or convex quadratic optimization model.
///
///     optimize   objective_offset + c.x + 0.5 x^T Q x
///     subject to row_lower <= A x <= row_upper
///                col_lower <=  x  <= col_upper
///                x_j integral for every j with col_type[j] == kInteger
class Model {
 public:
  // ---- Identification -----------------------------------------------------------------

  /// Problem name from the source file. Free-form, used only in logs and reports.
  std::string name;

  /// Path the model was read from, when it came from a file. Empty for models built
  /// programmatically through the C API.
  std::string source_path;

  // ---- Objective ----------------------------------------------------------------------

  ObjSense sense = ObjSense::kMinimize;

  /// Constant term. MPS carries this as an RHS entry on the objective row, whose sign
  /// convention is NEGATED relative to the objective constant - the reader is responsible
  /// for that flip so that everything downstream can simply add this value.
  double objective_offset = 0.0;

  /// Linear objective coefficients, one per column.
  std::vector<double> col_cost;

  /// Lower triangle (including diagonal) of the symmetric Hessian Q, num_cols x num_cols.
  /// Empty for an LP or MILP. See the file header for the 0.5 factor convention.
  SparseMatrix hessian;

  // ---- Columns ------------------------------------------------------------------------

  /// Bounds, one entry per column. Use -kInfinity / +kInfinity for absent bounds.
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<VarType> col_type;

  /// Column names. Either empty (names not retained) or exactly num_cols long. Solution
  /// files and the verifier match on these, so a reader that has names must keep them.
  std::vector<std::string> col_names;

  // ---- Rows ---------------------------------------------------------------------------

  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::vector<std::string> row_names;

  /// Constraint matrix, num_rows x num_cols, column-compressed. Must be frozen before any
  /// engine sees it; validate() enforces that.
  SparseMatrix matrix;

  // ---- Derived queries ----------------------------------------------------------------

  [[nodiscard]] Index num_cols() const noexcept { return static_cast<Index>(col_cost.size()); }
  [[nodiscard]] Index num_rows() const noexcept { return static_cast<Index>(row_lower.size()); }
  [[nodiscard]] Index num_nonzeros() const noexcept { return matrix.num_nonzeros(); }

  /// True when any column is integral - i.e. this is a MILP or MIQP.
  [[nodiscard]] bool has_integrality() const noexcept;

  /// Number of integral columns.
  [[nodiscard]] Index num_integer_columns() const noexcept;

  /// True when the Hessian holds any entry - i.e. this is a QP or MIQP.
  [[nodiscard]] bool has_quadratic_objective() const noexcept;

  /// True when column j is fixed (lower == upper).
  [[nodiscard]] bool is_fixed_column(Index j) const noexcept;

  /// True when row i is an equality.
  [[nodiscard]] bool is_equality_row(Index i) const noexcept;

  /// The sign multiplier that converts the stored objective into a minimization objective.
  /// +1 for kMinimize, -1 for kMaximize. Engines minimize internally and use this to
  /// report the objective in the sense of the original file.
  [[nodiscard]] double sense_multiplier() const noexcept {
    return sense == ObjSense::kMaximize ? -1.0 : 1.0;
  }

  // ---- Construction helpers -----------------------------------------------------------

  /// Resize every per-column array to `n`, filling new columns with cost 0, bounds
  /// [0, +inf) - the MPS default - and kContinuous.
  void resize_columns(Index n);

  /// Resize every per-row array to `m`, filling new rows with a free range.
  void resize_rows(Index m);

  /// Evaluate the objective at `x` in the sense of the original file, including the
  /// offset and the quadratic term. `x` must have num_cols entries.
  [[nodiscard]] double evaluate_objective(const double* x) const;

  /// Structural self-check. Returns an empty string when the model is well formed, or a
  /// one-line description of the first problem found. Every reader calls this before
  /// handing a model to an engine: a malformed model produces a plausible-looking wrong
  /// answer rather than a crash, which is exactly the failure mode CLAUDE.md warns about.
  [[nodiscard]] std::string validate() const;
};

// =========================================================================================
// Solution
// =========================================================================================

/// The result of a solve. Vectors are either empty (the engine produced nothing of that
/// kind) or exactly the right length; a consumer must check.
class Solution {
 public:
  SolveStatus status = SolveStatus::kNotSolved;

  /// Objective value at col_value, in the sense of the original model. Meaningless unless
  /// status is kOptimal or kFeasible.
  double objective = 0.0;

  /// Best proven bound on the objective. For an LP this equals `objective` at optimality.
  /// For a MIP it is the global dual bound over the open tree.
  double dual_bound = 0.0;

  /// Primal values, num_cols entries.
  std::vector<double> col_value;

  /// Row activities A x, num_rows entries. Recomputed rather than accumulated, so that it
  /// is an independent check on the primal values rather than a restatement of them.
  std::vector<double> row_activity;

  /// Dual multipliers on the rows, num_rows entries. Sign convention: for a minimization
  /// problem, y_i >= 0 on an active lower bound (a_i.x = row_lower[i]) and y_i <= 0 on an
  /// active upper bound. These are the shadow prices the case studies report in rupees.
  std::vector<double> row_dual;

  /// Reduced costs on the columns, num_cols entries: d = c - A^T y (plus Q x for a QP).
  std::vector<double> col_dual;

  /// Basis, when the engine produces one. Empty for first-order methods.
  std::vector<BasisStatus> col_status;
  std::vector<BasisStatus> row_status;

  // ---- Reported quality. Never assumed - always measured before reporting. -------------

  double primal_infeasibility = 0.0;  ///< max violation over row and column bounds

  /// The same violations, each divided by the numerical scale of the quantity it was
  /// measured on (#34).
  ///
  /// WHY BOTH EXIST. `primal_infeasibility` is an absolute number, and an absolute number is
  /// the wrong question on a badly scaled model. Netlib `grow7` is the case that forced this:
  /// its largest solution value is 4.8e+07, so the 1e-7 absolute tolerance is 2.1e-15
  /// RELATIVE - below what double precision can deliver after 297 iterations of arithmetic.
  /// Its worst violation, 2.0e-07, is 4.2e-15 relative, about nineteen machine epsilons. The
  /// point is as accurate as doubles allow and was being reported as a numerical failure.
  ///
  /// The scale is the ROW'S OWN TERM MAGNITUDE, max |a_ij * x_j|, not the row's bound. The
  /// row that fails on grow7 is an equality to ZERO, so dividing by the bound would change
  /// nothing; what makes its residual large is cancellation between terms of magnitude 1e+07,
  /// and the achievable accuracy of a sum is set by the size of what is being summed. For a
  /// column bound the scale is |x_j| for the same reason.
  ///
  /// The absolute figure is still what gets REPORTED, because it is the one a reader can
  /// check by hand against the model. This is what the status decision uses.
  double primal_infeasibility_scaled = 0.0;

  /// The dual violations, each divided by the numerical scale of the quantity it was
  /// measured on - the dual counterpart of primal_infeasibility_scaled, and needed for the
  /// same model: grow7's dual infeasibility is 6.1 absolute against costs and prices of
  /// order 1e+07, which is 6e-07 relative, i.e. a point at the precision floor being called
  /// a failed optimality claim.
  ///
  /// A COLUMN'S scale is the larger of its cost and the largest term of A^T y in that
  /// column: the reduced cost d_j = c_j - a_j^T y is a difference of those quantities, and
  /// when they are large and nearly equal the leading digits cancel, so the achievable
  /// accuracy of d_j is set by their size, exactly as a row activity's is by its terms.
  ///
  /// A ROW'S scale is the infinity norm of the whole dual vector. A row price has no terms
  /// of its own to compare against - its sign condition is the condition - so the only
  /// honest scale is the size of the prices it sits among. That is a weaker test than the
  /// column one and is stated as such: it says "this price is small relative to its
  /// neighbours", not "this price is right".
  double dual_infeasibility_scaled = 0.0;
  double dual_infeasibility = 0.0;  ///< max violation of the reduced-cost sign conditions
  double complementarity_violation = 0.0;
  double integrality_violation = 0.0;

  /// (objective - dual_bound) in absolute and relative terms. Zero for a solved LP.
  double absolute_gap = 0.0;
  double relative_gap = 0.0;

  // ---- Effort -------------------------------------------------------------------------

  Count iterations = 0;  ///< simplex/IPM/PDHG iterations
  Count nodes = 0;       ///< branch-and-cut nodes
  Count cuts_applied = 0;
  double solve_seconds = 0.0;

  /// Which engine produced this: "simplex-primal", "pdhg-cpu", "branch-and-cut", ...
  std::string algorithm;

  /// Free-form detail, especially for kNumericalError and kModelError.
  std::string message;

  /// True when the status indicates a usable primal point.
  [[nodiscard]] bool has_primal_values() const noexcept {
    return status == SolveStatus::kOptimal || status == SolveStatus::kFeasible;
  }

  /// Allocate every vector to match `model`, filled with zeros / kUnknown.
  void allocate_for(const Model& model);

  /// Recompute row_activity, every infeasibility measure and the gaps from col_value and
  /// row_dual. Engines call this immediately before returning, so that the quality numbers
  /// in the log are measured facts rather than the engine's own opinion of itself.
  void recompute_quality(const Model& model);
};

// =========================================================================================
// The single entry point
// =========================================================================================

/// Solve `model` under `options` and return a Solution.
///
/// This is the seam. The dispatcher picks an engine from the model class (LP / MILP / QP /
/// MIQP) and the "algorithm" option, and future engines are added here and nowhere else.
/// It never throws: every failure, including a malformed model, comes back as a status.
[[nodiscard]] Solution solve(const Model& model, const Options& options);

}  // namespace sankhya

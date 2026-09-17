// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the frozen model and solution interface.
//
// FROZEN INTERFACE (ENGINEERING_RULES.md). Model is what every reader produces and every engine
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

#include <limits>
#include <string>
#include <vector>

#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
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
  /// A feasible point exists and is reported, but optimality was not proven: a node or time
  /// limit hit with an incumbent in hand, or a first-order method that met its request but
  /// not the project standard. A MIP that met its gap target reports kOptimal with the
  /// achieved gap in the message (#188).
  kFeasible,
  kIterationLimit,
  /// The point in hand when the limit fell. On a MILP with no incumbent yet, that point is
  /// the last node's LP relaxation - fractional, and said so in the message - because a
  /// limited search that found nothing still has a point to show, and a caller who wants
  /// integrality reads integrality_violation (#223).
  kTimeLimit,
  kNodeLimit,
  kNumericalError,
  kModelError,
  /// Stopped by the caller - a progress callback that returned non-zero, SolveControl::
  /// interrupt(), or SIGINT on the CLI. Carries a point exactly as kTimeLimit does (#223).
  kInterrupted
};

[[nodiscard]] constexpr bool claims_a_point(SolveStatus status) noexcept;

/// Which resource ended a solve that did not end on the mathematics (#289).
///
/// The status already distinguishes a resource termination from a mathematical verdict, but
/// it cannot always name the resource: a MILP that hits a limit holding an incumbent reports
/// kFeasible, and which limit it was lived only in the message. This says it in a field.
enum class LimitReason : std::uint8_t { kNone, kInterrupt, kTime, kIterations, kNodes };

/// "none", "user_interrupt", "time_limit", "iteration_limit", "node_limit".
[[nodiscard]] const char* to_string(LimitReason reason) noexcept;

/// The status a solve carries when this is the reason it stopped and it found no point.
[[nodiscard]] SolveStatus status_for(LimitReason reason) noexcept;

// =========================================================================================

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
  /// answer rather than a crash, which is exactly the failure mode ENGINEERING_RULES.md warns
  /// about.
  [[nodiscard]] std::string validate() const;

  /// A 64-bit identity for this model: two runs that report the same fingerprint solved the
  /// same numbers (#288).
  ///
  /// FROZEN INTERFACE, ADDITION. This adds a member function to Model; nothing existing
  /// changes shape or meaning, and no engine is required to call it.
  ///
  /// It is FNV-1a over a canonical byte image of the dimensions, the sense, the offset, the
  /// column and row data and both matrices in their stored order - not a cryptographic
  /// digest, and it is not used for security. Distinct models CAN collide; what it is for is
  /// telling a reproducibility report that run 2 was handed the model run 1 was handed,
  /// which the name in a header cannot. Bit patterns are hashed, so -0.0 and 0.0 differ, and
  /// two NaNs differ unless they carry the same payload.
  [[nodiscard]] std::uint64_t fingerprint() const noexcept;
};

// =========================================================================================
// Solution
// =========================================================================================

/// The result of a solve. Vectors are either empty (the engine produced nothing of that
/// kind) or exactly the right length; a consumer must check.
class Solution {
 public:
  SolveStatus status = SolveStatus::kNotSolved;

  /// Which resource ended the solve, when one did (#289).
  ///
  /// FROZEN INTERFACE, ADDITION. A new field with a default that means "nothing stopped it";
  /// no existing field changes shape or meaning, and a consumer that ignores it reads the
  /// same solution it read before. kNone on a solve that ended on the mathematics.
  LimitReason stopped_by = LimitReason::kNone;

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

  // ---- Certificates for the two verdicts that have no point (#191) ---------------------
  //
  // These are an ADDITION to this frozen interface, made deliberately and called out here
  // rather than slipped in: every existing consumer ignores them, and both default to empty,
  // which is this class's established way of saying "the engine produced nothing of that
  // kind". See include/sankhya/certificate.hpp for what they mean and how they are checked.

  /// A valid point has all columns and row activities populated and semantically valid.

  /// Farkas multipliers, one per row, when `status` is kInfeasible and the engine could
  /// prove it. Aggregating the rows with these weights yields an inequality no point in the
  /// column box satisfies. Empty when no proof was produced - presolve concludes
  /// infeasibility from bound arithmetic and carries its reason in `message` instead.
  std::vector<double> farkas_dual;

  /// A ray, one entry per column, when `status` is kUnbounded: a direction no bound blocks
  /// along which the objective improves without limit. Read together with `col_value`, which
  /// carries the feasible point it starts from.
  std::vector<double> primal_ray;

  // ---- Sensitivity ranging (populated only when options.get_bool("ranging") is true) ----
  //
  // An ADDITION to this frozen interface, called out here as farkas_dual was in #191; every
  // field defaults to empty or false and every existing consumer ignores them (#220).
  //
  // For column j, in the MODEL'S OWN SENSE: col_ranging_lower[j] = how far c_j can fall
  //   and col_ranging_upper[j] how far it can rise before the optimal basis changes.
  // For row i: row_ranging_lower[i] = how far the row's active bound can fall and
  //   row_ranging_upper[i] how far it can rise before the basis becomes primal infeasible;
  //   for a row that is not binding, how far its upper bound can fall and its lower bound
  //   rise before it binds. A fixed column reports +inf on both sides.
  // ranging_basis_degenerate: a basic variable sits on a bound, so the vertex has more than
  //   one basis and the ranges are those of the reported one, not of the unique optimum.
  // Reference: Chvatal, "Linear Programming", ch. 10 (W. H. Freeman, 1983).
  std::vector<double> col_ranging_lower;
  std::vector<double> col_ranging_upper;
  std::vector<double> row_ranging_lower;
  std::vector<double> row_ranging_upper;
  bool ranging_basis_degenerate = false;

  // ---- Irreducible Infeasible Subsystem (IIS), computed by the deletion filter (#217) ----
  //
  // An ADDITION to this frozen interface, called out here as farkas_dual was in #191. These
  // default to empty; every existing consumer ignores them. Populated only when status is
  // kInfeasible, a Farkas certificate exists, and the compute_iis option is enabled.
  //
  // Algorithm: Chinneck & Dravnieks, "Locating minimal infeasible constraint sets in linear
  // programs", ORSA J. Computing 3(2) (1991). Starting from the k rows and column bounds
  // the Farkas certificate names, remove each one in turn, re-solve, and keep it out
  // permanently if the sub-problem stays infeasible. The result is irreducible: removing
  // any single element from it makes the sub-system feasible.

  /// Row indices (0-based) that form the IIS.
  std::vector<Index> iis_rows;
  /// Column indices (0-based) whose lower bound is in the IIS.
  std::vector<Index> iis_col_lo;
  /// Column indices (0-based) whose upper bound is in the IIS.
  std::vector<Index> iis_col_hi;
  /// One witness per IIS element, in the order iis_rows, iis_col_lo, iis_col_hi: a point
  /// (num_cols values) that satisfies every other element of the IIS and violates that one.
  /// It is the deletion filter's own evidence that the element is necessary - the trial
  /// solve that kept it - retained so that tools/verify_solution.py can check
  /// irreducibility by arithmetic alone. Empty when iis_inconclusive is set. When an IIS is
  /// reported, farkas_dual is the certificate of the IIS itself: its support lies inside
  /// iis_rows and the bounds it uses are iis_col_lo / iis_col_hi, so the same checker
  /// proves the subsystem infeasible on its own.
  std::vector<std::vector<double>> iis_witnesses;
  /// True when a trial solve ended in neither verdict (a limit or a numerical error), or the
  /// final certificate could not be re-proved: the candidates concerned were kept, the IIS
  /// may not be irreducible, and the .sol file says `iis_irreducible not-claimed`.
  bool iis_inconclusive = false;

  // ---- What presolve did (#286) ----------------------------------------------------------
  //
  // An ADDITION to this frozen interface, called out here as farkas_dual was in #191. Default
  // constructed with ran = false; every existing consumer ignores it.
  //
  // Presolve is the one stage that changes the model a user handed over, and until now the
  // only way to see what it changed was a single log line. This is the same information,
  // structured: what came in, what came out, which reductions accounted for the difference,
  // what presolve declined to do and why, and how long it took. Filled by presolve() and
  // carried through postsolve; written to the stats JSON and printed by the CLI.
  struct PresolveReport {
    /// Why presolve stopped. A fixed point is the ordinary outcome: a pass that removed
    /// nothing.
    enum class Termination {
      kNotRun,           ///< presolve was off, or skipped for a stated reason
      kFixedPoint,       ///< a pass changed nothing, which is where the reductions run out
      kPassLimit,        ///< the safety cap was reached with reductions still firing
      kProvedInfeasible  ///< a reduction settled the model on its own
    };

    bool ran = false;
    Termination termination = Termination::kNotRun;
    /// Set when `ran` is false and something other than the option decided it.
    std::string skipped_because;

    Index original_rows = 0, original_cols = 0, original_nonzeros = 0;
    Index reduced_rows = 0, reduced_cols = 0, reduced_nonzeros = 0;
    Count passes = 0;
    double seconds = 0.0;

    // Counts by the reduction that fired, named for the operation the implementation
    // actually performs rather than for a textbook category it does not distinguish.
    Count empty_rows = 0;
    Count redundant_rows = 0;
    Count singleton_rows = 0;
    Count fixed_columns = 0;
    Count empty_columns = 0;
    Count free_column_singletons = 0;
    Count doubleton_equations = 0;
    Count bounds_tightened = 0;
    Count integer_bounds_rounded = 0;

    // What presolve deliberately did NOT do, which is as much a part of explaining a reduced
    // model as what it did (#301): a column carrying curvature, or an integer column whose
    // substitution would come back fractional, is left in place on purpose.
    Count quadratic_columns_protected = 0;
    Count integer_reductions_declined = 0;

    [[nodiscard]] Index rows_removed() const noexcept { return original_rows - reduced_rows; }
    [[nodiscard]] Index columns_removed() const noexcept {
      return original_cols - reduced_cols;
    }
    [[nodiscard]] Index nonzeros_removed() const noexcept {
      return original_nonzeros - reduced_nonzeros;
    }

    /// Percentage removed, 0 when the model had none to begin with.
    [[nodiscard]] static double percentage(Index removed, Index original) noexcept {
      return original > 0 ? 100.0 * static_cast<double>(removed) / static_cast<double>(original)
                          : 0.0;
    }
    [[nodiscard]] double row_reduction_percent() const noexcept {
      return percentage(rows_removed(), original_rows);
    }
    [[nodiscard]] double column_reduction_percent() const noexcept {
      return percentage(columns_removed(), original_cols);
    }
    [[nodiscard]] double nonzero_reduction_percent() const noexcept {
      return percentage(nonzeros_removed(), original_nonzeros);
    }
  };

  PresolveReport presolve_report;

  // ---- Solution pool (#225) --------------------------------------------------------------
  //
  // An ADDITION to this frozen interface, called out here as farkas_dual was in #191. Empty
  // unless branch and bound produced it; every existing consumer ignores it.
  //
  // The integer-feasible points the search found, each a different integer assignment, the
  // reported solution first and the rest best first. pool[0] carries exactly `col_value` and
  // `objective`, so a reader of the main solution sees no change. Every member passed the same
  // feasibility test against the original model that the incumbent passes. Objectives are in
  // the model's own sense and include the offset. Options: pool_size, pool_gap,
  // pool_diversity, pool_complete.
  struct PoolEntry {
    double objective = 0.0;
    std::vector<double> col_value;
  };
  std::vector<PoolEntry> pool;

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
  ///
  /// A COMPLEMENTARITY PRODUCT |multiplier| * slack is divided by the multiplier's scale
  /// (as above) times the primal quantity's (as for primal_infeasibility_scaled): it is a
  /// product of two measured numbers and inherits the precision of both. A reduced cost
  /// that is a rounding residue of its terms, on a column that is interior by hundreds,
  /// is a product of order 1e-7 and a violation of nothing.
  double dual_infeasibility_scaled = 0.0;
  double dual_infeasibility = 0.0;  ///< max violation of the reduced-cost sign conditions
  double complementarity_violation = 0.0;
  double integrality_violation = 0.0;

  /// Iterative refinement of the final basis (#72; Wilkinson, "Rounding Errors in Algebraic
  /// Processes", 1963). Steps taken, and the largest residual of the basic system - the
  /// primal B x_B = -N x_N and the dual B^T y = c_B, whichever is worse - before the first
  /// step and after the last. Zero steps when the engine produces no basis. Reported so a
  /// point that only meets tolerance after refinement is visible as such rather than hidden.
  Count refinement_steps = 0;
  double residual_before_refinement = 0.0;
  double residual_after_refinement = 0.0;

  /// (objective - dual_bound) in absolute and relative terms. Zero for a solved LP.
  double absolute_gap = 0.0;
  double relative_gap = 0.0;

  // ---- Effort -------------------------------------------------------------------------

  Count iterations = 0;  ///< simplex/IPM/PDHG iterations
  Count nodes = 0;       ///< branch-and-cut nodes
  Count cuts_applied = 0;
  /// The root LP relaxation's objective before and after the root cut round (#221), in
  /// the model's own sense and units; NaN when no branch-and-cut ran. The share of the
  /// integrality gap the cuts closed is (after - before) / (objective - before), which the
  /// MIPLIB runner records per instance so the effect of a cut family is a number.
  double root_bound = std::numeric_limits<double>::quiet_NaN();
  double root_bound_after_cuts = std::numeric_limits<double>::quiet_NaN();
  /// Of `iterations`, those spent by the interior-point polish of a PDHG answer (#229);
  /// zero when no polish ran. A benchmark row can then say which phase did what.
  Count polish_iterations = 0;
  double solve_seconds = 0.0;

  /// Which engine produced this: "simplex-primal", "pdhg-cpu", "branch-and-cut", ...
  std::string algorithm;

  /// Free-form detail, especially for kNumericalError and kModelError.
  std::string message;

  /// True when the status says a point is reported; the same answer as
  /// claims_a_point(status), kept because this interface is frozen (ENGINEERING_RULES.md).
  [[nodiscard]] bool has_primal_values() const noexcept { return claims_a_point(status); }

  /// Clear the vectors and quality measurements, leaving the status intact.
  void clear_values() {
    col_value.clear();
    row_activity.clear();
    row_dual.clear();
    col_dual.clear();
    col_status.clear();
    row_status.clear();
    objective = 0.0;
    dual_bound = 0.0;
    primal_infeasibility = 0.0;
    primal_infeasibility_scaled = 0.0;
    dual_infeasibility = 0.0;
    dual_infeasibility_scaled = 0.0;
    integrality_violation = 0.0;
    pool.clear();
  }

  /// Allocate every vector to match `model`, filled with zeros / kUnknown.
  void allocate_for(const Model& model);

  /// Recompute row_activity, every infeasibility measure and the gaps from col_value and
  /// row_dual. Engines call this immediately before returning, so that the quality numbers
  /// in the log are measured facts rather than the engine's own opinion of itself.
  void recompute_quality(const Model& model);
};

/// Does a solve ending in this state hand back a point?
///
/// The question a writer, a checker and a caller all have to answer, asked once here so they
/// cannot answer it differently (#200). Getting it wrong in the permissive direction is what
/// #191 was: an `infeasible` answer was written as a full all-zero point, and the project's
/// own independent checker read that point, found it violated the rows, and printed REJECTED
/// at a correct answer. That was fixed for `infeasible` alone, and every other verdict with
/// nothing to show kept the bug.
///
/// `kUnbounded` says yes deliberately. Since #191 it carries the feasible point its ray
/// starts from, because a ray that begins outside the feasible region proves nothing, and a
/// checker needs both halves.
///
/// The limit states say yes because they normally stop with an iterate or an incumbent in
/// hand. The one exception is a node limit reached before branch and bound found any integer
/// point, which reports no objective and infinite gaps rather than a point (see
/// `src/mip/branch_and_bound.cpp`); that case predates this predicate and is unchanged by it.
///
[[nodiscard]] constexpr bool claims_a_point(SolveStatus status) noexcept {
  switch (status) {
    case SolveStatus::kOptimal:
    case SolveStatus::kFeasible:
    case SolveStatus::kUnbounded:
    case SolveStatus::kIterationLimit:
    case SolveStatus::kTimeLimit:
    case SolveStatus::kNodeLimit:
    case SolveStatus::kInterrupted: return true;
    case SolveStatus::kNotSolved:
    case SolveStatus::kInfeasible:
    case SolveStatus::kInfeasibleOrUnbounded:
    case SolveStatus::kNumericalError:
    case SolveStatus::kModelError: return false;
    default: return false;
  }
}

[[nodiscard]] inline bool claims_a_point(const Solution& solution) noexcept {
  return claims_a_point(solution.status);
}

// =========================================================================================
// The single entry point
// =========================================================================================

/// Solve `model` under `options` and return a Solution.
///
/// This is the seam. The dispatcher picks an engine from the model class (LP / MILP / QP /
/// MIQP) and the "algorithm" option, and future engines are added here and nowhere else.
/// It never throws: every failure, including a malformed model, comes back as a status.
[[nodiscard]] Solution solve(const Model& model, const Options& options,
                             SolveControl* control = nullptr);

}  // namespace sankhya

/* SPDX-License-Identifier: Apache-2.0
 *
 * SANKHYA - C API.
 *
 * A C89-compatible surface over the C++ core, for callers that cannot or will not link C++:
 * other languages' FFIs, and the Python bindings that sit on top of this rather than on the
 * C++ types directly. Nothing here exposes a C++ type, a template, or an exception - the
 * whole point is a boundary a `ctypes` script can cross.
 *
 * THE SHAPE IS DELIBERATELY FAMILIAR. create / set / solve / query, string-named options,
 * integer status codes. That is the surface every industrial solver presents, and matching
 * it is what makes this drop-in adoptable for someone with existing CPLEX or Gurobi calling
 * code. Per ENGINEERING_RULES.md that is interface compatibility, not derivation: it is written
 * from the public shape those APIs document, and no solver source was read to produce it.
 *
 * ERRORS ARE RETURNED, NEVER THROWN. Every fallible call returns a sankhya_status. When one
 * is not SANKHYA_OK, sankhya_last_error() carries a human-readable reason for that thread.
 * A C caller cannot catch a C++ exception, so every entry point that could raise one wraps
 * its body and converts.
 *
 * OWNERSHIP. Handles returned by a *_create or *_solve function are owned by the caller and
 * must be released with the matching *_free. Every `const char*` returned by this API points
 * into storage owned by the library and is valid until the next call ON THE SAME THREAD that
 * could replace it - copy it if you need to keep it.
 *
 * THREADING. Handles are not internally synchronised: two threads must not touch one handle
 * at once. Distinct handles in distinct threads are fine, and the error string is
 * thread-local, so concurrent solves do not overwrite each other's diagnostics.
 */
#ifndef SANKHYA_H
#define SANKHYA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status codes ---------------------------------------------------------------------- */

typedef enum sankhya_status {
  SANKHYA_OK = 0,
  SANKHYA_ERROR_ARGUMENT = 1, /**< a null handle, or an index outside the model */
  SANKHYA_ERROR_IO = 2,       /**< the file could not be read or parsed */
  SANKHYA_ERROR_MODEL = 3,    /**< the model is not internally consistent; see last_error */
  SANKHYA_ERROR_OPTION = 4,   /**< no such option, or a value it will not accept */
  SANKHYA_ERROR_MEMORY = 5,   /**< allocation failed */
  SANKHYA_ERROR_INTERNAL = 6  /**< a C++ exception crossed the boundary and was converted */
} sankhya_status;

/** Mirrors sankhya::SolveStatus. Values are stable; new ones are appended. */
typedef enum sankhya_solve_status {
  SANKHYA_NOT_SOLVED = 0,
  SANKHYA_OPTIMAL = 1,
  SANKHYA_FEASIBLE = 2, /**< a usable point; optimality NOT proven */
  SANKHYA_INFEASIBLE = 3,
  SANKHYA_UNBOUNDED = 4,
  /** Not both feasible and bounded, without separating the two. Some first-order
   *  methods legitimately stop here; reporting it beats guessing which it was. */
  SANKHYA_INFEASIBLE_OR_UNBOUNDED = 10,
  SANKHYA_ITERATION_LIMIT = 5,
  SANKHYA_TIME_LIMIT = 6,
  SANKHYA_NODE_LIMIT = 7,
  SANKHYA_NUMERICAL_ERROR = 8,
  SANKHYA_MODEL_ERROR = 9,
  /** 10 is INFEASIBLE_OR_UNBOUNDED above. */
  SANKHYA_INTERRUPTED = 11
} sankhya_solve_status;

/* ---- Opaque handles -------------------------------------------------------------------- */

typedef struct sankhya_model sankhya_model;
typedef struct sankhya_options sankhya_options;
typedef struct sankhya_solution sankhya_solution;

/* ---- Library ---------------------------------------------------------------------------- */

/** Version string, e.g. "0.1.0 (abc1234, Release)". Never NULL. */
const char* sankhya_version(void);

/**
 * Human-readable reason for the most recent failing call ON THIS THREAD.
 *
 * Returns an empty string when nothing has failed. The pointer is valid until the next
 * failing call on this thread.
 */
const char* sankhya_last_error(void);

/** The infinity this API uses for absent bounds. Bounds at or beyond it are treated as free. */
double sankhya_infinity(void);

/* ---- Model ------------------------------------------------------------------------------ */

/** An empty minimisation model with no rows or columns. NULL only on allocation failure. */
sankhya_model* sankhya_model_create(void);
void sankhya_model_free(sankhya_model* model);

/**
 * Read a model from an MPS, QPS or LP file, choosing the reader by extension and content.
 *
 * On failure the handle is left unmodified and last_error carries the parser's message,
 * which names the line - these readers refuse ambiguous files rather than guessing, so a
 * failure here is usually a genuine defect in the file.
 */
sankhya_status sankhya_model_read(sankhya_model* model, const char* path);

/** 0 to minimise (the default), non-zero to maximise. */
sankhya_status sankhya_model_set_maximize(sankhya_model* model, int maximize);

/** Constant added to the objective. */
sankhya_status sankhya_model_set_objective_offset(sankhya_model* model, double offset);

/**
 * Append one column, returning its index through `index` when that is non-NULL.
 *
 * `name` may be NULL. Use +/- sankhya_infinity() for absent bounds. `is_integer` non-zero
 * makes this an integer column, which makes the model a MILP.
 */
sankhya_status sankhya_model_add_column(sankhya_model* model, double cost, double lower,
                                        double upper, int is_integer, const char* name,
                                        int* index);

/**
 * Append one row, returning its index through `index` when that is non-NULL.
 *
 * A range row is lower <= a'x <= upper; pass equal bounds for an equality, and an infinite
 * bound on one side for a one-sided inequality.
 */
sankhya_status sankhya_model_add_row(sankhya_model* model, double lower, double upper,
                                     const char* name, int* index);

/**
 * Set one constraint-matrix coefficient.
 *
 * Entries may be supplied in any order. Setting the same (row, column) twice REPLACES the
 * earlier value rather than summing it - summing is what MPS files mean by a repeated entry
 * and this API deliberately does not inherit that, because silently doubling a coefficient
 * is not a mistake a caller can see in the answer.
 *
 * A value of exactly zero removes the entry.
 */
sankhya_status sankhya_model_set_coefficient(sankhya_model* model, int row, int col,
                                             double value);

/**
 * Set one entry of the objective Hessian Q, making this a quadratic program.
 *
 * The objective is c'x + 0.5 x'Qx and Q is symmetric, so ONLY THE LOWER TRIANGLE is stored:
 * an entry (i, j) with i > j stands for both Q[i][j] and Q[j][i]. Passing (j, i) instead is
 * accepted and means the same thing. The 0.5 belongs to the objective, not to the value you
 * pass here - the same convention QPS files use.
 *
 * A non-convex Q is REFUSED at solve time with SANKHYA_MODEL_ERROR rather than solved to a
 * local point.
 */
sankhya_status sankhya_model_set_quadratic_coefficient(sankhya_model* model, int row, int col,
                                                       double value);

/* ---- Editing a model between solves (#218) ------------------------------------------- */

/**
 * Replace one column's bounds, one row's bounds, or one objective coefficient, in place.
 *
 * These are what a planner changes between two solves - a crude price, a tank capacity, a
 * demand - and a model edited this way keeps its structure, so a previous solution's basis
 * still describes it and sankhya_solve_from() can restart from it. Pass -sankhya_infinity()
 * or sankhya_infinity() for an absent bound. A column index or row index outside the model
 * is SANKHYA_ERROR_ARGUMENT.
 */
sankhya_status sankhya_model_set_col_bounds(sankhya_model* model, int col, double lower,
                                            double upper);
sankhya_status sankhya_model_set_row_bounds(sankhya_model* model, int row, double lower,
                                            double upper);
sankhya_status sankhya_model_set_objective_coefficient(sankhya_model* model, int col,
                                                       double cost);

int sankhya_model_num_cols(const sankhya_model* model);
int sankhya_model_num_rows(const sankhya_model* model);
int sankhya_model_num_nonzeros(const sankhya_model* model);

/**
 * Check the model for internal consistency without solving it.
 *
 * Returns SANKHYA_OK when the model is well formed, SANKHYA_ERROR_MODEL otherwise with the
 * reason in last_error.
 */
sankhya_status sankhya_model_validate(const sankhya_model* model);

/* ---- Progress Callback ------------------------------------------------------------------ */

typedef enum sankhya_progress_phase {
  SANKHYA_PHASE_PRESOLVE = 0,
  SANKHYA_PHASE_LP = 1,
  SANKHYA_PHASE_TREE = 2
} sankhya_progress_phase;

typedef struct sankhya_progress {
  sankhya_progress_phase phase;
  int64_t iterations;
  int64_t nodes;
  double objective;
  double best_bound;
  double gap;
  double elapsed_seconds;
  int64_t open_nodes;
} sankhya_progress;

/**
 * Register a progress callback. It will be called periodically during the solve.
 * If the callback returns non-zero, the solver will interrupt at the next safe point.
 */
sankhya_status sankhya_set_callback(sankhya_model* model,
                                    int (*callback)(const sankhya_progress*, void*),
                                    void* user_data);

/**
 * Request an asynchronous interruption from another thread or a signal handler.
 */
sankhya_status sankhya_model_interrupt(sankhya_model* model);

/* ---- Options ---------------------------------------------------------------------------- */

/** Options preset to their documented defaults. Run `sankhya options` to list them. */
sankhya_options* sankhya_options_create(void);
void sankhya_options_free(sankhya_options* options);

sankhya_status sankhya_options_set_bool(sankhya_options* options, const char* name, int value);
sankhya_status sankhya_options_set_int(sankhya_options* options, const char* name,
                                       int64_t value);
sankhya_status sankhya_options_set_double(sankhya_options* options, const char* name,
                                          double value);
sankhya_status sankhya_options_set_string(sankhya_options* options, const char* name,
                                          const char* value);

/* ---- Solve ------------------------------------------------------------------------------ */

/**
 * Solve, writing a newly allocated solution handle to `*solution`.
 *
 * `options` may be NULL for the defaults. The return value reports whether the CALL
 * succeeded, not what the solver concluded: a model proved infeasible returns SANKHYA_OK
 * with a solution whose status is SANKHYA_INFEASIBLE. Check both.
 *
 * The `model` parameter is non-const because the solver may register interruption state
 * or a progress callback on it during the solve.
 */
sankhya_status sankhya_solve(sankhya_model* model, const sankhya_options* options,
                             sankhya_solution** solution);

/**
 * Solve again, starting from the basis a previous solution of THIS model reported (#218).
 *
 * The typical use: solve, edit a bound or a cost with the sankhya_model_set_* calls above,
 * and solve from the previous solution. The simplex engines restart from that basis - the
 * dual simplex after bound and right-hand-side edits (the default), the primal simplex
 * (option algorithm=simplex) after cost edits - and finish in a handful of pivots where a
 * cold solve takes thousands; the pivot count is sankhya_solution_iterations(). Presolve is
 * bypassed on a warm solve, since the basis names the caller's rows and columns, and the
 * message says so. A `start` whose basis does not fit the model (columns or rows added or
 * removed since) is ignored with a warning in the log and the solve runs cold; a `start` from
 * the interior point or PDHG carries no basis and is likewise ignored. `start` may be NULL,
 * which is sankhya_solve().
 */
sankhya_status sankhya_solve_from(sankhya_model* model, const sankhya_options* options,
                                  const sankhya_solution* start, sankhya_solution** solution);

void sankhya_solution_free(sankhya_solution* solution);

sankhya_solve_status sankhya_solution_status(const sankhya_solution* solution);

/** Explanatory message from the solver. Empty when there is nothing to add. */
const char* sankhya_solution_message(const sankhya_solution* solution);

double sankhya_solution_objective(const sankhya_solution* solution);

/** Best proven bound. Equals the objective when optimality was proved. */
double sankhya_solution_dual_bound(const sankhya_solution* solution);

/**
 * The gap the solve finished with: objective minus dual bound, absolute and relative (#207).
 * Zero for a solved LP. For a MILP stopped on its gap target this is the gap it actually
 * reached, which is what distinguishes `optimal within 1e-4` from `optimal, tree exhausted`.
 * The target itself is the mip_relative_gap / mip_absolute_gap option the caller passed.
 */
double sankhya_solution_absolute_gap(const sankhya_solution* solution);
double sankhya_solution_relative_gap(const sankhya_solution* solution);

int64_t sankhya_solution_iterations(const sankhya_solution* solution);
int64_t sankhya_solution_nodes(const sankhya_solution* solution);
double sankhya_solution_seconds(const sankhya_solution* solution);

/**
 * MEASURED quality of the returned point, not asserted by the engine about itself.
 *
 * These are recomputed from the returned vectors before the solver reports anything, and
 * the dispatcher downgrades a status that disagrees with them. A caller writing its own
 * acceptance test should read these rather than trusting the status alone.
 */
double sankhya_solution_primal_infeasibility(const sankhya_solution* solution);
double sankhya_solution_dual_infeasibility(const sankhya_solution* solution);
double sankhya_solution_integrality_violation(const sankhya_solution* solution);

/**
 * Copy the primal column values into `values`, which must have room for `count` doubles.
 *
 * `count` must equal the model's column count; a mismatch returns SANKHYA_ERROR_ARGUMENT
 * rather than writing a partial vector, because a caller that has the dimension wrong is
 * about to misread every number it copies.
 */
sankhya_status sankhya_solution_col_values(const sankhya_solution* solution, double* values,
                                           int count);

/** Row activities a'x, same contract as sankhya_solution_col_values. */
sankhya_status sankhya_solution_row_activities(const sankhya_solution* solution, double* values,
                                               int count);

/** Row dual values (shadow prices), same contract. */
sankhya_status sankhya_solution_row_duals(const sankhya_solution* solution, double* values,
                                          int count);

/** Column reduced costs, same contract. */
sankhya_status sankhya_solution_col_duals(const sankhya_solution* solution, double* values,
                                          int count);

/**
 * Basis status of a column or row (#218). The simplex engines report one per column and per
 * row, with exactly as many BASIC entries as the model has rows; first-order and interior-
 * point solves (without crossover) report UNKNOWN throughout.
 */
typedef enum sankhya_basis_status {
  SANKHYA_BASIS_UNKNOWN = 0,
  SANKHYA_BASIS_BASIC = 1,
  SANKHYA_BASIS_AT_LOWER = 2,
  SANKHYA_BASIS_AT_UPPER = 3,
  SANKHYA_BASIS_FREE = 4, /**< a free column held at zero */
  SANKHYA_BASIS_FIXED = 5 /**< lower == upper */
} sankhya_basis_status;

/** Column basis statuses, same size contract as sankhya_solution_col_values. */
sankhya_status sankhya_solution_col_statuses(const sankhya_solution* solution, int* statuses,
                                             int count);
/** Row basis statuses, same contract. */
sankhya_status sankhya_solution_row_statuses(const sankhya_solution* solution, int* statuses,
                                             int count);

/* ---- Whether there is a point, and the certificates behind a verdict ------------------ */

/**
 * 1 when the status carries a point - optimal, feasible, a limit with an incumbent, and
 * unbounded (whose point is where the ray starts) - and 0 otherwise.
 *
 * Ask this BEFORE reading sankhya_solution_col_values. When it is 0, col_values still
 * succeeds and fills the buffer, but the numbers in it are not a point the solver claims and
 * mean nothing; without this an API caller could only tell the two kinds of status apart by
 * knowing the list by heart.
 */
int sankhya_solution_claims_a_point(const sankhya_solution* solution);

/**
 * Length of the Farkas certificate: the model's row count when an infeasible verdict carries
 * a proof, and 0 otherwise.
 *
 * 0 IS NOT AN ERROR. A verdict can be correct with no certificate - presolve concludes
 * infeasibility from bound arithmetic and gives its reason in the message instead - so
 * "no multipliers" is an ordinary answer, reported as a length rather than as a failure.
 */
int sankhya_solution_farkas_dual_length(const sankhya_solution* solution);

/**
 * Copy the Farkas multipliers, one per row. Aggregating the rows with these weights yields an
 * inequality no point in the column box satisfies, which is the answer to "why is my model
 * infeasible": the rows that conflict, weighted.
 *
 * Same contract as sankhya_solution_col_values, with `count` equal to
 * sankhya_solution_farkas_dual_length. When that length is 0, `values` may be NULL.
 */
sankhya_status sankhya_solution_farkas_dual(const sankhya_solution* solution, double* values,
                                            int count);

/** Length of the unbounded ray: the column count when the verdict is unbounded, else 0. */
int sankhya_solution_primal_ray_length(const sankhya_solution* solution);

/**
 * Copy the ray, one entry per column: a direction no bound blocks along which the objective
 * improves without limit. It starts from the point sankhya_solution_col_values returns.
 *
 * Same contract as sankhya_solution_farkas_dual.
 */
sankhya_status sankhya_solution_primal_ray(const sankhya_solution* solution, double* values,
                                           int count);

/* ---- Sensitivity ranging (`--option ranging=true`; #220) ------------------------------ */

/**
 * 1 when ranging was computed - col_ranging_lower/upper then have the model's column count
 * and row_ranging_lower/upper its row count - and 0 otherwise.
 *
 * Ask this before reading any of the four vectors below, the role
 * sankhya_solution_claims_a_point plays for col_values: at 0 they are still safe to copy
 * (length 0), but empty rather than ranged.
 */
int sankhya_solution_has_ranging(const sankhya_solution* solution);

/**
 * How far column j's cost can fall (lower) or rise (upper) before the optimal basis
 * changes, in the model's own sense. Same contract as sankhya_solution_col_values, except
 * that length 0 is not an error - see sankhya_solution_has_ranging.
 */
sankhya_status sankhya_solution_col_ranging_lower(const sankhya_solution* solution,
                                                  double* values, int count);
sankhya_status sankhya_solution_col_ranging_upper(const sankhya_solution* solution,
                                                  double* values, int count);

/**
 * How far row i's active bound can fall (lower) or rise (upper) before the basis becomes
 * primal infeasible; for a non-binding row, how far the far bound can move before it binds.
 * Same contract as the column ranges above, with `count` equal to the row count.
 */
sankhya_status sankhya_solution_row_ranging_lower(const sankhya_solution* solution,
                                                  double* values, int count);
sankhya_status sankhya_solution_row_ranging_upper(const sankhya_solution* solution,
                                                  double* values, int count);

/**
 * 1 when a basic variable sits on a bound, so the vertex has more than one basis and the
 * ranges above are of the reported one, not of a unique optimum. 0 otherwise, including
 * when ranging was not computed at all.
 */
int sankhya_solution_ranging_basis_degenerate(const sankhya_solution* solution);

/* ---- Irreducible Infeasible Subsystem (`--option compute_iis=true`; #217) ------------- */

/** Row indices (0-based) in the IIS. 0 when none was computed. */
int sankhya_solution_iis_row_count(const sankhya_solution* solution);

/** Copy the IIS row indices. Same contract as sankhya_solution_col_values, except that
 * length 0 is not an error: no IIS computed and an IIS with no row both report it. */
sankhya_status sankhya_solution_iis_rows(const sankhya_solution* solution, int* indices,
                                         int count);

/** Column indices (0-based) whose LOWER bound is in the IIS. */
int sankhya_solution_iis_col_lower_count(const sankhya_solution* solution);
sankhya_status sankhya_solution_iis_col_lower(const sankhya_solution* solution, int* indices,
                                              int count);

/** Column indices (0-based) whose UPPER bound is in the IIS. */
int sankhya_solution_iis_col_upper_count(const sankhya_solution* solution);
sankhya_status sankhya_solution_iis_col_upper(const sankhya_solution* solution, int* indices,
                                              int count);

/**
 * 1 when the deletion filter could not prove every retained element necessary - a trial
 * solve hit a limit or a numerical error instead of reaching a verdict - so the IIS may not
 * be irreducible. 0 when it is confirmed irreducible, including when none was computed.
 */
int sankhya_solution_iis_inconclusive(const sankhya_solution* solution);

/**
 * Number of IIS witnesses: one per element, in the order rows, then column lower bounds,
 * then column upper bounds - so witness k corresponds to row iis_rows[k] while
 * k < sankhya_solution_iis_row_count, and to the column arrays beyond that. 0 when no IIS
 * was computed, or when sankhya_solution_iis_inconclusive is 1 for an element this witness
 * would otherwise cover.
 */
int sankhya_solution_iis_witness_count(const sankhya_solution* solution);

/**
 * Copy witness `index`: a point, one value per model column, that satisfies every OTHER
 * element of the IIS and violates this one - the deletion filter's own evidence that the
 * element is necessary, kept so a caller can check irreducibility by arithmetic alone.
 * `count` must equal the model's column count. `index` out of range is
 * SANKHYA_ERROR_ARGUMENT.
 */
sankhya_status sankhya_solution_iis_witness(const sankhya_solution* solution, int index,
                                            double* values, int count);

/* ---- Solution pool (`--option pool_size=N`; #225) -------------------------------------- */

/**
 * Number of integer-feasible points the search kept, each a distinct integer assignment.
 * 0 unless branch and bound produced one. Member 0 is always the reported solution, so
 * col_values/objective and pool member 0 agree exactly.
 */
int sankhya_solution_pool_size(const sankhya_solution* solution);

/**
 * Member `index`'s objective, in the model's own sense with the offset included.
 * `index` out of range is SANKHYA_ERROR_ARGUMENT.
 */
sankhya_status sankhya_solution_pool_objective(const sankhya_solution* solution, int index,
                                               double* objective);

/**
 * Copy member `index`'s column values. `count` must equal the model's column count.
 * `index` out of range is SANKHYA_ERROR_ARGUMENT.
 */
sankhya_status sankhya_solution_pool_col_values(const sankhya_solution* solution, int index,
                                                double* values, int count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SANKHYA_H */

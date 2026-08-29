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
 * code. Per CLAUDE.md that is interface compatibility, not derivation: it is written from
 * the public shape those APIs document, and no solver source was read to produce it.
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
  SANKHYA_MODEL_ERROR = 9
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

/* ---- Options ---------------------------------------------------------------------------- */

/** Options preset to their documented defaults. Run `sankhya options` to list them. */
sankhya_options* sankhya_options_create(void);
void sankhya_options_free(sankhya_options* options);

sankhya_status sankhya_options_set_bool(sankhya_options* options, const char* name, int value);
sankhya_status sankhya_options_set_int(sankhya_options* options, const char* name, long value);
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
 */
sankhya_status sankhya_solve(const sankhya_model* model, const sankhya_options* options,
                             sankhya_solution** solution);

void sankhya_solution_free(sankhya_solution* solution);

sankhya_solve_status sankhya_solution_status(const sankhya_solution* solution);

/** Explanatory message from the solver. Empty when there is nothing to add. */
const char* sankhya_solution_message(const sankhya_solution* solution);

double sankhya_solution_objective(const sankhya_solution* solution);

/** Best proven bound. Equals the objective when optimality was proved. */
double sankhya_solution_dual_bound(const sankhya_solution* solution);

long sankhya_solution_iterations(const sankhya_solution* solution);
long sankhya_solution_nodes(const sankhya_solution* solution);
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SANKHYA_H */

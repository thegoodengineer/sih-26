// SPDX-License-Identifier: Apache-2.0
// SANKHYA - C API implementation.
//
// Every entry point here does three things and nothing else: validate its arguments, call
// into the C++ core, and convert whatever comes back - including an exception - into a
// status code. There is no solver logic in this file and there must not be. The moment a
// decision lives only in the C wrapper, the C++ callers and the CLI stop agreeing with the
// bindings about what the library does.
//
// THE MATRIX IS ACCUMULATED, NOT BUILT. sankhya::SparseMatrix is frozen once finalized, but
// a C caller sets coefficients one at a time in whatever order suits it. So the handle keeps
// triplets in a map and materialises the matrix at solve time. The map also gives
// set_coefficient its REPLACE semantics for free, which is deliberately unlike the MPS
// reader's "a repeated entry is an error": a programmatic caller overwriting a cell is
// ordinary, whereas a file containing the same cell twice is a defect in the file.

#include "sankhya/sankhya.h"

#include <exception>
#include <map>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/version.hpp"

namespace {

// Thread-local so that concurrent solves cannot overwrite each other's diagnostics. A caller
// debugging one failing solve while another thread is busy would otherwise read a message
// belonging to a model it has never seen.
thread_local std::string g_error;

sankhya_status fail(sankhya_status code, std::string message) {
  g_error = std::move(message);
  return code;
}

sankhya_status ok() {
  g_error.clear();
  return SANKHYA_OK;
}

sankhya_solve_status to_c_status(sankhya::SolveStatus status) {
  switch (status) {
    case sankhya::SolveStatus::kNotSolved: return SANKHYA_NOT_SOLVED;
    case sankhya::SolveStatus::kOptimal: return SANKHYA_OPTIMAL;
    case sankhya::SolveStatus::kFeasible: return SANKHYA_FEASIBLE;
    case sankhya::SolveStatus::kInfeasible: return SANKHYA_INFEASIBLE;
    case sankhya::SolveStatus::kUnbounded: return SANKHYA_UNBOUNDED;
    case sankhya::SolveStatus::kInfeasibleOrUnbounded: return SANKHYA_INFEASIBLE_OR_UNBOUNDED;
    case sankhya::SolveStatus::kIterationLimit: return SANKHYA_ITERATION_LIMIT;
    case sankhya::SolveStatus::kTimeLimit: return SANKHYA_TIME_LIMIT;
    case sankhya::SolveStatus::kNodeLimit: return SANKHYA_NODE_LIMIT;
    case sankhya::SolveStatus::kNumericalError: return SANKHYA_NUMERICAL_ERROR;
    case sankhya::SolveStatus::kModelError: return SANKHYA_MODEL_ERROR;
  }
  return SANKHYA_NOT_SOLVED;
}

/// Copy one of the solution's vectors out, refusing a size mismatch.
sankhya_status copy_vector(const std::vector<double>& source, double* destination, int count,
                           const char* what) {
  if (destination == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "destination is null");
  if (count < 0 || static_cast<std::size_t>(count) != source.size()) {
    return fail(SANKHYA_ERROR_ARGUMENT, std::string("wrong buffer size for ") + what +
                                            ": the solution has " +
                                            std::to_string(source.size()) + " entries, " +
                                            std::to_string(count) + " were offered");
  }
  for (std::size_t i = 0; i < source.size(); ++i) destination[i] = source[i];
  return ok();
}

/// Reject an option the registry does not have, or has with another type, BEFORE the typed
/// setter is reached.
///
/// This is not defensive style, it is a hard requirement of the boundary. The C++ typed
/// accessors treat an unknown name as a programmer error and call std::abort() - correct for
/// C++ code, where a typo is a bug the compiler nearly caught, and fatal here. A C caller
/// passing a mistyped option string would take down the host process, and the Python
/// bindings that will sit on this API would take the interpreter with them. Measured: the
/// test for this crashed with 0xC0000409 until the check existed.
sankhya_status check_option(const char* name, sankhya::OptionType wanted) {
  if (!sankhya::Options::exists(name)) {
    return fail(SANKHYA_ERROR_OPTION, std::string("unknown option '") + name +
                                          "'; run `sankhya options` for the list");
  }
  const sankhya::OptionSpec* spec = sankhya::Options::find_spec(name);
  if (spec != nullptr && spec->type != wanted) {
    return fail(SANKHYA_ERROR_OPTION,
                std::string("option '") + name + "' is not of the type this setter writes");
  }
  return SANKHYA_OK;
}

}  // namespace

// The handles. Defined here so the header can keep them opaque.
struct sankhya_model {
  sankhya::Model model;
  // (row, col) -> value, pending until the matrix is materialised.
  std::map<std::pair<int, int>, double> entries;
  std::map<std::pair<int, int>, double> quadratic;
};

struct sankhya_options {
  sankhya::Options options;
};

struct sankhya_solution {
  sankhya::Solution solution;
};

namespace {

/// Run `body`, converting any exception into a status code.
///
/// An exception crossing into C is undefined behaviour, so every escape route has to be
/// closed - including ones this file does not know about, hence the bare `catch (...)`.
///
/// A template rather than the macro this started as. A function-like macro splits its
/// argument on commas, so a body containing `entries[{row, col}]` arrives as two arguments
/// and the preprocessor rejects it - which it duly did, in four places.
template <typename Body>
sankhya_status guarded(Body&& body) {
  try {
    return body();
  } catch (const std::bad_alloc&) {
    return fail(SANKHYA_ERROR_MEMORY, "out of memory");
  } catch (const std::exception& error) {
    return fail(SANKHYA_ERROR_INTERNAL, error.what());
  } catch (...) {
    return fail(SANKHYA_ERROR_INTERNAL, "an unknown exception crossed the C API");
  }
}

}  // namespace

extern "C" {

// ---- Library -----------------------------------------------------------------------------

const char* sankhya_version(void) {
  static const std::string version = sankhya::version_string();
  return version.c_str();
}

const char* sankhya_last_error(void) {
  return g_error.c_str();
}

double sankhya_infinity(void) {
  return sankhya::kInfinity;
}

// ---- Model -------------------------------------------------------------------------------

sankhya_model* sankhya_model_create(void) {
  try {
    return new sankhya_model();
  } catch (...) {
    return nullptr;
  }
}

void sankhya_model_free(sankhya_model* model) {
  delete model;
}

sankhya_status sankhya_model_read(sankhya_model* model, const char* path) {
  if (model == nullptr || path == nullptr) {
    return fail(SANKHYA_ERROR_ARGUMENT, "model or path is null");
  }
  return guarded([&]() -> sankhya_status {
    sankhya::Model fresh;
    const sankhya::io::ReadResult result = sankhya::io::read_model(path, &fresh);
    if (!result.ok) return fail(SANKHYA_ERROR_IO, result.error);
    // Only on success. A half-populated handle after a failed read would be the worst of
    // both outcomes: the caller sees an error and still holds something solvable.
    model->model = std::move(fresh);
    model->entries.clear();
    model->quadratic.clear();
    return ok();
  });
}

sankhya_status sankhya_model_set_maximize(sankhya_model* model, int maximize) {
  if (model == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "model is null");
  model->model.sense =
      maximize != 0 ? sankhya::ObjSense::kMaximize : sankhya::ObjSense::kMinimize;
  return ok();
}

sankhya_status sankhya_model_set_objective_offset(sankhya_model* model, double offset) {
  if (model == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "model is null");
  model->model.objective_offset = offset;
  return ok();
}

sankhya_status sankhya_model_add_column(sankhya_model* model, double cost, double lower,
                                        double upper, int is_integer, const char* name,
                                        int* index) {
  if (model == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "model is null");
  return guarded([&]() -> sankhya_status {
    sankhya::Model& m = model->model;
    const int position = static_cast<int>(m.col_cost.size());
    m.col_cost.push_back(cost);
    m.col_lower.push_back(lower);
    m.col_upper.push_back(upper);
    m.col_type.push_back(is_integer != 0 ? sankhya::VarType::kInteger
                                         : sankhya::VarType::kContinuous);
    // Names are all-or-nothing: the solution writer and the verifier match on them, so a
    // model with names for only some columns would write a file neither can read back.
    // Columns added without one are given a positional name so the vector stays complete.
    m.col_names.push_back(name != nullptr ? std::string(name)
                                          : "C" + std::to_string(position + 1));
    if (index != nullptr) *index = position;
    return ok();
  });
}

sankhya_status sankhya_model_add_row(sankhya_model* model, double lower, double upper,
                                     const char* name, int* index) {
  if (model == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "model is null");
  return guarded([&]() -> sankhya_status {
    sankhya::Model& m = model->model;
    const int position = static_cast<int>(m.row_lower.size());
    m.row_lower.push_back(lower);
    m.row_upper.push_back(upper);
    m.row_names.push_back(name != nullptr ? std::string(name)
                                          : "R" + std::to_string(position + 1));
    if (index != nullptr) *index = position;
    return ok();
  });
}

sankhya_status sankhya_model_set_coefficient(sankhya_model* model, int row, int col,
                                             double value) {
  if (model == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "model is null");
  const sankhya::Model& m = model->model;
  if (row < 0 || row >= static_cast<int>(m.row_lower.size())) {
    return fail(SANKHYA_ERROR_ARGUMENT, "row " + std::to_string(row) +
                                            " is outside the model, which has " +
                                            std::to_string(m.row_lower.size()) + " row(s)");
  }
  if (col < 0 || col >= static_cast<int>(m.col_cost.size())) {
    return fail(SANKHYA_ERROR_ARGUMENT, "column " + std::to_string(col) +
                                            " is outside the model, which has " +
                                            std::to_string(m.col_cost.size()) + " column(s)");
  }
  return guarded([&]() -> sankhya_status {
    if (value == 0.0) {
      model->entries.erase({row, col});
    } else {
      model->entries[{row, col}] = value;
    }
    return ok();
  });
}

sankhya_status sankhya_model_set_quadratic_coefficient(sankhya_model* model, int row, int col,
                                                       double value) {
  if (model == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "model is null");
  const int columns = static_cast<int>(model->model.col_cost.size());
  if (row < 0 || row >= columns || col < 0 || col >= columns) {
    return fail(SANKHYA_ERROR_ARGUMENT, "quadratic index (" + std::to_string(row) + ", " +
                                            std::to_string(col) +
                                            ") is outside the model, which has " +
                                            std::to_string(columns) + " column(s)");
  }
  return guarded([&]() -> sankhya_status {
    // Normalised to the lower triangle, so (i, j) and (j, i) name one entry of the symmetric
    // Q rather than two. A caller giving both would otherwise set the same coefficient twice
    // and, on a naive implementation, double it.
    const int lower_row = row > col ? row : col;
    const int lower_col = row > col ? col : row;
    if (value == 0.0) {
      model->quadratic.erase({lower_row, lower_col});
    } else {
      model->quadratic[{lower_row, lower_col}] = value;
    }
    return ok();
  });
}

int sankhya_model_num_cols(const sankhya_model* model) {
  return model == nullptr ? 0 : static_cast<int>(model->model.col_cost.size());
}

int sankhya_model_num_rows(const sankhya_model* model) {
  return model == nullptr ? 0 : static_cast<int>(model->model.row_lower.size());
}

int sankhya_model_num_nonzeros(const sankhya_model* model) {
  if (model == nullptr) return 0;
  // Pending triplets when the caller built the model programmatically; the frozen matrix
  // when it was read from a file. Reporting only one of the two would make this function
  // answer a different question depending on how the handle was populated.
  const int pending = static_cast<int>(model->entries.size());
  return pending > 0 ? pending : static_cast<int>(model->model.matrix.num_nonzeros());
}

namespace {

/// Materialise the pending triplets into the Model's frozen matrices.
///
/// Called on a COPY at solve time rather than mutating the handle, so that a caller can
/// solve, add another column, and solve again without the first solve having frozen
/// anything underneath it.
void materialise(const sankhya_model& handle, sankhya::Model* out) {
  *out = handle.model;
  const auto rows = static_cast<sankhya::Index>(out->row_lower.size());
  const auto cols = static_cast<sankhya::Index>(out->col_cost.size());

  if (!handle.entries.empty() || out->matrix.num_nonzeros() == 0) {
    out->matrix.reset(rows, cols);
    out->matrix.reserve(handle.entries.size());
    for (const auto& entry : handle.entries) {
      out->matrix.add_entry(static_cast<sankhya::Index>(entry.first.first),
                            static_cast<sankhya::Index>(entry.first.second), entry.second);
    }
    out->matrix.finalize();
  }

  if (!handle.quadratic.empty()) {
    out->hessian.reset(cols, cols);
    out->hessian.reserve(handle.quadratic.size());
    for (const auto& entry : handle.quadratic) {
      out->hessian.add_entry(static_cast<sankhya::Index>(entry.first.first),
                             static_cast<sankhya::Index>(entry.first.second), entry.second);
    }
    out->hessian.finalize();
  }
}

}  // namespace

sankhya_status sankhya_model_validate(const sankhya_model* model) {
  if (model == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "model is null");
  return guarded([&]() -> sankhya_status {
    sankhya::Model built;
    materialise(*model, &built);
    const std::string problem = built.validate();
    if (!problem.empty()) return fail(SANKHYA_ERROR_MODEL, problem);
    return ok();
  });
}

// ---- Options -------------------------------------------------------------------------------

sankhya_options* sankhya_options_create(void) {
  try {
    return new sankhya_options();
  } catch (...) {
    return nullptr;
  }
}

void sankhya_options_free(sankhya_options* options) {
  delete options;
}

sankhya_status sankhya_options_set_bool(sankhya_options* options, const char* name, int value) {
  if (options == nullptr || name == nullptr) {
    return fail(SANKHYA_ERROR_ARGUMENT, "options or name is null");
  }
  const sankhya_status check = check_option(name, sankhya::OptionType::Bool);
  if (check != SANKHYA_OK) return check;
  return guarded([&]() -> sankhya_status {
    options->options.set_bool(name, value != 0);
    return ok();
  });
}

sankhya_status sankhya_options_set_int(sankhya_options* options, const char* name, long value) {
  if (options == nullptr || name == nullptr) {
    return fail(SANKHYA_ERROR_ARGUMENT, "options or name is null");
  }
  const sankhya_status check = check_option(name, sankhya::OptionType::Int);
  if (check != SANKHYA_OK) return check;
  return guarded([&]() -> sankhya_status {
    options->options.set_int(name, static_cast<std::int64_t>(value));
    return ok();
  });
}

sankhya_status sankhya_options_set_double(sankhya_options* options, const char* name,
                                          double value) {
  if (options == nullptr || name == nullptr) {
    return fail(SANKHYA_ERROR_ARGUMENT, "options or name is null");
  }
  const sankhya_status check = check_option(name, sankhya::OptionType::Double);
  if (check != SANKHYA_OK) return check;
  return guarded([&]() -> sankhya_status {
    options->options.set_double(name, value);
    return ok();
  });
}

sankhya_status sankhya_options_set_string(sankhya_options* options, const char* name,
                                          const char* value) {
  if (options == nullptr || name == nullptr || value == nullptr) {
    return fail(SANKHYA_ERROR_ARGUMENT, "options, name or value is null");
  }
  const sankhya_status check = check_option(name, sankhya::OptionType::String);
  if (check != SANKHYA_OK) return check;
  return guarded([&]() -> sankhya_status {
    options->options.set_string(name, value);
    return ok();
  });
}

// ---- Solve ---------------------------------------------------------------------------------

sankhya_status sankhya_solve(const sankhya_model* model, const sankhya_options* options,
                             sankhya_solution** solution) {
  if (model == nullptr || solution == nullptr) {
    return fail(SANKHYA_ERROR_ARGUMENT, "model or solution pointer is null");
  }
  *solution = nullptr;
  return guarded([&]() -> sankhya_status {
    sankhya::Model built;
    materialise(*model, &built);

    sankhya::Options effective;
    if (options != nullptr) effective = options->options;

    auto* result = new sankhya_solution();
    result->solution = sankhya::solve(built, effective);
    *solution = result;
    return ok();
  });
}

void sankhya_solution_free(sankhya_solution* solution) {
  delete solution;
}

sankhya_solve_status sankhya_solution_status(const sankhya_solution* solution) {
  return solution == nullptr ? SANKHYA_NOT_SOLVED : to_c_status(solution->solution.status);
}

const char* sankhya_solution_message(const sankhya_solution* solution) {
  return solution == nullptr ? "" : solution->solution.message.c_str();
}

double sankhya_solution_objective(const sankhya_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.objective;
}

double sankhya_solution_dual_bound(const sankhya_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.dual_bound;
}

long sankhya_solution_iterations(const sankhya_solution* solution) {
  return solution == nullptr ? 0 : static_cast<long>(solution->solution.iterations);
}

long sankhya_solution_nodes(const sankhya_solution* solution) {
  return solution == nullptr ? 0 : static_cast<long>(solution->solution.nodes);
}

double sankhya_solution_seconds(const sankhya_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.solve_seconds;
}

double sankhya_solution_primal_infeasibility(const sankhya_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.primal_infeasibility;
}

double sankhya_solution_dual_infeasibility(const sankhya_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.dual_infeasibility;
}

double sankhya_solution_integrality_violation(const sankhya_solution* solution) {
  return solution == nullptr ? 0.0 : solution->solution.integrality_violation;
}

sankhya_status sankhya_solution_col_values(const sankhya_solution* solution, double* values,
                                           int count) {
  if (solution == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "solution is null");
  return copy_vector(solution->solution.col_value, values, count, "column values");
}

sankhya_status sankhya_solution_row_activities(const sankhya_solution* solution, double* values,
                                               int count) {
  if (solution == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "solution is null");
  return copy_vector(solution->solution.row_activity, values, count, "row activities");
}

sankhya_status sankhya_solution_row_duals(const sankhya_solution* solution, double* values,
                                          int count) {
  if (solution == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "solution is null");
  return copy_vector(solution->solution.row_dual, values, count, "row duals");
}

sankhya_status sankhya_solution_col_duals(const sankhya_solution* solution, double* values,
                                          int count) {
  if (solution == nullptr) return fail(SANKHYA_ERROR_ARGUMENT, "solution is null");
  return copy_vector(solution->solution.col_dual, values, count, "column duals");
}

}  // extern "C"

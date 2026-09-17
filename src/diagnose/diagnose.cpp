// SPDX-License-Identifier: Apache-2.0
// SANKHYA - model diagnostics (#294). See diagnose.hpp for what this is and is not.

#include "diagnose.hpp"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "presolve/presolve.hpp"
#include "qp/convexity.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::diagnose {
namespace {

/// Where the spread of coefficient magnitudes stops being ordinary.
///
/// The thresholds are the ones this project's own evidence points at rather than round
/// numbers: the Netlib instances that needed the unscaled retry and the generated families in
/// docs/BENCHMARKS.md section 5 sit above 1e8, and double precision holds about sixteen
/// digits, so a ratio past 1e12 leaves a solve fewer than four digits of headroom before the
/// residual tests start deciding things.
constexpr double kMediumRatio = 1e8;
constexpr double kHighRatio = 1e12;

[[nodiscard]] bool is_binary(const Model& model, Index j) {
  const auto u = static_cast<std::size_t>(j);
  if (model.col_type[u] != VarType::kInteger) return false;
  return model.col_lower[u] >= -tol::kIntegrality &&
         model.col_upper[u] <= 1.0 + tol::kIntegrality;
}

}  // namespace

const char* to_string(Risk risk) noexcept {
  switch (risk) {
    case Risk::kLow: return "LOW";
    case Risk::kMedium: return "MEDIUM";
    case Risk::kHigh: return "HIGH";
  }
  return "UNKNOWN";
}

Diagnosis analyse(const Model& model, const Options& options, Logger& logger) {
  Diagnosis d;
  d.name = model.name.empty() ? "(unnamed)" : model.name;
  d.source_path = model.source_path;
  d.maximize = model.sense == ObjSense::kMaximize;
  d.rows = model.num_rows();
  d.columns = model.num_cols();
  d.nonzeros = model.num_nonzeros();
  d.objective_offset = model.objective_offset;

  // ---- columns -------------------------------------------------------------------------
  for (Index j = 0; j < d.columns; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const bool lo = is_finite_bound(model.col_lower[u]);
    const bool hi = is_finite_bound(model.col_upper[u]);
    if (model.col_type[u] == VarType::kInteger) {
      ++d.integer_columns;
      if (is_binary(model, j)) ++d.binary_columns;
    } else {
      ++d.continuous_columns;
    }
    if (lo && hi && model.col_lower[u] == model.col_upper[u]) {
      ++d.fixed_columns;
    } else if (lo && hi) {
      ++d.boxed_columns;
    } else if (!lo && !hi) {
      ++d.free_columns;
    }
    if (model.col_cost[u] != 0.0) {
      ++d.objective_nonzeros;
      d.largest_objective_coefficient =
          std::max(d.largest_objective_coefficient, std::fabs(model.col_cost[u]));
    }
    if (lo) d.largest_bound = std::max(d.largest_bound, std::fabs(model.col_lower[u]));
    if (hi) d.largest_bound = std::max(d.largest_bound, std::fabs(model.col_upper[u]));

    const ColumnView column = model.matrix.column(j);
    if (column.size == 0) ++d.columns_without_entries;
    d.densest_column_nonzeros = std::max(d.densest_column_nonzeros, column.size);
  }

  // ---- rows ----------------------------------------------------------------------------
  std::vector<Index> row_counts(static_cast<std::size_t>(d.rows), 0);
  for (Index j = 0; j < d.columns; ++j) {
    const ColumnView column = model.matrix.column(j);
    for (Index k = 0; k < column.size; ++k) {
      ++row_counts[static_cast<std::size_t>(column.rows[k])];
    }
  }
  for (Index i = 0; i < d.rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const bool lo = is_finite_bound(model.row_lower[u]);
    const bool hi = is_finite_bound(model.row_upper[u]);
    if (lo && hi && model.row_lower[u] == model.row_upper[u]) {
      ++d.equality_rows;
    } else if (lo && hi) {
      ++d.range_rows;
    } else if (hi) {
      ++d.less_than_rows;
    } else if (lo) {
      ++d.greater_than_rows;
    } else {
      ++d.free_rows;
    }
    if (lo) d.largest_bound = std::max(d.largest_bound, std::fabs(model.row_lower[u]));
    if (hi) d.largest_bound = std::max(d.largest_bound, std::fabs(model.row_upper[u]));
    if (row_counts[u] == 0) ++d.empty_rows;
    if (row_counts[u] == 1) ++d.singleton_rows;
    d.densest_row_nonzeros = std::max(d.densest_row_nonzeros, row_counts[u]);
  }

  // ---- sparsity ------------------------------------------------------------------------
  if (d.rows > 0 && d.columns > 0) {
    d.density_percent = 100.0 * static_cast<double>(d.nonzeros) /
                        (static_cast<double>(d.rows) * static_cast<double>(d.columns));
  }
  if (d.rows > 0) {
    d.average_nonzeros_per_row = static_cast<double>(d.nonzeros) / static_cast<double>(d.rows);
  }
  if (d.columns > 0) {
    d.average_nonzeros_per_column =
        static_cast<double>(d.nonzeros) / static_cast<double>(d.columns);
  }

  // ---- magnitudes ----------------------------------------------------------------------
  for (const double v : model.matrix.values()) {
    const double a = std::fabs(v);
    if (a == 0.0) continue;
    if (d.smallest_coefficient == 0.0 || a < d.smallest_coefficient) d.smallest_coefficient = a;
    d.largest_coefficient = std::max(d.largest_coefficient, a);
  }
  if (d.smallest_coefficient > 0.0) {
    d.coefficient_ratio = d.largest_coefficient / d.smallest_coefficient;
  }
  if (d.coefficient_ratio >= kHighRatio) {
    d.scaling_risk = Risk::kHigh;
    d.scaling_note = fmt::format(
        "coefficients span {:.1e}, which leaves under four decimal digits of the sixteen "
        "double precision carries; expect the scaled and unscaled solves to disagree",
        d.coefficient_ratio);
  } else if (d.coefficient_ratio >= kMediumRatio) {
    d.scaling_risk = Risk::kMedium;
    d.scaling_note = fmt::format(
        "coefficients span {:.1e}; equilibration handles this routinely, but a residual "
        "test near tolerance is worth a second look",
        d.coefficient_ratio);
  } else {
    d.scaling_note =
        "coefficient magnitudes are within a range double precision handles "
        "without help";
  }

  // ---- class ---------------------------------------------------------------------------
  // The same two properties solve() dispatches on, asked of the same model.
  const bool integral = model.has_integrality();
  const bool quadratic = model.has_quadratic_objective();
  d.problem_class = quadratic ? (integral ? "MIQP" : "QP") : (integral ? "MILP" : "LP");

  if (quadratic) {
    const qp::ConvexityResult convexity = qp::check_convexity(model);
    switch (convexity.verdict) {
      case qp::Convexity::kConvex: d.convexity = "convex"; break;
      case qp::Convexity::kIndefinite: d.convexity = "not convex"; break;
      case qp::Convexity::kUnverified: d.convexity = "unverified"; break;
    }
    d.convexity_detail = convexity.detail;
    if (convexity.verdict == qp::Convexity::kIndefinite) {
      d.warnings.push_back(
          "the quadratic objective is not convex, and this solver refuses such a model "
          "rather than returning a local point labelled optimal");
    } else if (convexity.verdict == qp::Convexity::kUnverified) {
      d.warnings.push_back("convexity could not be established, so the model will be refused");
    }
  }

  // ---- what presolve would do ----------------------------------------------------------
  if (options.get_bool("presolve")) {
    const presolve::Result reduced = presolve::presolve(model, options, logger);
    d.presolve_ran = true;
    d.presolve_proved_infeasible = reduced.proved_infeasible;
    d.presolve_message = reduced.message;
    if (!reduced.proved_infeasible) {
      d.presolve_rows_removed = reduced.rows_removed();
      d.presolve_columns_removed = reduced.cols_removed();
    }
    if (reduced.proved_infeasible) {
      d.warnings.push_back("presolve settles this model on its own: it has no feasible point");
    }
  }

  // ---- guidance ------------------------------------------------------------------------
  // What solve() would pick, and why, rather than a recommendation invented here.
  if (d.problem_class == "LP") {
    d.recommended_engine = "dual simplex (algorithm=auto)";
    d.engine_reason =
        "the measured default on the Netlib full set; exact, and it produces the basis that "
        "sensitivity ranging and warm starts need";
  } else if (d.problem_class == "MILP") {
    d.recommended_engine = "branch and cut";
    d.engine_reason = fmt::format(
        "{} integer columns ({} binary); node relaxations are warm-started dual simplex "
        "solves",
        d.integer_columns, d.binary_columns);
  } else if (d.problem_class == "QP") {
    d.recommended_engine = "Condat-Vu primal-dual";
    d.engine_reason = "the convex QP engine; the Hessian is tested for convexity first";
  } else {
    d.recommended_engine = "branch and bound over convex QP relaxations";
    d.engine_reason =
        "MIQP: the node bound comes from a first-order method, so pruning is deliberately "
        "conservative and the search costs more nodes than a MILP of the same size";
  }

  // The honest GPU line, which is the same one every document in this project carries.
  d.gpu_note =
      "no CUDA backend is compiled into this build; --gpu warns and runs on the CPU. The "
      "first-order engine (algorithm=pdhg) is the one a GPU would accelerate, and it is "
      "worth considering on a large sparse LP where an approximate answer arrives early";

  if (d.integer_columns > 0 && d.problem_class != "LP") {
    d.warnings.push_back(fmt::format(
        "MILP difficulty is not predictable from size: this model has {} integer columns, "
        "and the search may close in seconds or run out the clock",
        d.integer_columns));
  }
  if (d.free_rows > 0) {
    d.warnings.push_back(fmt::format(
        "{} row(s) have no bound on either side and constrain nothing", d.free_rows));
  }
  if (d.columns_without_entries > 0) {
    d.warnings.push_back(
        fmt::format("{} column(s) appear in no row at all", d.columns_without_entries));
  }

  // ---- what the model itself occupies --------------------------------------------------
  const auto entries = static_cast<std::size_t>(d.nonzeros);
  const auto hessian_entries = static_cast<std::size_t>(model.hessian.num_nonzeros());
  d.model_bytes = (entries + hessian_entries) * (sizeof(double) + sizeof(Index)) +
                  static_cast<std::size_t>(d.columns) * 4 * sizeof(double) +
                  static_cast<std::size_t>(d.rows) * 2 * sizeof(double);
  return d;
}

std::string format_text(const Diagnosis& d) {
  std::string out;
  const auto line = [&out](std::string text) { out += text + "\n"; };

  line(fmt::format("Model              {}", d.name));
  if (!d.source_path.empty()) line(fmt::format("Source             {}", d.source_path));
  line(fmt::format("Class              {}{}", d.problem_class,
                   d.convexity.empty() ? "" : fmt::format("  ({})", d.convexity)));
  line(fmt::format("Sense              {}", d.maximize ? "maximize" : "minimize"));
  line("");
  line("Size");
  line(fmt::format("  rows             {}", d.rows));
  line(fmt::format("  columns          {}", d.columns));
  line(fmt::format("  nonzeros         {}", d.nonzeros));
  line(fmt::format("  objective terms  {}", d.objective_nonzeros));
  line("");
  line("Columns");
  line(fmt::format("  continuous       {}", d.continuous_columns));
  line(fmt::format("  integer          {}  (binary {})", d.integer_columns, d.binary_columns));
  line(fmt::format("  fixed            {}", d.fixed_columns));
  line(fmt::format("  boxed            {}", d.boxed_columns));
  line(fmt::format("  free             {}", d.free_columns));
  if (d.columns_without_entries > 0) {
    line(fmt::format("  in no row        {}", d.columns_without_entries));
  }
  line("");
  line("Rows");
  line(fmt::format("  equality         {}", d.equality_rows));
  line(fmt::format("  <=               {}", d.less_than_rows));
  line(fmt::format("  >=               {}", d.greater_than_rows));
  line(fmt::format("  range            {}", d.range_rows));
  if (d.free_rows > 0) line(fmt::format("  free             {}", d.free_rows));
  if (d.empty_rows > 0) line(fmt::format("  empty            {}", d.empty_rows));
  if (d.singleton_rows > 0) line(fmt::format("  singleton        {}", d.singleton_rows));
  line("");
  line("Sparsity");
  line(fmt::format("  density          {:.4f}%", d.density_percent));
  line(fmt::format("  nonzeros / row   {:.2f} average, {} in the densest",
                   d.average_nonzeros_per_row, d.densest_row_nonzeros));
  line(fmt::format("  nonzeros / col   {:.2f} average, {} in the densest",
                   d.average_nonzeros_per_column, d.densest_column_nonzeros));
  line(fmt::format("  model storage    {:.2f} MB", static_cast<double>(d.model_bytes) / 1e6));
  line("");
  line("Numerics");
  if (d.largest_coefficient > 0.0) {
    line(fmt::format("  |coefficient|    [{:.3g}, {:.3g}], ratio {:.3g}",
                     d.smallest_coefficient, d.largest_coefficient, d.coefficient_ratio));
  } else {
    line("  |coefficient|    the matrix has no entries");
  }
  line(fmt::format("  largest bound    {:.3g}", d.largest_bound));
  line(fmt::format("  scaling risk     {}", to_string(d.scaling_risk)));
  line(fmt::format("                   {}", d.scaling_note));
  if (!d.convexity_detail.empty()) {
    line(fmt::format("  convexity        {}", d.convexity_detail));
  }
  line("");
  if (d.presolve_ran) {
    line("Presolve (run here, on a copy; the solve will repeat it)");
    if (d.presolve_proved_infeasible) {
      line(fmt::format("  verdict          infeasible: {}", d.presolve_message));
    } else {
      line(fmt::format("  rows removed     {} of {}", d.presolve_rows_removed, d.rows));
      line(fmt::format("  columns removed  {} of {}", d.presolve_columns_removed, d.columns));
    }
    line("");
  }
  line("Guidance");
  line(fmt::format("  engine           {}", d.recommended_engine));
  line(fmt::format("                   {}", d.engine_reason));
  line(fmt::format("  GPU              {}", d.gpu_note));
  if (!d.warnings.empty()) {
    line("");
    line("Worth knowing");
    for (const std::string& warning : d.warnings) line(fmt::format("  - {}", warning));
  }
  return out;
}

std::string format_json(const Diagnosis& d) {
  nlohmann::json blob;
  blob["model"] = {{"name", d.name},
                   {"source", d.source_path},
                   {"class", d.problem_class},
                   {"sense", d.maximize ? "maximize" : "minimize"},
                   {"rows", d.rows},
                   {"columns", d.columns},
                   {"nonzeros", d.nonzeros},
                   {"objective_terms", d.objective_nonzeros},
                   {"objective_offset", d.objective_offset},
                   {"storage_bytes", d.model_bytes}};
  blob["columns"] = {{"continuous", d.continuous_columns},
                     {"integer", d.integer_columns},
                     {"binary", d.binary_columns},
                     {"fixed", d.fixed_columns},
                     {"boxed", d.boxed_columns},
                     {"free", d.free_columns},
                     {"without_entries", d.columns_without_entries}};
  blob["rows"] = {{"equality", d.equality_rows},
                  {"less_than", d.less_than_rows},
                  {"greater_than", d.greater_than_rows},
                  {"range", d.range_rows},
                  {"free", d.free_rows},
                  {"empty", d.empty_rows},
                  {"singleton", d.singleton_rows}};
  blob["sparsity"] = {{"density_percent", d.density_percent},
                      {"average_nonzeros_per_row", d.average_nonzeros_per_row},
                      {"average_nonzeros_per_column", d.average_nonzeros_per_column},
                      {"densest_row", d.densest_row_nonzeros},
                      {"densest_column", d.densest_column_nonzeros}};
  blob["numerics"] = {{"smallest_coefficient", d.smallest_coefficient},
                      {"largest_coefficient", d.largest_coefficient},
                      {"coefficient_ratio", d.coefficient_ratio},
                      {"largest_bound", d.largest_bound},
                      {"largest_objective_coefficient", d.largest_objective_coefficient},
                      {"scaling_risk", to_string(d.scaling_risk)},
                      {"scaling_note", d.scaling_note}};
  if (!d.convexity.empty()) {
    blob["numerics"]["convexity"] = d.convexity;
    blob["numerics"]["convexity_detail"] = d.convexity_detail;
  }
  blob["presolve"] = {{"ran", d.presolve_ran},
                      {"rows_removed", d.presolve_rows_removed},
                      {"columns_removed", d.presolve_columns_removed},
                      {"proved_infeasible", d.presolve_proved_infeasible},
                      {"message", d.presolve_message}};
  blob["guidance"] = {{"engine", d.recommended_engine},
                      {"reason", d.engine_reason},
                      {"gpu", d.gpu_note},
                      {"warnings", d.warnings}};
  return blob.dump(2) + "\n";
}

}  // namespace sankhya::diagnose

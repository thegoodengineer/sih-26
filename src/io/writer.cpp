// SPDX-License-Identifier: Apache-2.0
// SANKHYA - solution and statistics writers.
//
// The solution file is a deliverable, not a debug dump. tools/verify_solution.py reads it,
// re-derives the model from the original .mps without touching our C++, and recomputes
// every quantity we claim. That only works if the file round-trips doubles exactly, so
// every number is written with {:.17g}: seventeen significant digits is the shortest width
// that is guaranteed to recover an IEEE-754 double bit-for-bit. Printing %.6f instead - the
// obvious choice, and the wrong one - would make the checker's recomputed objective differ
// from ours in the eighth digit and turn a correct solve into a spurious verification
// failure.
//
// Format is deliberately line-oriented and greppable rather than JSON: a judge reads this
// in a terminal. The JSON blob beside it is for the benchmark runners.

#include <cstdio>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/version.hpp"

namespace sankhya::io {
namespace {

/// Shortest exact decimal for a double. See the file header for why this is not negotiable.
[[nodiscard]] std::string exact(double v) {
  if (v == kInfinity) return "inf";
  if (v == -kInfinity) return "-inf";
  return fmt::format("{:.17g}", normalize_zero(v));
}

[[nodiscard]] std::string column_name(const Model& model, Index j) {
  const auto u = static_cast<std::size_t>(j);
  if (u < model.col_names.size() && !model.col_names[u].empty()) return model.col_names[u];
  return fmt::format("C{}", j);
}

[[nodiscard]] std::string row_name(const Model& model, Index i) {
  const auto u = static_cast<std::size_t>(i);
  if (u < model.row_names.size() && !model.row_names[u].empty()) return model.row_names[u];
  return fmt::format("R{}", i);
}

[[nodiscard]] double value_or(const std::vector<double>& v, Index i) {
  const auto u = static_cast<std::size_t>(i);
  return u < v.size() ? v[u] : 0.0;
}

[[nodiscard]] const char* status_or(const std::vector<BasisStatus>& v, Index i) {
  const auto u = static_cast<std::size_t>(i);
  return to_string(u < v.size() ? v[u] : BasisStatus::kUnknown);
}

}  // namespace

bool write_solution(const std::string& path, const Model& model, const Solution& solution,
                    std::string* error) {
  std::FILE* out = std::fopen(path.c_str(), "wb");
  if (out == nullptr) {
    if (error != nullptr) *error = fmt::format("{}: cannot open for writing", path);
    return false;
  }

  const Index n = model.num_cols();
  const Index m = model.num_rows();

  fmt::print(out, "# SANKHYA solution file\n");
  fmt::print(out, "# generated-by {}\n", banner());
  fmt::print(out, "# All numbers carry 17 significant digits and round-trip exactly.\n");
  fmt::print(out, "model {}\n", model.name.empty() ? "(unnamed)" : model.name);
  fmt::print(out, "source {}\n", model.source_path);
  fmt::print(out, "sense {}\n",
             model.sense == ObjSense::kMaximize ? "maximize" : "minimize");
  fmt::print(out, "status {}\n", to_string(solution.status));
  fmt::print(out, "algorithm {}\n",
             solution.algorithm.empty() ? "unknown" : solution.algorithm);
  fmt::print(out, "objective {}\n", exact(solution.objective));
  fmt::print(out, "dual_bound {}\n", exact(solution.dual_bound));
  fmt::print(out, "objective_offset {}\n", exact(model.objective_offset));
  fmt::print(out, "rows {}\n", m);
  fmt::print(out, "columns {}\n", n);
  fmt::print(out, "iterations {}\n", solution.iterations);
  fmt::print(out, "nodes {}\n", solution.nodes);
  fmt::print(out, "solve_seconds {}\n", exact(solution.solve_seconds));
  fmt::print(out, "primal_infeasibility {}\n", exact(solution.primal_infeasibility));
  fmt::print(out, "dual_infeasibility {}\n", exact(solution.dual_infeasibility));
  fmt::print(out, "integrality_violation {}\n", exact(solution.integrality_violation));
  if (!solution.message.empty()) fmt::print(out, "message {}\n", solution.message);

  fmt::print(out, "\n# name value reduced_cost basis_status\n");
  fmt::print(out, "begin columns {}\n", n);
  for (Index j = 0; j < n; ++j) {
    fmt::print(out, "{} {} {} {}\n", column_name(model, j), exact(value_or(solution.col_value, j)),
               exact(value_or(solution.col_dual, j)), status_or(solution.col_status, j));
  }
  fmt::print(out, "end columns\n");

  fmt::print(out, "\n# name activity dual basis_status\n");
  fmt::print(out, "begin rows {}\n", m);
  for (Index i = 0; i < m; ++i) {
    fmt::print(out, "{} {} {} {}\n", row_name(model, i), exact(value_or(solution.row_activity, i)),
               exact(value_or(solution.row_dual, i)), status_or(solution.row_status, i));
  }
  fmt::print(out, "end rows\n");

  const bool ok = std::fclose(out) == 0;
  if (!ok && error != nullptr) *error = fmt::format("{}: write failed", path);
  return ok;
}

bool write_stats_json(const std::string& path, const Model& model, const Solution& solution,
                      std::string* error) {
  // These key names are consumed by bench/runners/*.py. Renaming one silently breaks the
  // benchmark CSVs, which are the project's only evidence, so treat them as an interface.
  nlohmann::json blob;
  blob["sankhya"] = {{"version", version_string()},
                     {"commit", git_commit()},
                     {"build_type", build_type()},
                     {"compiler", compiler_string()},
                     {"cuda_enabled", cuda_enabled()}};
  blob["model"] = {{"name", model.name},
                   {"source", model.source_path},
                   {"sense", model.sense == ObjSense::kMaximize ? "maximize" : "minimize"},
                   {"rows", model.num_rows()},
                   {"columns", model.num_cols()},
                   {"nonzeros", model.num_nonzeros()},
                   {"integer_columns", model.num_integer_columns()},
                   {"objective_offset", model.objective_offset}};
  blob["result"] = {{"status", to_string(solution.status)},
                    {"algorithm", solution.algorithm},
                    {"objective", solution.objective},
                    {"dual_bound", solution.dual_bound},
                    {"absolute_gap", solution.absolute_gap},
                    {"relative_gap", solution.relative_gap},
                    {"message", solution.message}};
  blob["quality"] = {{"primal_infeasibility", solution.primal_infeasibility},
                     {"dual_infeasibility", solution.dual_infeasibility},
                     {"complementarity_violation", solution.complementarity_violation},
                     {"integrality_violation", solution.integrality_violation}};
  blob["effort"] = {{"iterations", solution.iterations},
                    {"nodes", solution.nodes},
                    {"cuts_applied", solution.cuts_applied},
                    {"solve_seconds", solution.solve_seconds}};

  std::FILE* out = std::fopen(path.c_str(), "wb");
  if (out == nullptr) {
    if (error != nullptr) *error = fmt::format("{}: cannot open for writing", path);
    return false;
  }
  const std::string text = blob.dump(2);
  const std::size_t written = std::fwrite(text.data(), 1, text.size(), out);
  std::fputc('\n', out);
  const bool ok = std::fclose(out) == 0 && written == text.size();
  if (!ok && error != nullptr) *error = fmt::format("{}: write failed", path);
  return ok;
}

}  // namespace sankhya::io

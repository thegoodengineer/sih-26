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

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/version.hpp"

namespace sankhya::io {
namespace {

/// Shortest exact decimal for a double. See the file header for why this is not negotiable.
[[nodiscard]] std::string exact(double v) {
  if (v == kInfinity) return "inf";
  if (v == -kInfinity) return "-inf";
  return fmt::format("{:.17g}", normalize_zero(v));
}

/// A double as a JSON value, keeping infinities and NaN recoverable.
///
/// JSON has no literal for infinity or NaN, and nlohmann's response is to serialise both as
/// `null` - silently, with no error and no warning, producing perfectly valid JSON. That is
/// exactly the wrong failure for us. An interrupted solve reports its dual bound as an
/// infinity on purpose, to say "nothing has been proven"; written as `null` that intent is
/// destroyed, and a runner doing float(blob["result"]["dual_bound"]) raises a TypeError on
/// NoneType, or worse treats the field as absent. From Phase 5 onward, stopping on a time
/// limit is the NORMAL outcome for a hard MILP, so this would hit precisely the runs whose
/// remaining gap is the number we most need to report.
///
/// Non-finite values therefore go out as the strings "inf", "-inf" and "nan", matching what
/// the .sol writer already emits. Python's float() accepts all three, so a consumer needs
/// no special case beyond calling float() on the field, which it must do anyway.
[[nodiscard]] nlohmann::json json_number(double v) {
  if (std::isfinite(v)) return normalize_zero(v);
  if (std::isnan(v)) return "nan";
  return v > 0.0 ? "inf" : "-inf";
}

/// Quote a name for the whitespace-delimited record format, if it needs it.
///
/// WHY THIS EXISTS. The columns and rows sections write `name value dual status` separated by
/// spaces. Fixed-format MPS permits names that CONTAIN spaces - Netlib's `forplan` has
/// columns called `DEDO3 11` and rows called `AZ 100` - and writing one of those bare makes
/// the record structurally ambiguous: nothing in `DEDO3 11 0 0.0246 at_lower` says whether
/// the name is one field or two. That is not a reader bug to work around, it is an output
/// format with an undecidable case, and tools/verify_solution.py consumes this file without
/// linking any of our code (ENGINEERING_RULES.md, "Frozen interfaces"), so both sides have to
/// agree on something actually parseable.
///
/// Quoting was chosen over fixed-width fields because it keeps the file readable by eye,
/// which is most of the point of this format, and it costs nothing on the overwhelmingly
/// common case: a name with no whitespace is written exactly as before, so every existing
/// .sol file is still byte-identical.
[[nodiscard]] std::string quoted_name(const std::string& name) {
  const bool needs_quotes =
      name.empty() || name.find_first_of(" \t\r\n\"\\") != std::string::npos;
  if (!needs_quotes) return name;

  std::string out;
  out.reserve(name.size() + 2);
  out.push_back('"');
  for (const char c : name) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  out.push_back('"');
  return out;
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
  return write_solution(path, model, solution, Options{}, error);
}

bool write_solution(const std::string& path, const Model& model, const Solution& solution,
                    const Options& options, std::string* error) {
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
  fmt::print(out, "sense {}\n", model.sense == ObjSense::kMaximize ? "maximize" : "minimize");
  fmt::print(out, "status {}\n", to_string(solution.status));
  fmt::print(out, "algorithm {}\n",
             solution.algorithm.empty() ? "unknown" : solution.algorithm);
  fmt::print(out, "objective {}\n", exact(solution.objective));
  fmt::print(out, "dual_bound {}\n", exact(solution.dual_bound));
  // The targets an `optimal` MILP was held to (#188). Written for every model so the header
  // has one shape; the verifier only reads them when there are integer columns.
  fmt::print(out, "mip_relative_gap {}\n", exact(options.get_double("mip_relative_gap")));
  fmt::print(out, "mip_absolute_gap {}\n", exact(options.get_double("mip_absolute_gap")));
  fmt::print(out, "objective_offset {}\n", exact(model.objective_offset));
  fmt::print(out, "certificate {}\n",
             !solution.farkas_dual.empty()  ? "farkas"
             : !solution.primal_ray.empty() ? "ray"
                                            : "none");
  if (!solution.iis_rows.empty() || !solution.iis_col_lo.empty() ||
      !solution.iis_col_hi.empty()) {
    // Whether the IIS section below claims both defining properties: infeasible on its own
    // (the farkas section is then its certificate) and irreducible (one witness per
    // element follows it). `not-claimed` when a trial solve was inconclusive.
    fmt::print(out, "iis_irreducible {}\n", solution.iis_inconclusive ? "not-claimed" : "yes");
  }
  if (!solution.col_ranging_lower.empty()) {
    fmt::print(out, "ranging_basis {}\n",
               solution.ranging_basis_degenerate ? "degenerate" : "nondegenerate");
  }
  fmt::print(out, "rows {}\n", m);
  fmt::print(out, "columns {}\n", n);
  fmt::print(out, "iterations {}\n", solution.iterations);
  fmt::print(out, "nodes {}\n", solution.nodes);
  fmt::print(out, "solve_seconds {}\n", exact(solution.solve_seconds));
  fmt::print(out, "primal_infeasibility {}\n", exact(solution.primal_infeasibility));
  fmt::print(out, "dual_infeasibility {}\n", exact(solution.dual_infeasibility));
  fmt::print(out, "integrality_violation {}\n", exact(solution.integrality_violation));
  if (!solution.message.empty()) fmt::print(out, "message {}\n", solution.message);

  // A VERDICT WITH NO POINT DOES NOT GET A POINT (#191, generalised in #200). `infeasible`
  // is a statement about the whole feasible region, and this file used to answer it with a
  // full all-zero columns and rows block, indistinguishable from a claimed solution. Our own
  // independent checker then read that point, found it violated the rows, and printed
  // REJECTED at a correct answer.
  //
  // #191 fixed that for `infeasible` and named it directly, so every other verdict with
  // nothing to show kept the bug: a numerical failure still wrote a full point and was still
  // rejected. The question is asked of claims_a_point() now, in one place, so there is no
  // second list to forget. What is written instead is the proof, when the engine had one.
  // An UNBOUNDED claim keeps its columns block, because its ray starts from a feasible point
  // and both halves are needed to check it.
  if (!claims_a_point(solution)) {
    if (!solution.farkas_dual.empty()) {
      fmt::print(out,
                 "\n# Farkas certificate: one multiplier per row. Aggregating the rows\n"
                 "# with these weights gives an inequality that no point inside the\n"
                 "# column bounds can satisfy. tools/verify_solution.py recomputes both\n"
                 "# sides.\n");
      fmt::print(out, "begin farkas {}\n", m);
      for (Index i = 0; i < m; ++i) {
        fmt::print(out, "{} {}\n", quoted_name(row_name(model, i)),
                   exact(value_or(solution.farkas_dual, i)));
      }
      fmt::print(out, "end farkas\n");
    }

    // IIS section (#217). Lists the irreducible infeasible subsystem by name so that the
    // diagnosis is human-readable without writing row indices that shift when a model is
    // edited. Format: each line is `kind name` where kind is row, col_lo or col_hi.
    if (!solution.iis_rows.empty() || !solution.iis_col_lo.empty() ||
        !solution.iis_col_hi.empty()) {
      const Index iis_count = static_cast<Index>(solution.iis_rows.size()) +
                              static_cast<Index>(solution.iis_col_lo.size()) +
                              static_cast<Index>(solution.iis_col_hi.size());
      fmt::print(out,
                 "\n# Irreducible infeasible subsystem: the smallest set of constraints\n"
                 "# that cannot all hold. Removing any one of them makes the model feasible.\n"
                 "# Computed by the Chinneck-Dravnieks deletion filter (#217).\n");
      fmt::print(out, "begin iis {}\n", iis_count);
      for (const Index i : solution.iis_rows) {
        fmt::print(out, "row {}\n", quoted_name(row_name(model, i)));
      }
      for (const Index j : solution.iis_col_lo) {
        fmt::print(out, "col_lo {}\n", quoted_name(column_name(model, j)));
      }
      for (const Index j : solution.iis_col_hi) {
        fmt::print(out, "col_hi {}\n", quoted_name(column_name(model, j)));
      }
      fmt::print(out, "end iis\n");

      // WITNESSES (#217): one point per element, satisfying every other element of the IIS
      // and violating its own, so the checker can confirm irreducibility by arithmetic. Only
      // the columns the IIS touches are written - the others enter no IIS row and carry no
      // IIS bound, so the checker treats them as zero without loss.
      if (!solution.iis_witnesses.empty()) {
        std::vector<char> touched(static_cast<std::size_t>(n), 0);
        std::vector<char> iis_row(static_cast<std::size_t>(m), 0);
        for (const Index i : solution.iis_rows) iis_row[static_cast<std::size_t>(i)] = 1;
        for (Index j = 0; j < n; ++j) {
          const ColumnView column = model.matrix.column(j);
          for (Index k = 0; k < column.size; ++k) {
            if (iis_row[static_cast<std::size_t>(column.rows[k])] != 0) {
              touched[static_cast<std::size_t>(j)] = 1;
              break;
            }
          }
        }
        for (const Index j : solution.iis_col_lo) touched[static_cast<std::size_t>(j)] = 1;
        for (const Index j : solution.iis_col_hi) touched[static_cast<std::size_t>(j)] = 1;
        Index touched_count = 0;
        for (const char t : touched) touched_count += t;

        std::size_t w = 0;
        const auto write_witness = [&](const char* kind, const std::string& name) {
          if (w >= solution.iis_witnesses.size()) return;
          const std::vector<double>& point = solution.iis_witnesses[w++];
          fmt::print(out, "begin iis_witness {} {} {}\n", kind, quoted_name(name),
                     touched_count);
          for (Index j = 0; j < n; ++j) {
            if (touched[static_cast<std::size_t>(j)] == 0) continue;
            fmt::print(out, "{} {}\n", quoted_name(column_name(model, j)),
                       exact(value_or(point, j)));
          }
          fmt::print(out, "end iis_witness\n");
        };
        fmt::print(out,
                   "\n# One witness per IIS element: a point that satisfies every other\n"
                   "# element and violates the named one, which is what irreducible means.\n");
        for (const Index i : solution.iis_rows) write_witness("row", row_name(model, i));
        for (const Index j : solution.iis_col_lo)
          write_witness("col_lo", column_name(model, j));
        for (const Index j : solution.iis_col_hi)
          write_witness("col_hi", column_name(model, j));
      }
    }

    const bool closed = std::fclose(out) == 0;
    if (!closed && error != nullptr) *error = fmt::format("{}: write failed", path);
    return closed;
  }

  fmt::print(out, "\n# name value reduced_cost basis_status\n");
  fmt::print(out, "begin columns {}\n", n);
  for (Index j = 0; j < n; ++j) {
    fmt::print(out, "{} {} {} {}\n", quoted_name(column_name(model, j)),
               exact(value_or(solution.col_value, j)), exact(value_or(solution.col_dual, j)),
               status_or(solution.col_status, j));
  }
  fmt::print(out, "end columns\n");

  fmt::print(out, "\n# name activity dual basis_status\n");
  fmt::print(out, "begin rows {}\n", m);
  for (Index i = 0; i < m; ++i) {
    fmt::print(out, "{} {} {} {}\n", quoted_name(row_name(model, i)),
               exact(value_or(solution.row_activity, i)), exact(value_or(solution.row_dual, i)),
               status_or(solution.row_status, i));
  }
  fmt::print(out, "end rows\n");

  // The solution pool (#225). By default INTEGER COLUMNS ONLY: the continuous values follow
  // from re-solving the LP with the integers fixed, and ten full vectors of a 100,000-column
  // model is a file nobody reads. `pool_write_all_columns` writes every column, so the
  // verifier can check each member's rows exactly. The first member is the solution above.
  if (!solution.pool.empty() && claims_a_point(solution.status)) {
    const bool all_columns = options.get_bool("pool_write_all_columns");
    std::vector<Index> integers;
    for (Index j = 0; j < n; ++j) {
      if (all_columns || model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger)
        integers.push_back(j);
    }
    fmt::print(out,
               "\n# Solution pool (#225): integer-feasible solutions with different integer\n"
               "# assignments, the one above first and the rest best first. {}\n"
               "# solution rank objective, then: name value\n",
               all_columns ? "Every column is written (pool_write_all_columns)."
                           : "Only the integer\n# columns are written; the continuous ones "
                             "follow from the LP with these fixed.");
    fmt::print(out, "begin pool {} {}\n", solution.pool.size(), integers.size());
    for (std::size_t k = 0; k < solution.pool.size(); ++k) {
      const Solution::PoolEntry& member = solution.pool[k];
      fmt::print(out, "solution {} {}\n", k + 1, exact(member.objective));
      for (const Index j : integers) {
        fmt::print(out, "{} {}\n", quoted_name(column_name(model, j)),
                   exact(value_or(member.col_value, j)));
      }
    }
    fmt::print(out, "end pool\n");
  }

  // Sensitivity ranging: objective and RHS intervals for each column and row.
  if (!solution.col_ranging_lower.empty() &&
      solution.col_ranging_lower.size() == static_cast<std::size_t>(n) &&
      solution.row_ranging_lower.size() == static_cast<std::size_t>(m)) {
    fmt::print(out,
               "\n# Sensitivity ranging (#220), in the model's own sense: how far each cost\n"
               "# coefficient may fall and rise before the optimal basis changes.\n"
               "# name allow_decrease allow_increase\n");
    fmt::print(out, "begin ranging_columns {}\n", n);
    for (Index j = 0; j < n; ++j) {
      const auto jj = static_cast<std::size_t>(j);
      fmt::print(out, "{} {} {}\n", quoted_name(column_name(model, j)),
                 exact(solution.col_ranging_lower[jj]), exact(solution.col_ranging_upper[jj]));
    }
    fmt::print(out, "end ranging_columns\n");
    fmt::print(out,
               "\n# How far each row's active bound may fall and rise before the basis\n"
               "# becomes infeasible; for a row that is not binding, how far its upper\n"
               "# bound may fall and its lower bound rise before it binds.\n"
               "# name allow_decrease allow_increase\n");
    fmt::print(out, "begin ranging_rows {}\n", m);
    for (Index i = 0; i < m; ++i) {
      const auto ii = static_cast<std::size_t>(i);
      fmt::print(out, "{} {} {}\n", quoted_name(row_name(model, i)),
                 exact(solution.row_ranging_lower[ii]), exact(solution.row_ranging_upper[ii]));
    }
    fmt::print(out, "end ranging_rows\n");
  }

  // The ray, read together with the point above: x + t*d stays feasible for every
  // t >= 0 and the objective improves without limit along it (#191).
  if (solution.status == SolveStatus::kUnbounded && !solution.primal_ray.empty()) {
    fmt::print(out,
               "\n# Unboundedness certificate: a direction no bound blocks, along\n"
               "# which the objective improves forever, starting from the feasible point\n"
               "# above.\n");
    fmt::print(out, "begin ray {}\n", n);
    for (Index j = 0; j < n; ++j) {
      fmt::print(out, "{} {}\n", quoted_name(column_name(model, j)),
                 exact(value_or(solution.primal_ray, j)));
    }
    fmt::print(out, "end ray\n");
  }

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
                   {"objective_offset", json_number(model.objective_offset)}};
  blob["result"] = {{"status", to_string(solution.status)},
                    {"algorithm", solution.algorithm},
                    {"objective", json_number(solution.objective)},
                    {"dual_bound", json_number(solution.dual_bound)},
                    {"absolute_gap", json_number(solution.absolute_gap)},
                    {"relative_gap", json_number(solution.relative_gap)},
                    {"message", solution.message}};
  blob["quality"] = {
      {"primal_infeasibility", json_number(solution.primal_infeasibility)},
      {"dual_infeasibility", json_number(solution.dual_infeasibility)},
      {"complementarity_violation", json_number(solution.complementarity_violation)},
      {"integrality_violation", json_number(solution.integrality_violation)}};
  blob["effort"] = {
      {"iterations", solution.iterations},
      {"refinement_steps", solution.refinement_steps},
      {"residual_before_refinement", json_number(solution.residual_before_refinement)},
      {"residual_after_refinement", json_number(solution.residual_after_refinement)},
      {"nodes", solution.nodes},
      {"cuts_applied", solution.cuts_applied},
      {"polish_iterations", solution.polish_iterations},
      {"solve_seconds", json_number(solution.solve_seconds)}};

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

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - solution and statistics writer tests.
//
// THE .sol FORMAT IS AN INTERFACE, not a debug dump. tools/verify_solution.py (Phase 3) is
// an independent checker that never links our C++: it re-reads the .mps, recomputes every
// quantity, and compares against what we wrote. That comparison is only meaningful if the
// file round-trips a double EXACTLY. If we print 17 significant digits and the checker
// recovers a value differing in the last bit, its recomputed objective differs from ours in
// the eighth digit and a correct solve is reported as a verification failure - and the
// natural "fix" is to loosen the checker's tolerance, which quietly destroys the only
// independent guarantee in the project.
//
// So the central test here is not "did it write a file". It is: parse every number back
// with the same routine a Python checker would use, and require bit-for-bit equality.
//
// The key names are equally an interface. bench/runners/*.py parse the JSON blob, and a
// renamed key silently breaks the benchmark CSVs, which are the project's only evidence.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using testing::TempFile;

/// Read a whole file back as text.
[[nodiscard]] std::string slurp(const std::string& path) {
  std::FILE* in = std::fopen(path.c_str(), "rb");
  if (in == nullptr) return {};
  std::string text;
  char buffer[4096];
  std::size_t got = 0;
  while ((got = std::fread(buffer, 1, sizeof(buffer), in)) > 0) text.append(buffer, got);
  std::fclose(in);
  return text;
}

/// The scalar header lines of a .sol file, as "key -> value".
[[nodiscard]] std::map<std::string, std::string> header_fields(const std::string& text) {
  std::map<std::string, std::string> fields;
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (line.rfind("begin ", 0) == 0) break;
    const std::size_t space = line.find(' ');
    if (space == std::string::npos) continue;
    fields[line.substr(0, space)] = line.substr(space + 1);
  }
  return fields;
}

struct Entry {
  double value = 0.0;
  double dual = 0.0;
  std::string status;
};

/// Parse one of the two tables. `section` is "columns" or "rows".
[[nodiscard]] std::map<std::string, Entry> table(const std::string& text,
                                                 const std::string& section) {
  std::map<std::string, Entry> rows;
  std::istringstream stream(text);
  std::string line;
  bool inside = false;
  while (std::getline(stream, line)) {
    if (line.rfind("begin " + section, 0) == 0) {
      inside = true;
      continue;
    }
    if (line.rfind("end " + section, 0) == 0) break;
    if (!inside || line.empty() || line[0] == '#') continue;

    std::istringstream fields(line);
    std::string name;
    std::string value;
    std::string dual;
    std::string status;
    fields >> name >> value >> dual >> status;
    Entry entry;
    // strtod is what a Python float() or a C checker would do. Using it here rather than
    // any bespoke parser is the point: it proves the file is recoverable by an outsider.
    entry.value = std::strtod(value.c_str(), nullptr);
    entry.dual = std::strtod(dual.c_str(), nullptr);
    entry.status = status;
    rows[name] = entry;
  }
  return rows;
}

/// A small LP with a deliberately irrational-looking optimum, so that the round-trip test
/// is exercised on values with a full mantissa rather than on tidy integers.
Model make_model() {
  Model model;
  model.name = "WRITERTEST";
  model.source_path = "in-memory";
  model.sense = ObjSense::kMaximize;
  model.objective_offset = 2.5;
  model.col_cost = {2.4, 1.6, 1.64};
  model.col_lower = {10.0, 0.0, 0.0};
  model.col_upper = {kInfinity, 45.0, 60.0};
  model.col_type = {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous};
  model.col_names = {"AL", "BN", "MU"};
  model.row_lower = {90.0, 40.0, -kInfinity};
  model.row_upper = {120.0, 40.0, 0.0};
  model.row_names = {"THRUPUT", "DIESEL", "SULPHUR"};

  model.matrix.reset(3, 3);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.add_entry(0, 2, 1.0);
  model.matrix.add_entry(1, 0, 0.30);
  model.matrix.add_entry(1, 1, 0.45);
  model.matrix.add_entry(1, 2, 0.38);
  model.matrix.add_entry(2, 0, 0.80);
  model.matrix.add_entry(2, 1, -0.86);
  model.matrix.add_entry(2, 2, -0.22);
  model.matrix.finalize();
  model.hessian.reset(3, 3);
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

Solution solve_it(const Model& model) {
  Options options;
  options.set_bool("log_to_console", false);
  return solve(model, options);
}

// =========================================================================================

TEST(SolutionWriter, EveryNumberRoundTripsBitForBit) {
  const Model model = make_model();
  const Solution solution = solve_it(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;

  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, solution, &error)) << error;

  const std::string text = slurp(file.path());
  ASSERT_FALSE(text.empty());

  const std::map<std::string, Entry> columns = table(text, "columns");
  const std::map<std::string, Entry> rows = table(text, "rows");
  ASSERT_EQ(columns.size(), 3u);
  ASSERT_EQ(rows.size(), 3u);

  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const auto it = columns.find(model.col_names[u]);
    ASSERT_NE(it, columns.end()) << "column " << model.col_names[u] << " is missing";
    // DOUBLE_EQ, not NEAR. Anything looser and the file has silently lost precision that
    // the independent checker depends on.
    EXPECT_DOUBLE_EQ(it->second.value, solution.col_value[u]) << model.col_names[u];
    EXPECT_DOUBLE_EQ(it->second.dual, solution.col_dual[u]) << model.col_names[u];
    EXPECT_EQ(it->second.status, to_string(solution.col_status[u]));
  }

  for (Index i = 0; i < model.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    const auto it = rows.find(model.row_names[u]);
    ASSERT_NE(it, rows.end()) << "row " << model.row_names[u] << " is missing";
    EXPECT_DOUBLE_EQ(it->second.value, solution.row_activity[u]) << model.row_names[u];
    EXPECT_DOUBLE_EQ(it->second.dual, solution.row_dual[u]) << model.row_names[u];
    EXPECT_EQ(it->second.status, to_string(solution.row_status[u]));
  }
}

TEST(SolutionWriter, TheObjectiveRecomputesFromTheWrittenValuesAlone) {
  // Exactly what tools/verify_solution.py will do in Phase 3: take only the numbers in the
  // file, re-evaluate the objective through the model, and require agreement far tighter
  // than feasibility. A solve is only verifiable if this holds.
  const Model model = make_model();
  const Solution solution = solve_it(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;

  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, solution, &error)) << error;

  const std::map<std::string, Entry> columns = table(slurp(file.path()), "columns");
  std::vector<double> x(static_cast<std::size_t>(model.num_cols()), 0.0);
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    x[u] = columns.at(model.col_names[u]).value;
  }

  const double recomputed = model.evaluate_objective(x.data());
  const std::map<std::string, std::string> fields = header_fields(slurp(file.path()));
  const double written = std::strtod(fields.at("objective").c_str(), nullptr);

  EXPECT_DOUBLE_EQ(recomputed, solution.objective);
  EXPECT_DOUBLE_EQ(written, solution.objective);
}

TEST(SolutionWriter, HeaderCarriesTheFieldsTheCheckerNeeds) {
  const Model model = make_model();
  const Solution solution = solve_it(model);
  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, solution, &error)) << error;

  const std::map<std::string, std::string> fields = header_fields(slurp(file.path()));
  for (const char* key :
       {"model", "sense", "status", "algorithm", "objective", "dual_bound", "objective_offset",
        "rows", "columns", "iterations", "primal_infeasibility", "dual_infeasibility"}) {
    EXPECT_NE(fields.find(key), fields.end())
        << "the .sol header lost the '" << key << "' field";
  }
  EXPECT_EQ(fields.at("status"), "optimal");
  EXPECT_EQ(fields.at("sense"), "maximize");
  EXPECT_EQ(fields.at("algorithm"), "simplex-primal");
  EXPECT_EQ(fields.at("rows"), "3");
  EXPECT_EQ(fields.at("columns"), "3");
  // The offset is part of the objective the checker recomputes, so it has to be recoverable.
  EXPECT_DOUBLE_EQ(std::strtod(fields.at("objective_offset").c_str(), nullptr), 2.5);
}

TEST(SolutionWriter, NegativeZeroIsNeverPrinted) {
  // A maximization model multiplies zero reduced costs by -1. "-0" in a solution file is
  // numerically identical to 0 but a reader cannot tell it from a real negative quantity
  // that rounded away.
  const Model model = make_model();
  const Solution solution = solve_it(model);
  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, solution, &error)) << error;

  const std::string text = slurp(file.path());
  EXPECT_EQ(text.find(" -0\n"), std::string::npos) << "a bare -0 reached the solution file";
  EXPECT_EQ(text.find(" -0 "), std::string::npos) << "a bare -0 reached the solution file";
}

TEST(SolutionWriter, InfiniteBoundsArePrintedAsWords) {
  // A raw "inf"/"-inf" is what Python's float() accepts; printing 1e30 instead would make
  // the checker treat an absent bound as a real one.
  Model model = make_model();
  model.objective_offset = kInfinity;
  Solution solution;
  solution.allocate_for(model);
  solution.status = SolveStatus::kOptimal;

  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, solution, &error)) << error;
  EXPECT_EQ(header_fields(slurp(file.path())).at("objective_offset"), "inf");
}

TEST(SolutionWriter, ReportsAnUnwritablePath) {
  const Model model = make_model();
  const Solution solution = solve_it(model);
  std::string error;
  EXPECT_FALSE(io::write_solution("no_such_directory/deeper/out.sol", model, solution, &error));
  EXPECT_FALSE(error.empty());
}

// =========================================================================================
// Names containing whitespace (issue #86)
// =========================================================================================

/// Split a record the way tools/verify_solution.py does: an optionally quoted leading name,
/// then whitespace-delimited fields. Written here rather than shared with the writer on
/// purpose - a round-trip test that uses the writer to parse proves only self-consistency.
[[nodiscard]] std::vector<std::string> split_record(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t i = 0;
  if (!line.empty() && line[0] == '"') {
    std::string name;
    ++i;
    while (i < line.size()) {
      if (line[i] == '\\' && i + 1 < line.size()) {
        name.push_back(line[i + 1]);
        i += 2;
        continue;
      }
      if (line[i] == '"') {
        ++i;
        break;
      }
      name.push_back(line[i]);
      ++i;
    }
    fields.push_back(name);
  }
  std::istringstream rest(line.substr(i));
  std::string token;
  while (rest >> token) fields.push_back(token);
  return fields;
}

/// The record for `name` from one section, unparsed.
[[nodiscard]] std::string record_for(const std::string& text, const std::string& section,
                                     const std::string& needle) {
  std::istringstream stream(text);
  std::string line;
  bool inside = false;
  while (std::getline(stream, line)) {
    if (line.rfind("begin " + section, 0) == 0) {
      inside = true;
      continue;
    }
    if (line.rfind("end " + section, 0) == 0) break;
    if (inside && line.find(needle) != std::string::npos) return line;
  }
  return {};
}

TEST(SolutionWriter, NamesContainingWhitespaceAreQuotedAndRecoverExactly) {
  // Fixed-format MPS permits names with spaces - Netlib forplan has a column `DEDO3 11`
  // and a row `AZ 100`. Written bare into a whitespace-delimited record, nothing says
  // whether the name is one field or two, and the file becomes undecidable for ANY reader,
  // including the independent verifier the whole evidence story rests on.
  Model model = make_model();
  model.col_names = {"AL PRIME", "BN", "MU"};
  model.row_names = {"THRUPUT 1", "DIESEL", "SULPHUR"};
  const Solution solution = solve_it(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;

  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, solution, &error)) << error;
  const std::string text = slurp(file.path());

  const std::string column = record_for(text, "columns", "AL PRIME");
  ASSERT_FALSE(column.empty()) << text;
  const std::vector<std::string> column_fields = split_record(column);
  ASSERT_EQ(column_fields.size(), 4u) << column;
  EXPECT_EQ(column_fields[0], "AL PRIME") << column;
  EXPECT_DOUBLE_EQ(std::strtod(column_fields[1].c_str(), nullptr), solution.col_value[0]);

  const std::string row = record_for(text, "rows", "THRUPUT 1");
  ASSERT_FALSE(row.empty()) << text;
  const std::vector<std::string> row_fields = split_record(row);
  ASSERT_EQ(row_fields.size(), 4u) << row;
  EXPECT_EQ(row_fields[0], "THRUPUT 1") << row;
  EXPECT_DOUBLE_EQ(std::strtod(row_fields[1].c_str(), nullptr), solution.row_activity[0]);
}

TEST(SolutionWriter, OrdinaryNamesAreNotQuoted) {
  // No gratuitous format churn: every .sol file written before #86 must still be written
  // byte for byte the same way, or the change breaks readers to fix one instance.
  const Model model = make_model();
  const Solution solution = solve_it(model);
  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, solution, &error)) << error;
  const std::string text = slurp(file.path());

  const std::string column = record_for(text, "columns", "AL");
  ASSERT_FALSE(column.empty());
  EXPECT_EQ(column.find('"'), std::string::npos)
      << "a name with no whitespace must be written bare: " << column;
  EXPECT_EQ(split_record(column)[0], "AL");
}

// =========================================================================================
// The JSON blob. Key names are consumed by bench/runners/*.py.
// =========================================================================================

TEST(StatsWriter, CarriesTheKeysTheBenchmarkRunnersParse) {
  const Model model = make_model();
  const Solution solution = solve_it(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;

  const TempFile file("", ".json");
  std::string error;
  ASSERT_TRUE(io::write_stats_json(file.path(), model, solution, &error)) << error;

  const nlohmann::json blob = nlohmann::json::parse(slurp(file.path()));

  ASSERT_TRUE(blob.contains("sankhya"));
  for (const char* key : {"version", "commit", "build_type", "compiler", "cuda_enabled"}) {
    EXPECT_TRUE(blob["sankhya"].contains(key)) << "sankhya." << key;
  }
  // CLAUDE.md requires the commit in every benchmark record, so an empty one is a defect.
  EXPECT_FALSE(blob["sankhya"]["commit"].get<std::string>().empty());

  ASSERT_TRUE(blob.contains("model"));
  for (const char* key : {"name", "source", "sense", "rows", "columns", "nonzeros",
                          "integer_columns", "objective_offset"}) {
    EXPECT_TRUE(blob["model"].contains(key)) << "model." << key;
  }
  EXPECT_EQ(blob["model"]["rows"].get<int>(), 3);
  EXPECT_EQ(blob["model"]["columns"].get<int>(), 3);
  EXPECT_EQ(blob["model"]["nonzeros"].get<int>(), 9);

  ASSERT_TRUE(blob.contains("result"));
  for (const char* key : {"status", "algorithm", "objective", "dual_bound", "absolute_gap",
                          "relative_gap", "message"}) {
    EXPECT_TRUE(blob["result"].contains(key)) << "result." << key;
  }
  EXPECT_EQ(blob["result"]["status"].get<std::string>(), "optimal");

  ASSERT_TRUE(blob.contains("quality"));
  for (const char* key : {"primal_infeasibility", "dual_infeasibility",
                          "complementarity_violation", "integrality_violation"}) {
    EXPECT_TRUE(blob["quality"].contains(key)) << "quality." << key;
  }

  ASSERT_TRUE(blob.contains("effort"));
  for (const char* key : {"iterations", "nodes", "cuts_applied", "solve_seconds"}) {
    EXPECT_TRUE(blob["effort"].contains(key)) << "effort." << key;
  }
}

TEST(StatsWriter, ObjectiveSurvivesTheJsonRoundTrip) {
  const Model model = make_model();
  const Solution solution = solve_it(model);
  const TempFile file("", ".json");
  std::string error;
  ASSERT_TRUE(io::write_stats_json(file.path(), model, solution, &error)) << error;

  const nlohmann::json blob = nlohmann::json::parse(slurp(file.path()));
  EXPECT_DOUBLE_EQ(blob["result"]["objective"].get<double>(), solution.objective);
  EXPECT_DOUBLE_EQ(blob["result"]["dual_bound"].get<double>(), solution.dual_bound);
}

TEST(StatsWriter, NonFiniteNumbersStayRecoverable) {
  // JSON has no literal for infinity, and nlohmann's answer is to emit `null` - silently,
  // with no error, producing valid JSON. An interrupted solve sets its dual bound to an
  // infinity ON PURPOSE, to say that nothing has been proven. Serialised as null that
  // meaning is destroyed and a runner calling float() on the field raises TypeError.
  //
  // This is not an edge case from Phase 5 onward: stopping on a time limit is the NORMAL
  // outcome for a hard MILP, so it would hit exactly the runs whose remaining gap is the
  // number we most need to publish.
  const Model model = make_model();
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("iteration_limit", 1);
  const Solution solution = solve(model, options);
  ASSERT_EQ(solution.status, SolveStatus::kIterationLimit) << solution.message;
  ASSERT_TRUE(std::isinf(solution.dual_bound)) << "precondition: the bound should be infinite";

  const TempFile file("", ".json");
  std::string error;
  ASSERT_TRUE(io::write_stats_json(file.path(), model, solution, &error)) << error;

  const nlohmann::json blob = nlohmann::json::parse(slurp(file.path()));
  for (const char* key : {"dual_bound", "absolute_gap", "relative_gap"}) {
    EXPECT_FALSE(blob["result"][key].is_null())
        << key << " was serialised as null, losing the fact that it is infinite";
  }
  // Written the way the .sol file already writes them, so float() recovers the value.
  EXPECT_EQ(blob["result"]["dual_bound"].get<std::string>(), "inf");
  EXPECT_EQ(blob["result"]["absolute_gap"].get<std::string>(), "inf");
}

TEST(StatsWriter, FiniteNumbersAreStillPlainJsonNumbers) {
  // The infinity handling must not turn ordinary values into strings; a runner reading a
  // solved instance should see numbers, and every optimal LP must report a zero gap.
  const Model model = make_model();
  const Solution solution = solve_it(model);
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;

  const TempFile file("", ".json");
  std::string error;
  ASSERT_TRUE(io::write_stats_json(file.path(), model, solution, &error)) << error;

  const nlohmann::json blob = nlohmann::json::parse(slurp(file.path()));
  EXPECT_TRUE(blob["result"]["objective"].is_number());
  EXPECT_TRUE(blob["result"]["dual_bound"].is_number());
  EXPECT_TRUE(blob["result"]["absolute_gap"].is_number());
  EXPECT_TRUE(blob["result"]["relative_gap"].is_number());

  // An optimal basis is its own certificate. Reported as exactly zero, not merely small.
  EXPECT_DOUBLE_EQ(blob["result"]["absolute_gap"].get<double>(), 0.0);
  EXPECT_DOUBLE_EQ(blob["result"]["relative_gap"].get<double>(), 0.0);
  EXPECT_DOUBLE_EQ(blob["result"]["dual_bound"].get<double>(), solution.objective);
}

TEST(StatsWriter, ReportsAnUnwritablePath) {
  const Model model = make_model();
  const Solution solution = solve_it(model);
  std::string error;
  EXPECT_FALSE(
      io::write_stats_json("no_such_directory/deeper/out.json", model, solution, &error));
  EXPECT_FALSE(error.empty());
}

}  // namespace
}  // namespace sankhya

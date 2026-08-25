// SPDX-License-Identifier: Apache-2.0
// SANKHYA - command line front end.
//
// Phase 1 provides `version` and `options` only; `solve` and `info` arrive in Phase 2 with
// the MPS reader. The subcommand structure and the generic --option name=value passthrough
// are set up now so that adding a reader later touches one function.

#include <cstdio>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <CLI/CLI.hpp>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/version.hpp"

namespace {

/// Apply repeated --option name=value pairs. Returns false after printing the first error.
bool apply_options(const std::vector<std::string>& assignments, sankhya::Options* options) {
  for (const std::string& assignment : assignments) {
    const std::size_t eq = assignment.find('=');
    if (eq == std::string::npos) {
      fmt::print(stderr, "error: --option expects name=value, got '{}'\n", assignment);
      return false;
    }
    std::string error;
    if (!options->set_from_string(assignment.substr(0, eq), assignment.substr(eq + 1),
                                  &error)) {
      fmt::print(stderr, "error: {}\n", error);
      return false;
    }
  }
  return true;
}

void print_option_table() {
  fmt::print("{:<32} {:<8} {:<12} {}\n", "NAME", "TYPE", "DEFAULT", "DESCRIPTION");
  const sankhya::Options defaults;
  for (const sankhya::OptionSpec& spec : sankhya::Options::registry()) {
    const char* type_name = "string";
    switch (spec.type) {
      case sankhya::OptionType::Bool: type_name = "bool"; break;
      case sankhya::OptionType::Int: type_name = "int"; break;
      case sankhya::OptionType::Double: type_name = "double"; break;
      case sankhya::OptionType::String: type_name = "string"; break;
    }
    fmt::print("{:<32} {:<8} {:<12} {}\n", spec.name, type_name,
               defaults.value_as_string(spec.name), spec.description);
  }
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"SANKHYA - LP / MILP / QP solver", "sankhya"};
  app.require_subcommand(1);
  app.set_version_flag("--version", std::string(sankhya::banner()));

  std::vector<std::string> option_assignments;

  CLI::App* version_cmd = app.add_subcommand("version", "Print build identification");

  CLI::App* options_cmd = app.add_subcommand("options", "List every solver option");

  CLI::App* solve_cmd = app.add_subcommand("solve", "Solve a model file");
  std::string model_path;
  solve_cmd->add_option("file", model_path, "Model file (.mps, .lp)")->required();
  solve_cmd->add_option("--option", option_assignments, "Set a solver option (name=value)");
  double time_limit = -1.0;
  solve_cmd->add_option("--time-limit", time_limit, "Wall-clock limit in seconds");
  std::string solution_path;
  solve_cmd->add_option("--write-sol", solution_path, "Write the solution to this path");
  std::string stats_path;
  solve_cmd->add_option("--stats", stats_path, "Write a JSON result blob to this path");

  CLI::App* info_cmd = app.add_subcommand("info", "Report the dimensions of a model file");
  std::string info_path;
  info_cmd->add_option("file", info_path, "Model file (.mps, .lp)")->required();

  CLI11_PARSE(app, argc, argv);

  if (version_cmd->parsed()) {
    fmt::print("{}\n", sankhya::banner());
    return 0;
  }

  if (options_cmd->parsed()) {
    print_option_table();
    return 0;
  }

  sankhya::Options options;
  if (!apply_options(option_assignments, &options)) return 2;
  if (time_limit > 0.0) options.set_double("time_limit", time_limit);

  if (solve_cmd->parsed() || info_cmd->parsed()) {
    // The MPS and LP readers land in Phase 2. Reporting that plainly beats pretending.
    fmt::print(stderr,
               "error: model readers are not implemented yet (Phase 2). "
               "`sankhya version` and `sankhya options` work today.\n");
    return 3;
  }

  return 0;
}

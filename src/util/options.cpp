// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the option registry and its string parser.
//
// THE REGISTRY IS THE ONLY PLACE AN OPTION IS DECLARED. The CLI, the C API and the Python
// bindings all read it. Adding a knob is one row here.
//
// Defaults are taken from include/sankhya/tolerances.hpp so that there is exactly one
// numerical source of truth, per CLAUDE.md.

#include "sankhya/options.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <unordered_map>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace {

constexpr double kNoLimit = std::numeric_limits<double>::max();

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string trim(std::string_view s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) --e;
  return std::string(s.substr(b, e - b));
}

/// Index of each option name in the registry, built once.
const std::unordered_map<std::string, std::size_t>& name_index() {
  static const std::unordered_map<std::string, std::size_t> index = [] {
    std::unordered_map<std::string, std::size_t> m;
    const std::vector<OptionSpec>& specs = Options::registry();
    for (std::size_t i = 0; i < specs.size(); ++i) m.emplace(specs[i].name, i);
    return m;
  }();
  return index;
}

std::size_t require_index(const std::string& name) {
  const auto it = name_index().find(name);
  assert(it != name_index().end() && "unknown option name in a typed accessor");
  return it->second;
}

}  // namespace

const std::vector<OptionSpec>& Options::registry() {
  static const std::vector<OptionSpec> specs = [] {
    std::vector<OptionSpec> s;

    // ---- Termination -------------------------------------------------------------------
    s.push_back({"time_limit",
                 OptionType::Double,
                 kNoLimit,
                 "Wall-clock limit in seconds.",
                 0.0,
                 kNoLimit,
                 {}});
    s.push_back({"iteration_limit",
                 OptionType::Int,
                 std::int64_t{-1},
                 "Simplex/IPM/PDHG iteration limit; -1 for no limit.",
                 -1.0,
                 kNoLimit,
                 {}});
    s.push_back({"node_limit",
                 OptionType::Int,
                 std::int64_t{-1},
                 "Branch-and-cut node limit; -1 for no limit.",
                 -1.0,
                 kNoLimit,
                 {}});

    // ---- Engine selection --------------------------------------------------------------
    s.push_back({"algorithm",
                 OptionType::String,
                 std::string("auto"),
                 "LP engine: auto, simplex, dual-simplex, pdhg, ipm.",
                 0.0,
                 0.0,
                 {"auto", "simplex", "dual-simplex", "pdhg", "ipm"}});
    s.push_back({"presolve",
                 OptionType::Bool,
                 true,
                 "Run presolve reductions before solving.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"gpu",
                 OptionType::Bool,
                 false,
                 "Use the CUDA path where available; silently falls back to CPU.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"threads",
                 OptionType::Int,
                 std::int64_t{1},
                 "Worker threads; 0 means one per hardware core.",
                 0.0,
                 1024.0,
                 {}});
    s.push_back({"deterministic",
                 OptionType::Bool,
                 true,
                 "Reproduce identical results across thread counts (work-based clock).",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"random_seed",
                 OptionType::Int,
                 std::int64_t{0},
                 "Seed for every randomised decision in the solver.",
                 0.0,
                 kNoLimit,
                 {}});

    // ---- Tolerances. Defaults come from tolerances.hpp, never from a literal here. -----
    s.push_back({"primal_feasibility_tolerance",
                 OptionType::Double,
                 tol::kPrimalFeasibility,
                 "Max allowed row/column bound violation.",
                 1e-12,
                 1e-3,
                 {}});
    s.push_back({"dual_feasibility_tolerance",
                 OptionType::Double,
                 tol::kDualFeasibility,
                 "Max allowed reduced-cost sign violation.",
                 1e-12,
                 1e-3,
                 {}});
    s.push_back({"integrality_tolerance",
                 OptionType::Double,
                 tol::kIntegrality,
                 "Max distance from an integer still counted as integral.",
                 1e-12,
                 1e-3,
                 {}});
    s.push_back({"mip_relative_gap",
                 OptionType::Double,
                 tol::kMipRelativeGap,
                 "Stop when the relative MIP gap falls below this.",
                 0.0,
                 1.0,
                 {}});
    s.push_back({"mip_absolute_gap",
                 OptionType::Double,
                 tol::kMipAbsoluteGap,
                 "Stop when the absolute MIP gap falls below this.",
                 0.0,
                 kNoLimit,
                 {}});
    s.push_back({"pdhg_tolerance",
                 OptionType::Double,
                 tol::kPdhgLoose,
                 "PDHG relative KKT termination tolerance.",
                 1e-14,
                 1e-1,
                 {}});

    // ---- Reporting ---------------------------------------------------------------------
    s.push_back({"log_level",
                 OptionType::String,
                 std::string("info"),
                 "off, error, warning, info, verbose, debug.",
                 0.0,
                 0.0,
                 {"off", "error", "warning", "info", "verbose", "debug"}});
    s.push_back({"log_to_console",
                 OptionType::Bool,
                 true,
                 "Write the solver log to stdout.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"numerics_report",
                 OptionType::Bool,
                 false,
                 "Print scaling ranges, basis conditioning and refactorization counts.",
                 0.0,
                 0.0,
                 {}});
    return s;
  }();
  return specs;
}

Options::Options() {
  const std::vector<OptionSpec>& specs = registry();
  values_.reserve(specs.size());
  for (const OptionSpec& spec : specs) values_.push_back(spec.default_value);
}

const OptionSpec* Options::find_spec(const std::string& name) {
  const auto it = name_index().find(name);
  if (it == name_index().end()) return nullptr;
  return &registry()[it->second];
}

bool Options::exists(const std::string& name) {
  return find_spec(name) != nullptr;
}

const OptionValue& Options::value_of(const std::string& name) const {
  return values_[require_index(name)];
}

OptionValue& Options::mutable_value_of(const std::string& name) {
  return values_[require_index(name)];
}

bool Options::set_from_string(const std::string& raw_name, const std::string& raw_text,
                              std::string* error) {
  const std::string name = to_lower(trim(raw_name));
  const std::string text = trim(raw_text);
  const auto it = name_index().find(name);
  if (it == name_index().end()) {
    if (error != nullptr) *error = fmt::format("unknown option '{}'", raw_name);
    return false;
  }
  const OptionSpec& spec = registry()[it->second];

  switch (spec.type) {
    case OptionType::Bool: {
      const std::string v = to_lower(text);
      if (v == "1" || v == "true" || v == "on" || v == "yes") {
        values_[it->second] = true;
        return true;
      }
      if (v == "0" || v == "false" || v == "off" || v == "no") {
        values_[it->second] = false;
        return true;
      }
      if (error != nullptr) {
        *error = fmt::format("option '{}' expects a boolean, got '{}'", name, raw_text);
      }
      return false;
    }
    case OptionType::Int: {
      errno = 0;
      char* end = nullptr;
      const long long parsed = std::strtoll(text.c_str(), &end, 10);
      if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' expects an integer, got '{}'", name, raw_text);
        }
        return false;
      }
      const auto v = static_cast<double>(parsed);
      if (v < spec.min_value || v > spec.max_value) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' must be in [{:g}, {:g}], got {}", name,
                               spec.min_value, spec.max_value, parsed);
        }
        return false;
      }
      values_[it->second] = static_cast<std::int64_t>(parsed);
      return true;
    }
    case OptionType::Double: {
      errno = 0;
      char* end = nullptr;
      const double parsed = std::strtod(text.c_str(), &end);
      if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' expects a number, got '{}'", name, raw_text);
        }
        return false;
      }
      if (parsed < spec.min_value || parsed > spec.max_value) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' must be in [{:g}, {:g}], got {:g}", name,
                               spec.min_value, spec.max_value, parsed);
        }
        return false;
      }
      values_[it->second] = parsed;
      return true;
    }
    case OptionType::String: {
      const std::string v = to_lower(text);
      if (!spec.choices.empty() &&
          std::find(spec.choices.begin(), spec.choices.end(), v) == spec.choices.end()) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' must be one of [{}], got '{}'", name,
                               fmt::join(spec.choices, ", "), raw_text);
        }
        return false;
      }
      values_[it->second] = v;
      return true;
    }
  }
  if (error != nullptr) *error = "unreachable option type";
  return false;
}

void Options::set_bool(const std::string& name, bool value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<bool>(slot) && "option is not a bool");
  slot = value;
}

void Options::set_int(const std::string& name, std::int64_t value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<std::int64_t>(slot) && "option is not an int");
  slot = value;
}

void Options::set_double(const std::string& name, double value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<double>(slot) && "option is not a double");
  slot = value;
}

void Options::set_string(const std::string& name, const std::string& value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<std::string>(slot) && "option is not a string");
  slot = to_lower(value);
}

bool Options::get_bool(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  assert(std::holds_alternative<bool>(slot) && "option is not a bool");
  return std::get<bool>(slot);
}

std::int64_t Options::get_int(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  assert(std::holds_alternative<std::int64_t>(slot) && "option is not an int");
  return std::get<std::int64_t>(slot);
}

double Options::get_double(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  assert(std::holds_alternative<double>(slot) && "option is not a double");
  return std::get<double>(slot);
}

const std::string& Options::get_string(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  assert(std::holds_alternative<std::string>(slot) && "option is not a string");
  return std::get<std::string>(slot);
}

bool Options::is_modified(const std::string& name) const {
  const std::size_t i = require_index(name);
  return values_[i] != registry()[i].default_value;
}

std::vector<std::string> Options::modified_names() const {
  std::vector<std::string> names;
  const std::vector<OptionSpec>& specs = registry();
  for (std::size_t i = 0; i < specs.size(); ++i) {
    if (values_[i] != specs[i].default_value) names.push_back(specs[i].name);
  }
  return names;
}

std::string Options::value_as_string(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
          return v ? "true" : "false";
        } else if constexpr (std::is_same_v<T, std::string>) {
          return v;
        } else if constexpr (std::is_same_v<T, double>) {
          return v >= kNoLimit ? std::string("inf") : fmt::format("{:g}", v);
        } else {
          return fmt::format("{}", v);
        }
      },
      slot);
}

}  // namespace sankhya

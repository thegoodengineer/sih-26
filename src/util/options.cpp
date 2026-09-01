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

/// A typed accessor naming an option that is not in the registry is a programming error in
/// our own code, not bad user input. Fail loudly and identically in Release and Debug: an
/// `assert` alone vanishes under NDEBUG and leaves a genuine out-of-range dereference in the
/// shipped binary, which is exactly the silent-wrong-answer failure mode CLAUDE.md warns
/// about. Marking the failure path [[noreturn]] also tells the optimizer the iterator is
/// dereferenceable, which is what clears -Wnull-dereference on GCC 16.
[[noreturn]] void unknown_option_name(const std::string& name) {
  fmt::print(stderr, "sankhya: internal error - unknown option name '{}' in a typed accessor\n",
             name);
  std::abort();
}

std::size_t require_index(const std::string& name) {
  const std::unordered_map<std::string, std::size_t>& index = name_index();
  const auto it = index.find(name);
  if (it == index.end()) unknown_option_name(name);
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
    s.push_back({"pricing",
                 OptionType::String,
                 std::string("dantzig"),
                 "Simplex entering-variable rule: dantzig (default) or devex. Devex takes "
                 "far fewer iterations but is NOT yet numerically safe with the current "
                 "textbook ratio test - it costs two Netlib medium instances to a singular "
                 "basis. It needs the Harris two-pass ratio test (#67) before it can be the "
                 "default; see #66 for the measurements.",
                 0.0,
                 0.0,
                 {"devex", "dantzig"}});
    s.push_back({"mps_format",
                 OptionType::String,
                 std::string("auto"),
                 "MPS dialect: auto, free, fixed. auto reads with the whitespace tokenizer "
                 "and retries in fixed columns only if that fails.",
                 0.0,
                 0.0,
                 {"auto", "free", "fixed"}});
    // Default TRUE. Measured: equilibration took the Netlib medium tier from 26/50 to
    // 39/50 by itself. Exposed as an option because turning it off is how a scaling bug
    // gets localised, not because it is optional.
    s.push_back({"scaling",
                 OptionType::Bool,
                 true,
                 "Equilibrate the constraint matrix (Ruiz + Pock-Chambolle) before solving.",
                 0.0,
                 0.0,
                 {}});
    // Implemented as of #43, so there is no planned_for marker any more. Default TRUE for
    // the same reason as scaling: the postsolve round-trip is asserted against the exact
    // rational oracle and re-measured against the ORIGINAL model on every solve, so
    // defaulting it off would mean shipping a deliberately slower solver to dodge a risk
    // the tests already cover.
    s.push_back({"presolve",
                 OptionType::Bool,
                 true,
                 "Run presolve reductions before solving, and postsolve the answer back.",
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
                 {},
                 "Phase 7"});
    s.push_back({"deterministic",
                 OptionType::Bool,
                 true,
                 "Reproduce identical results across thread counts (work-based clock).",
                 0.0,
                 0.0,
                 {},
                 "Phase 7"});
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
    s.push_back({"pdhg_restart",
                 OptionType::Bool,
                 true,
                 "Restart PDHG on the KKT-error criterion; off is for evidence runs.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"qp_tolerance",
                 OptionType::Double,
                 1e-8,
                 "Convex QP primal/dual residual termination tolerance.",
                 1e-14,
                 0.1,
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
                 {},
                 "Phase 9"});
    s.push_back({"progress_out",
                 OptionType::String,
                 std::string(""),
                 "Append live solve progress as JSON lines to this file; empty disables it.",
                 0.0,
                 0.0,
                 {},
                 /*planned_for=*/std::string(""),
                 /*case_sensitive=*/true});
    return s;
  }();
  return specs;
}

Options::Options() {
  const std::vector<OptionSpec>& specs = registry();

  values_.reserve(specs.size());
  explicitly_set_.reserve(specs.size());

  for (const OptionSpec& spec : specs) {
    values_.push_back(spec.default_value);
    explicitly_set_.push_back(false);
  }
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
        explicitly_set_[it->second] = true;
        return true;
      }
      if (v == "0" || v == "false" || v == "off" || v == "no") {
        values_[it->second] = false;
        explicitly_set_[it->second] = true;
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
      explicitly_set_[it->second] = true;
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
      explicitly_set_[it->second] = true;
      return true;
    }

    case OptionType::String: {
      const std::string v = spec.case_sensitive ? text : to_lower(text);
      if (!spec.choices.empty() &&
          std::find(spec.choices.begin(), spec.choices.end(), v) == spec.choices.end()) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' must be one of [{}], got '{}'", name,
                               fmt::join(spec.choices, ", "), raw_text);
        }
        return false;
      }
      values_[it->second] = v;
      explicitly_set_[it->second] = true;
      return true;
    }
  }

  return false;
}

void Options::set_bool(const std::string& name, bool value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<bool>(slot) && "option is not a bool");
  const std::size_t i = require_index(name);
  slot = value;
  explicitly_set_[i] = true;
}

void Options::set_int(const std::string& name, std::int64_t value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<std::int64_t>(slot) && "option is not an int");
  const std::size_t i = require_index(name);
  slot = value;
  explicitly_set_[i] = true;
}

void Options::set_double(const std::string& name, double value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<double>(slot) && "option is not a double");
  const std::size_t i = require_index(name);
  slot = value;
  explicitly_set_[i] = true;
}

void Options::set_string(const std::string& name, const std::string& value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<std::string>(slot) && "option is not a string");
  const std::size_t i = require_index(name);

  const OptionSpec* spec = find_spec(name);
  slot = (spec != nullptr && spec->case_sensitive) ? value : to_lower(value);
  explicitly_set_[i] = true;
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

bool Options::is_explicitly_set(const std::string& name) const {
  return explicitly_set_[require_index(name)];
}

bool Options::is_modified(const std::string& name) const {
  const std::size_t i = require_index(name);
  return explicitly_set_[i] && values_[i] != registry()[i].default_value;
}

bool Options::is_non_default(const std::string& name) const {
  const std::size_t i = require_index(name);
  return values_[i] != registry()[i].default_value;
}

std::vector<std::string> Options::modified_names() const {
  std::vector<std::string> names;
  const std::vector<OptionSpec>& specs = registry();

  for (std::size_t i = 0; i < specs.size(); ++i) {
    if (explicitly_set_[i] && values_[i] != specs[i].default_value) {
      names.push_back(specs[i].name);
    }
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

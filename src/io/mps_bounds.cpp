// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MPS RANGES and BOUNDS section handlers.
//
// Reference: IBM, "MPS file format" (the de-facto specification everyone implements), plus
// Maros, "Computational Techniques of the Simplex Method" (Kluwer, 2003), appendix A, for
// the RANGES and BOUNDS semantics reproduced below.
//
// Split out of mps_reader.cpp by issue #9 (a pure file split, no logic change) once that
// file passed the project's ~600 line guideline. These two sections carry the remaining
// correctness traps in the reader - a RANGES row whose sign convention is inverted, or a
// BOUNDS UP entry that fails to imply a free lower bound, produces a model that is
// perfectly well formed, solves cleanly, and returns the optimum of a different problem -
// so they get a file of their own next to the tests that cover them.
//
// MpsParser and the row-index sentinels are declared once in mps_parser.hpp, shared with
// mps_reader.cpp. The (type, rhs, range) -> two-sided bounds resolution itself stays in
// mps_reader.cpp's finish_rows(), which is part of the two-pass driver, not a section
// handler.

#include <string>

#include <fmt/format.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

#include "mps_parser.hpp"
#include "token.hpp"

namespace sankhya::io {

// ---- RANGES -------------------------------------------------------------------------

bool MpsParser::do_ranges(std::string* error) {
  std::size_t k = 0;
  if (tok_.size() % 2 == 1) {
    const std::string name(tok_[0]);
    if (range_vector_.empty()) range_vector_ = name;
    if (name != range_vector_) return true;
    k = 1;
  }
  if (tok_.size() <= k) {
    *error = reader_.error_at("RANGES entry has no row/value pair");
    return false;
  }

  for (; k + 1 < tok_.size(); k += 2) {
    const Index row = find_row(tok_[k]);
    if (row == kUnknownRow) {
      *error = reader_.error_at(fmt::format("RANGES entry names unknown row '{}'", tok_[k]));
      return false;
    }
    double value = 0.0;
    if (!parse_double(tok_[k + 1], &value)) {
      *error = reader_.error_at(fmt::format("'{}' is not a number", tok_[k + 1]));
      return false;
    }
    if (row == kIgnoredRow) continue;
    if (row == kObjectiveRow) {
      *error = reader_.error_at("RANGES entry on the objective row is not meaningful");
      return false;
    }
    row_range_[static_cast<std::size_t>(row)] = value;
    row_has_range_[static_cast<std::size_t>(row)] = 1;
  }
  return true;
}

// ---- BOUNDS -------------------------------------------------------------------------

bool MpsParser::do_bounds(std::string* error) {
  if (tok_.size() < 2) {
    *error = reader_.error_at("BOUNDS entry needs a type and a column");
    return false;
  }
  const std::string type = to_upper(tok_[0]);

  const bool takes_value = (type == "UP" || type == "LO" || type == "FX" || type == "LI" ||
                            type == "UI" || type == "SC");
  const bool valueless = (type == "FR" || type == "MI" || type == "PL" || type == "BV");
  if (!takes_value && !valueless) {
    *error = reader_.error_at(
        fmt::format("unknown bound type '{}'; expected UP LO FX FR MI PL BV LI UI", tok_[0]));
    return false;
  }

  // The bound-vector name is optional, exactly as for RHS. With a value-taking type the
  // payload is (column, value), so 4 fields means the name is present and 3 means it is
  // not; with a value-less type the payload is (column) alone, so 3 means present.
  std::size_t column_field = 0;
  if (takes_value) {
    if (tok_.size() == 4) {
      column_field = 2;
    } else if (tok_.size() == 3) {
      column_field = 1;
    } else {
      *error = reader_.error_at(fmt::format("bound type {} needs a column and a value", type));
      return false;
    }
  } else {
    if (tok_.size() >= 3) {
      column_field = 2;  // a dummy value in field 4 is tolerated and ignored
    } else {
      column_field = 1;
    }
  }

  if (column_field == 2) {
    const std::string name(tok_[1]);
    if (bound_vector_.empty()) bound_vector_ = name;
    if (name != bound_vector_) return true;
  }

  const Index col = find_column(tok_[column_field]);
  if (col < 0) {
    *error = reader_.error_at(fmt::format(
        "BOUNDS entry names column '{}', which has no COLUMNS entry", tok_[column_field]));
    return false;
  }
  const auto u = static_cast<std::size_t>(col);

  double value = 0.0;
  if (takes_value) {
    if (!parse_double(tok_[column_field + 1], &value)) {
      *error = reader_.error_at(fmt::format("'{}' is not a number", tok_[column_field + 1]));
      return false;
    }
    value = normalize_infinity(value);
  }

  if (type == "UP") {
    col_upper_[u] = value;
    // The trap. An UP bound with a negative value on a column whose lower bound is still
    // the implicit 0 means the modeller intends a negative variable, so the lower bound
    // becomes -inf. Without this, [0, -5] is an empty interval and the model reads as
    // infeasible.
    //
    // The convention is documented for continuous columns and is implementation-defined for
    // integer ones, where established readers disagree. This reader previously declined to
    // choose and left the lower bound at 0 for an integer column - which produced [0, -5],
    // an empty interval, so Model::validate() rejected the model and the file could not be
    // loaded AT ALL. Declining to guess produced a worse outcome than either guess, and the
    // resulting error named crossed bounds, which is the symptom rather than the cause.
    //
    // The convention is now applied to integer columns too, with the warning kept so the
    // ambiguity is still visible in the log. That also removes a real inconsistency:
    // tools/verify_solution.py already reads the file this way, so the independent checker
    // and the C++ reader were interpreting the same bytes differently - exactly the class of
    // disagreement that verifier exists to detect, sitting inside the pair by construction.
    if (value < 0.0 && col_lower_explicit_[u] == 0) {
      col_lower_[u] = -kInfinity;
      if (col_type_[u] == VarType::kInteger) {
        default_logger().warning(
            "UP bound {} on integer column '{}' with no explicit lower bound: applying the "
            "negative-upper convention and setting the lower bound to -inf (readers disagree "
            "on this case for integer columns)",
            value, col_names_[u]);
      }
    }
  } else if (type == "LO") {
    col_lower_[u] = value;
    col_lower_explicit_[u] = 1;
  } else if (type == "FX") {
    col_lower_[u] = value;
    col_upper_[u] = value;
    col_lower_explicit_[u] = 1;
  } else if (type == "FR") {
    col_lower_[u] = -kInfinity;
    col_upper_[u] = kInfinity;
    col_lower_explicit_[u] = 1;
  } else if (type == "MI") {
    // MI sets the lower bound only. Some pre-1990 readers also forced the upper bound to
    // zero; that behaviour is long obsolete and would silently cut off the feasible region.
    col_lower_[u] = -kInfinity;
    col_lower_explicit_[u] = 1;
  } else if (type == "PL") {
    col_upper_[u] = kInfinity;
  } else if (type == "BV") {
    col_type_[u] = VarType::kInteger;
    col_lower_[u] = 0.0;
    col_upper_[u] = 1.0;
    col_lower_explicit_[u] = 1;
  } else if (type == "LI") {
    col_type_[u] = VarType::kInteger;
    col_lower_[u] = value;
    col_lower_explicit_[u] = 1;
  } else if (type == "UI") {
    col_type_[u] = VarType::kInteger;
    col_upper_[u] = value;
  } else if (type == "SC") {
    *error = reader_.error_at(
        "semi-continuous bounds (SC) are not supported; the model would be misread as a "
        "plain integer problem");
    return false;
  }
  return true;
}

}  // namespace sankhya::io

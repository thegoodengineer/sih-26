// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MPS reader, fixed and free dialect.
//
// Reference: IBM, "MPS file format" (the de-facto specification everyone implements), plus
// Maros, "Computational Techniques of the Simplex Method" (Kluwer, 2003), appendix A, for
// the RANGES and BOUNDS semantics reproduced in the tables below.
//
// This is the highest-bug-density file in the project and the bugs are silent. A RANGES
// row whose sign convention is inverted, or a BOUNDS UP entry that fails to imply a free
// lower bound, produces a model that is perfectly well formed, solves cleanly, and returns
// the optimum of a different problem. Two structural decisions defend against that:
//
//   1. Row bounds are NOT written as the sections are read. rhs[] and range[] are recorded
//      separately and the (type, rhs, range) triple is resolved once, at the end, in
//      finish_rows(). The conversion table therefore exists in exactly one place and is
//      testable in isolation, rather than being smeared across two section handlers whose
//      order of appearance in the file could vary.
//   2. Every duplicate matrix entry is detected and rejected. Summing them is what a
//      generic triplet builder does by default, and it is silently wrong: MPS has no
//      accumulate semantics.
//
// The two dialects differ only in tokenization, so they share every handler below.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "sankhya/io.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

#include "mps_parser.hpp"
#include "token.hpp"

// The RANGES and BOUNDS section handlers (do_ranges, do_bounds) live in mps_bounds.cpp -
// issue #9, a pure file split with no logic change. MpsParser itself, and the sentinel row
// indices both files need, are declared once in mps_parser.hpp.

namespace sankhya::io {
namespace {

enum class Section {
  kNone,
  kName,
  kObjsense,
  kObjsenseValue,  ///< OBJSENSE given as a section, its value on the following line
  kRows,
  kColumns,
  kRhs,
  kRanges,
  kBounds,
  kQuadratic,  ///< QUADOBJ / QMATRIX / QSECTION - the QPS quadratic objective
  kEnd
};

/// Byte offsets and widths of the six fixed-format fields, per the IBM specification.
/// Columns are quoted 1-based in the spec: 2-3, 5-12, 15-22, 25-36, 40-47, 50-61.
struct FixedField {
  std::size_t offset;
  std::size_t width;
};
constexpr FixedField kFixedFields[6] = {{1, 2}, {4, 8}, {14, 8}, {24, 12}, {39, 8}, {49, 12}};

[[nodiscard]] bool section_from_keyword(std::string_view keyword, Section* out) {
  const std::string k = to_upper(keyword);
  if (k == "NAME") {
    *out = Section::kName;
    return true;
  }
  if (k == "OBJSENSE" || k == "OBJSENS") {
    *out = Section::kObjsense;
    return true;
  }
  if (k == "ROWS") {
    *out = Section::kRows;
    return true;
  }
  if (k == "COLUMNS") {
    *out = Section::kColumns;
    return true;
  }
  if (k == "RHS") {
    *out = Section::kRhs;
    return true;
  }
  if (k == "RANGES") {
    *out = Section::kRanges;
    return true;
  }
  if (k == "BOUNDS") {
    *out = Section::kBounds;
    return true;
  }
  // QPS quadratic sections, all four spellings in circulation. Before these were
  // recognised at all, a QUADOBJ block was absorbed by whatever section preceded it - a QP
  // written after RHS was read as an extra RHS vector, the model came back with problem
  // class LP, and the solver returned `optimal` for the LP RELAXATION of a quadratic
  // program. They were then recognised so the file could be REFUSED; now they are read.
  if (k == "QUADOBJ" || k == "QMATRIX" || k == "QSECTION" || k == "QUADS") {
    *out = Section::kQuadratic;
    return true;
  }
  if (k == "ENDATA") {
    *out = Section::kEnd;
    return true;
  }
  return false;
}

}  // namespace

void MpsParser::split(const std::string& line) {
  // Section headers are NOT field-formatted in either dialect: they begin in column 1 and
  // are read as plain words. Running the column windows over "NAME  FIXED" would take
  // characters 2-3 and yield the token "AM". The indentation rule is what separates the two
  // cases, and it is the same rule the section dispatcher below uses.
  const bool is_header_line = !line.empty() && !is_space(line[0]);

  if (format_ == MpsFormat::kFixed && !is_header_line) {
    tok_.clear();
    const std::string_view view(line);
    for (const FixedField& f : kFixedFields) {
      if (f.offset >= view.size()) break;
      const std::size_t width = std::min(f.width, view.size() - f.offset);
      const std::string_view field = trim(view.substr(f.offset, width));
      if (!field.empty()) tok_.push_back(field);
    }
    return;
  }

  tokenize(line, &tok_);
  // Free-format MPS uses '$' to open a comment. Fixed format does not - there, everything
  // past column 61 is the comment - so this must not run on a fixed-format header line that
  // reached the whitespace tokenizer above.
  if (format_ == MpsFormat::kFixed) return;
  for (std::size_t i = 0; i < tok_.size(); ++i) {
    if (tok_[i].front() == '$') {
      tok_.resize(i);
      break;
    }
  }
}

Index MpsParser::find_row(std::string_view name) const {
  const auto it = row_index_.find(std::string(name));
  return it == row_index_.end() ? kUnknownRow : it->second;
}

Index MpsParser::find_column(std::string_view name) const {
  const auto it = col_index_.find(std::string(name));
  return it == col_index_.end() ? -1 : it->second;
}

bool MpsParser::record_entry(Index row, Index col, double value, std::string* error) {
  if (row == kIgnoredRow) return true;  // a dropped free row

  // An infinite coefficient is meaningless. MPS spells infinity 1e30, but that convention
  // describes a BOUND - "this variable is unconstrained above" - and says nothing about an
  // entry of A. A coefficient that overflowed to infinity (a stray "1e400", a corrupted
  // exponent) is a broken file, and letting it through produces a model that still solves,
  // still prints "optimal", and reports NaN reduced costs alongside it. parse_double has
  // already rejected NaN in every field; this is the coefficient-only half of the rule.
  if (!std::isfinite(value)) {
    *error = reader_.error_at(
        fmt::format("coefficient for column '{}' is {}; a constraint or objective "
                    "coefficient must be finite",
                    col_names_[static_cast<std::size_t>(col)], value));
    return false;
  }

  if (row == kObjectiveRow) {
    if (!seen_objective_.insert(col).second) {
      *error = reader_.error_at(fmt::format("duplicate objective coefficient for column '{}'",
                                            col_names_[static_cast<std::size_t>(col)]));
      return false;
    }
    col_cost_[static_cast<std::size_t>(col)] = value;
    return true;
  }

  const auto key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row)) << 32) |
                   static_cast<std::uint64_t>(static_cast<std::uint32_t>(col));
  if (!seen_entries_.insert(key).second) {
    *error = reader_.error_at(fmt::format(
        "duplicate entry for row '{}' and column '{}'; MPS has no accumulate semantics",
        row_names_[static_cast<std::size_t>(row)], col_names_[static_cast<std::size_t>(col)]));
    return false;
  }

  // Structural zeros are dropped here rather than by finalize() so that the nonzero count
  // we report matches what a judge counts by eye in the file.
  if (value != 0.0) {
    tri_row_.push_back(row);
    tri_col_.push_back(col);
    tri_value_.push_back(value);
  }
  return true;
}

// ---- ROWS ---------------------------------------------------------------------------

bool MpsParser::do_rows(std::string* error) {
  // Exactly two fields, never more. Being lenient here is what breaks fixed-format
  // detection: a fixed-format row named "MY ROW" free-tokenizes to three tokens, and a
  // reader that shrugs and takes the first two silently creates a row called "MY" instead
  // of failing and letting read_mps() retry in fixed columns.
  if (tok_.size() != 2) {
    *error = reader_.error_at(fmt::format(
        "ROWS entry has {} fields, expected exactly 2 (type and name)", tok_.size()));
    return false;
  }
  const std::string type = to_upper(tok_[0]);
  if (type.size() != 1 || (type != "N" && type != "L" && type != "G" && type != "E")) {
    *error =
        reader_.error_at(fmt::format("unknown row type '{}', expected N, L, G or E", tok_[0]));
    return false;
  }
  const std::string name(tok_[1]);

  Index assigned = 0;
  if (type == "N") {
    if (!have_objective_row_) {
      have_objective_row_ = true;
      assigned = kObjectiveRow;
    } else {
      // Second and later N rows are free rows. Dropping them is universal practice: they
      // constrain nothing, and carrying them would inflate every reported row count.
      assigned = kIgnoredRow;
      ++free_rows_dropped_;
    }
  } else {
    assigned = static_cast<Index>(row_type_.size());
    row_type_.push_back(type[0]);
    row_rhs_.push_back(0.0);
    row_range_.push_back(0.0);
    row_has_range_.push_back(0);
    row_names_.push_back(name);
  }

  if (!row_index_.emplace(name, assigned).second) {
    *error = reader_.error_at(fmt::format("duplicate row name '{}'", name));
    return false;
  }
  return true;
}

// ---- COLUMNS ------------------------------------------------------------------------

bool MpsParser::do_columns(std::string* error) {
  // MARKER records switch integrality on and off. They are recognised by the quoted
  // 'MARKER' token, whose position varies between writers, so we scan rather than index.
  bool is_marker = false;
  bool intorg = false;
  bool intend = false;
  for (const std::string_view raw : tok_) {
    const std::string t = to_upper(unquote(raw));
    if (t == "MARKER") is_marker = true;
    if (t == "INTORG") intorg = true;
    if (t == "INTEND") intend = true;
  }
  if (is_marker || intorg || intend) {
    if (intorg) {
      integer_marker_active_ = true;
    } else if (intend) {
      integer_marker_active_ = false;
    } else {
      *error = reader_.error_at("MARKER record names neither INTORG nor INTEND");
      return false;
    }
    return true;
  }

  if (tok_.size() != 3 && tok_.size() != 5) {
    *error = reader_.error_at(fmt::format(
        "COLUMNS entry has {} fields, expected 3 or 5 (column, row, value[, row, value])",
        tok_.size()));
    return false;
  }

  const std::string name(tok_[0]);
  Index col = find_column(name);
  if (col < 0) {
    col = static_cast<Index>(col_cost_.size());
    col_index_.emplace(name, col);
    col_names_.push_back(name);
    col_cost_.push_back(0.0);
    col_lower_.push_back(0.0);  // the MPS default is [0, +inf)
    col_upper_.push_back(kInfinity);
    col_type_.push_back(integer_marker_active_ ? VarType::kInteger : VarType::kContinuous);
    col_lower_explicit_.push_back(0);
    if (integer_marker_active_) {
      // An INTORG column with no explicit bound is [0, +inf) in the specification. Readers
      // that silently make it [0, 1] change the problem; we do not, and note it in the log
      // only if the model turns out to have no BOUNDS entry for it at all.
      col_upper_.back() = kInfinity;
    }
  }

  for (std::size_t k = 1; k + 1 < tok_.size(); k += 2) {
    const Index row = find_row(tok_[k]);
    if (row == kUnknownRow) {
      *error = reader_.error_at(fmt::format("COLUMNS entry names unknown row '{}'", tok_[k]));
      return false;
    }
    double value = 0.0;
    if (!parse_double(tok_[k + 1], &value)) {
      *error = reader_.error_at(fmt::format("'{}' is not a number", tok_[k + 1]));
      return false;
    }
    if (!record_entry(row, col, value, error)) return false;
  }
  return true;
}

// ---- RHS ----------------------------------------------------------------------------

bool MpsParser::do_rhs(std::string* error) {
  // The vector name is optional. An even field count means it was omitted, since the
  // payload is always (row, value) pairs.
  std::size_t k = 0;
  if (tok_.size() % 2 == 1) {
    const std::string name(tok_[0]);
    if (rhs_vector_.empty()) rhs_vector_ = name;
    if (name != rhs_vector_) {
      if (!warned_extra_vector_) {
        default_logger().warning(
            "{}",
            reader_.error_at(fmt::format(
                "ignoring RHS vector '{}'; using the first vector '{}'", name, rhs_vector_)));
        warned_extra_vector_ = true;
      }
      return true;
    }
    k = 1;
  }
  if (tok_.size() <= k) {
    *error = reader_.error_at("RHS entry has no row/value pair");
    return false;
  }

  for (; k + 1 < tok_.size(); k += 2) {
    const Index row = find_row(tok_[k]);
    if (row == kUnknownRow) {
      *error = reader_.error_at(fmt::format("RHS entry names unknown row '{}'", tok_[k]));
      return false;
    }
    double value = 0.0;
    if (!parse_double(tok_[k + 1], &value)) {
      *error = reader_.error_at(fmt::format("'{}' is not a number", tok_[k + 1]));
      return false;
    }
    if (row == kIgnoredRow) continue;
    if (row == kObjectiveRow) {
      // An RHS entry on the objective row carries the NEGATIVE of the objective constant.
      // This flip is the reader's job; Model::objective_offset is added as-is downstream.
      objective_rhs_ = value;
      continue;
    }
    row_rhs_[static_cast<std::size_t>(row)] = value;
  }
  return true;
}

// ---- RANGES and BOUNDS ----------------------------------------------------------------
// do_ranges() and do_bounds() live in mps_bounds.cpp (issue #9).

// ---- completion ---------------------------------------------------------------------

bool MpsParser::do_quadratic(std::string* error) {
  // QPS QUADOBJ: "colname1 colname2 value", giving one entry of the Hessian of the OBJECTIVE.
  //
  // THE CONVENTION, and it is the trap in this section. QPS states the objective as
  //
  //     c'x + 0.5 x' Q x
  //
  // and lists only the LOWER TRIANGLE of the symmetric Q. A stored off-diagonal entry
  // therefore stands for TWO entries of Q, and the 0.5 is part of the objective rather than
  // part of the data. `sankhya::Model` was defined in exactly this convention - see the note
  // in model.hpp - so entries map across with no transformation at all. A reader that
  // "helpfully" halved the off-diagonals, or mirrored them into both triangles, would produce
  // a model that solves cleanly to the optimum of a different problem.
  //
  // Files differ on which order the two column names appear in, so the pair is normalised to
  // (max, min) rather than trusted.
  if (tok_.size() < 3) {
    *error = reader_.error_at(fmt::format(
        "QUADOBJ entry has {} field(s), expected 3 (column, column, value)", tok_.size()));
    return false;
  }

  const Index first = find_column(tok_[0]);
  const Index second = find_column(tok_[1]);
  if (first < 0 || second < 0) {
    *error = reader_.error_at(
        fmt::format("QUADOBJ names column '{}' which never appeared in COLUMNS",
                    first < 0 ? std::string(tok_[0]) : std::string(tok_[1])));
    return false;
  }

  double value = 0.0;
  if (!parse_double(tok_[2], &value)) {
    *error = reader_.error_at(fmt::format("'{}' is not a number", tok_[2]));
    return false;
  }
  if (!std::isfinite(value)) {
    *error = reader_.error_at(fmt::format(
        "Hessian entry for ('{}', '{}') is {}; it must be finite", tok_[0], tok_[1], value));
    return false;
  }

  const Index row = std::max(first, second);
  const Index col = std::min(first, second);
  const auto key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row)) << 32) |
                   static_cast<std::uint64_t>(static_cast<std::uint32_t>(col));
  if (!seen_quad_entries_.insert(key).second) {
    // A file listing both (i, j) and (j, i) hits this, which is the point: the two name the
    // same entry of a symmetric matrix, and summing them would double the coefficient.
    *error = reader_.error_at(fmt::format(
        "duplicate Hessian entry for columns '{}' and '{}'; QPS lists the lower triangle of a "
        "symmetric matrix, so each pair may appear only once and has no accumulate semantics",
        col_names_[static_cast<std::size_t>(row)], col_names_[static_cast<std::size_t>(col)]));
    return false;
  }

  if (value != 0.0) {
    quad_row_.push_back(row);
    quad_col_.push_back(col);
    quad_value_.push_back(value);
  }
  return true;
}

bool MpsParser::finish_rows(std::string* error) {
  const std::size_t m = row_type_.size();
  model_->row_lower.assign(m, -kInfinity);
  model_->row_upper.assign(m, kInfinity);

  for (std::size_t i = 0; i < m; ++i) {
    const char type = row_type_[i];
    const double b = normalize_infinity(row_rhs_[i]);
    double lo = -kInfinity;
    double hi = kInfinity;

    if (row_has_range_[i] == 0) {
      switch (type) {
        case 'L': hi = b; break;
        case 'G': lo = b; break;
        case 'E':
          lo = b;
          hi = b;
          break;
        default: break;
      }
    } else {
      // RANGES, per the IBM specification. R is the range value:
      //
      //   type   sign of R      row becomes
      //   ----   ---------      -----------------------
      //     G      any          [ b        , b + |R| ]
      //     L      any          [ b - |R|  , b       ]
      //     E      R >= 0       [ b        , b + R   ]
      //     E      R <  0       [ b + R    , b       ]
      //
      // The E case is the one that is routinely inverted, because it is the only place in
      // the whole format where the SIGN of the value changes which end it applies to.
      const double r = row_range_[i];
      const double magnitude = std::fabs(r);
      switch (type) {
        case 'G':
          lo = b;
          hi = b + magnitude;
          break;
        case 'L':
          lo = b - magnitude;
          hi = b;
          break;
        case 'E':
          if (r >= 0.0) {
            lo = b;
            hi = b + r;
          } else {
            lo = b + r;
            hi = b;
          }
          break;
        default: break;
      }
    }

    if (lo > hi) {
      *error = fmt::format("row '{}' has an empty range [{}, {}]", row_names_[i], lo, hi);
      return false;
    }
    model_->row_lower[i] = lo;
    model_->row_upper[i] = hi;
  }
  return true;
}

void MpsParser::finish_model() {
  const auto n = static_cast<Index>(col_cost_.size());
  const auto m = static_cast<Index>(row_type_.size());

  model_->col_cost = col_cost_;
  model_->col_lower = col_lower_;
  model_->col_upper = col_upper_;
  model_->col_type = col_type_;
  model_->col_names = col_names_;
  model_->row_names = row_names_;
  model_->objective_offset = normalize_zero(-objective_rhs_);

  for (std::size_t j = 0; j < model_->col_lower.size(); ++j) {
    model_->col_lower[j] = normalize_infinity(model_->col_lower[j]);
    model_->col_upper[j] = normalize_infinity(model_->col_upper[j]);
  }

  model_->matrix.reset(m, n);
  model_->matrix.reserve(tri_row_.size());
  for (std::size_t k = 0; k < tri_row_.size(); ++k) {
    model_->matrix.add_entry(tri_row_[k], tri_col_[k], tri_value_[k]);
  }
  model_->matrix.finalize();
  // The Hessian, in the QPS convention the Model already uses: lower triangle, with the 0.5
  // carried by the objective rather than by the data, so entries need no transformation.
  model_->hessian.reset(n, n);
  model_->hessian.reserve(quad_row_.size());
  for (std::size_t k = 0; k < quad_row_.size(); ++k) {
    model_->hessian.add_entry(quad_row_[k], quad_col_[k], quad_value_[k]);
  }
  model_->hessian.finalize();
}

ReadResult MpsParser::parse(const std::string& path) {
  std::string error;
  if (!reader_.open(path, &error)) return ReadResult::failure(error);

  *model_ = Model{};
  model_->source_path = path;

  Section section = Section::kNone;
  std::string line;
  bool saw_endata = false;

  while (reader_.next(&line)) {
    if (line.empty()) continue;
    if (line[0] == '*') continue;  // comment

    const bool indented = is_space(line[0]);
    split(line);
    if (tok_.empty()) continue;

    if (!indented) {
      Section next = Section::kNone;
      if (section_from_keyword(tok_[0], &next)) {
        if (next == Section::kEnd) {
          saw_endata = true;
          break;
        }
        if (next == Section::kName) {
          model_->name = tok_.size() >= 2 ? std::string(tok_[1]) : std::string();
          section = Section::kNone;
          continue;
        }
        if (next == Section::kObjsense) {
          // Both spellings occur: "OBJSENSE MAX" on one line, or OBJSENSE as a section
          // header with MAX indented beneath it.
          if (tok_.size() >= 2) {
            const std::string value = to_upper(tok_[1]);
            model_->sense = (value == "MAX" || value == "MAXIMIZE") ? ObjSense::kMaximize
                                                                    : ObjSense::kMinimize;
            section = Section::kNone;
          } else {
            section = Section::kObjsenseValue;
          }
          continue;
        }
        section = next;
        continue;
      }
      // An unindented line that is not a known section keyword is treated as data. Free
      // format files in the wild are not reliably indented, and rejecting them here would
      // fail on valid input; the cost is that a row literally named ROWS would confuse us.
    }

    switch (section) {
      case Section::kObjsenseValue: {
        const std::string value = to_upper(tok_[0]);
        model_->sense =
            (value == "MAX" || value == "MAXIMIZE") ? ObjSense::kMaximize : ObjSense::kMinimize;
        section = Section::kNone;
        break;
      }
      case Section::kRows:
        if (!do_rows(&error)) return ReadResult::failure(error);
        break;
      case Section::kColumns:
        if (!do_columns(&error)) return ReadResult::failure(error);
        break;
      case Section::kRhs:
        if (!do_rhs(&error)) return ReadResult::failure(error);
        break;
      case Section::kRanges:
        if (!do_ranges(&error)) return ReadResult::failure(error);
        break;
      case Section::kBounds:
        if (!do_bounds(&error)) return ReadResult::failure(error);
        break;
      case Section::kQuadratic:
        if (!do_quadratic(&error)) return ReadResult::failure(error);
        break;
      case Section::kNone:
      case Section::kName:
      case Section::kObjsense:
      case Section::kEnd:
        return ReadResult::failure(
            reader_.error_at("data line appears before any recognised section header"));
    }
  }

  if (!saw_endata) {
    return ReadResult::failure(fmt::format("{}: file ends without an ENDATA record", path));
  }
  if (!have_objective_row_) {
    default_logger().warning("{}: no N row; the objective is identically zero", path);
  }
  if (free_rows_dropped_ > 0) {
    default_logger().warning("{}: dropped {} free row(s) after the objective row", path,
                             free_rows_dropped_);
  }

  if (!finish_rows(&error)) return ReadResult::failure(fmt::format("{}: {}", path, error));
  finish_model();

  const std::string problem = model_->validate();
  if (!problem.empty()) {
    return ReadResult::failure(fmt::format("{}: model failed validation: {}", path, problem));
  }
  return ReadResult::success();
}

bool parse_mps_format(const std::string& text, MpsFormat* out) noexcept {
  const std::string t = to_upper(text);
  if (t == "AUTO") {
    *out = MpsFormat::kAuto;
    return true;
  }
  if (t == "FREE") {
    *out = MpsFormat::kFree;
    return true;
  }
  if (t == "FIXED") {
    *out = MpsFormat::kFixed;
    return true;
  }
  return false;
}

ReadResult read_mps(const std::string& path, Model* model, MpsFormat format,
                    MpsFormat* format_used) {
  if (format != MpsFormat::kAuto) {
    if (format_used != nullptr) *format_used = format;
    MpsParser parser(model, format);
    return parser.parse(path);
  }

  // Auto: the whitespace tokenizer handles every file whose names contain no blanks, which
  // is all of Netlib and all of MIPLIB. Only when that fails is the column-oriented reader
  // worth trying, and if it also fails the free-format diagnostic is the more useful one.
  MpsParser free_parser(model, MpsFormat::kFree);
  ReadResult free_result = free_parser.parse(path);
  if (free_result.ok) {
    if (format_used != nullptr) *format_used = MpsFormat::kFree;
    return free_result;
  }

  MpsParser fixed_parser(model, MpsFormat::kFixed);
  const ReadResult fixed_result = fixed_parser.parse(path);
  if (fixed_result.ok) {
    if (format_used != nullptr) *format_used = MpsFormat::kFixed;
    default_logger().info("{}: parsed as fixed-format MPS", path);
    return fixed_result;
  }

  return ReadResult::failure(fmt::format("{} (the fixed-format reader also failed: {})",
                                         free_result.error, fixed_result.error));
}

}  // namespace sankhya::io

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the MPS/QPS parser state, shared between mps_reader.cpp (section dispatch, the
// two-pass driver, ROWS/COLUMNS/RHS/QUADOBJ) and mps_bounds.cpp (RANGES/BOUNDS).
//
// Not a public header - internal to src/io/, exactly like line_reader.hpp and token.hpp.
// Split out by issue #9 once mps_reader.cpp passed the project's ~600 line guideline; no
// logic changed in the split, only which .cpp file a given method's body lives in.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"

#include "line_reader.hpp"

namespace sankhya::io {

/// Sentinel row indices used while parsing COLUMNS, RHS and RANGES.
constexpr Index kObjectiveRow = -1;
constexpr Index kIgnoredRow = -2;  ///< a second or later N row: a free row, dropped
constexpr Index kUnknownRow = -3;  ///< the name is not in ROWS at all

class MpsParser {
 public:
  MpsParser(Model* model, MpsFormat format) : model_(model), format_(format) {}

  ReadResult parse(const std::string& path);

 private:
  // ---- tokenization -------------------------------------------------------------------
  void split(const std::string& line);

  // ---- section handlers ---------------------------------------------------------------
  [[nodiscard]] bool do_rows(std::string* error);
  [[nodiscard]] bool do_columns(std::string* error);
  [[nodiscard]] bool do_rhs(std::string* error);
  [[nodiscard]] bool do_ranges(std::string* error);
  [[nodiscard]] bool do_bounds(std::string* error);
  [[nodiscard]] bool do_quadratic(std::string* error);

  // ---- completion ---------------------------------------------------------------------
  [[nodiscard]] bool finish_rows(std::string* error);
  void finish_model();

  [[nodiscard]] Index find_row(std::string_view name) const;
  [[nodiscard]] Index find_column(std::string_view name) const;
  [[nodiscard]] bool record_entry(Index row, Index col, double value, std::string* error);

  Model* model_;
  MpsFormat format_;
  LineReader reader_;
  std::vector<std::string_view> tok_;

  // Rows. row_index_ maps a name to a constraint index, or to the sentinels above.
  std::unordered_map<std::string, Index> row_index_;
  std::vector<char> row_type_;  // 'L', 'G', 'E'
  std::vector<double> row_rhs_;
  std::vector<double> row_range_;
  std::vector<char> row_has_range_;
  std::vector<std::string> row_names_;
  bool have_objective_row_ = false;
  double objective_rhs_ = 0.0;

  // Columns.
  std::unordered_map<std::string, Index> col_index_;
  std::vector<double> col_cost_;
  std::vector<double> col_lower_;
  std::vector<double> col_upper_;
  std::vector<VarType> col_type_;
  std::vector<char> col_lower_explicit_;
  std::vector<std::string> col_names_;
  bool integer_marker_active_ = false;

  // Matrix triplets, plus a duplicate guard keyed on (row, col).
  std::vector<Index> tri_row_;
  std::vector<Index> tri_col_;
  std::vector<double> tri_value_;
  std::unordered_set<std::uint64_t> seen_entries_;
  std::unordered_set<Index> seen_objective_;

  // Hessian triplets, always stored lower-triangular, with the same duplicate guard the
  // constraint matrix uses. QPS has no accumulate semantics either.
  std::vector<Index> quad_row_;
  std::vector<Index> quad_col_;
  std::vector<double> quad_value_;
  std::unordered_set<std::uint64_t> seen_quad_entries_;

  // Only the first named RHS / RANGES / BOUNDS vector is honoured, which is what every
  // established reader does with a multi-vector file.
  std::string rhs_vector_;
  std::string range_vector_;
  std::string bound_vector_;
  bool warned_extra_vector_ = false;
  Count free_rows_dropped_ = 0;
};

}  // namespace sankhya::io

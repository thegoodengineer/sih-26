// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CPLEX LP format reader.
//
// Reference: the LP file format as documented in the public CPLEX and Gurobi reference
// manuals. Per CLAUDE.md this is interface compatibility read from published documentation;
// no solver source was consulted.
//
// The format exists here for the Phase 10 case studies: a refinery blending model is far
// more legible to a judge as `0.8 arab_light + 1.2 bonny <= 10` than as an MPS column
// block, and being able to hand them the algebraic file next to the MPS one is worth the
// reader. Scope is deliberately linear: the quadratic `[ ... ] / 2` section is detected and
// rejected with a clear message rather than silently ignored, because ignoring it would
// solve the LP relaxation of a QP and report it as optimal.
//
// Structure: lex the whole file into a token stream carrying line numbers, split it into
// sections on keywords that begin a line, then parse each section from its token range.
// Statement-level recursive descent over a token vector is far easier to keep correct than
// line-at-a-time parsing, because LP statements wrap across lines freely.

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "sankhya/io.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

#include "line_reader.hpp"
#include "token.hpp"

namespace sankhya::io {
namespace {

enum class TokKind { kIdent, kNumber, kOp, kEof };

struct Tok {
  TokKind kind = TokKind::kEof;
  std::string text;      ///< identifier spelling, or the operator
  double value = 0.0;    ///< numeric value when kind == kNumber
  Count line = 0;
  bool first_on_line = false;
};

/// Relational operators, normalised so that "=<" and "<" both arrive as "<=".
[[nodiscard]] bool is_relop(const Tok& t) {
  return t.kind == TokKind::kOp && (t.text == "<=" || t.text == ">=" || t.text == "=");
}

[[nodiscard]] bool ident_start(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '!' ||
         c == '"' || c == '#' || c == '$' || c == '%' || c == '&' || c == '(' || c == ')' ||
         c == ',' || c == ';' || c == '?' || c == '@' || c == '\'' || c == '`' || c == '|' ||
         c == '~';
}

[[nodiscard]] bool ident_body(char c) {
  return ident_start(c) || std::isdigit(static_cast<unsigned char>(c)) != 0 || c == '.';
}

/// Section kinds, in the order they may appear.
enum class LpSection { kObjective, kConstraints, kBounds, kGeneral, kBinary, kEnd };

struct SectionSpan {
  LpSection kind;
  std::size_t begin;  ///< first token of the section body
  std::size_t end;    ///< one past the last
  bool maximize = false;
};

class LpParser {
 public:
  explicit LpParser(Model* model) : model_(model) {}
  ReadResult parse(const std::string& path);

 private:
  [[nodiscard]] bool lex(const std::string& path, std::string* error);
  [[nodiscard]] bool split_sections(std::string* error);

  [[nodiscard]] bool parse_objective(const SectionSpan& span, std::string* error);
  [[nodiscard]] bool parse_constraints(const SectionSpan& span, std::string* error);
  [[nodiscard]] bool parse_bounds(const SectionSpan& span, std::string* error);
  [[nodiscard]] bool parse_integrality(const SectionSpan& span, bool binary, std::string* error);

  /// Parse a signed linear expression starting at `i`, stopping before the first token that
  /// cannot continue it. Coefficients accumulate into `terms` keyed by column, and any bare
  /// number accumulates into `constant`.
  [[nodiscard]] bool parse_expression(std::size_t* i, std::size_t end,
                                      std::unordered_map<Index, double>* terms, double* constant,
                                      std::string* error);

  [[nodiscard]] Index intern_column(const std::string& name);
  [[nodiscard]] std::string at(std::size_t i, const std::string& message) const;

  Model* model_;
  std::string path_;
  std::vector<Tok> tokens_;
  std::vector<SectionSpan> sections_;

  std::unordered_map<std::string, Index> col_index_;
  std::vector<std::string> col_names_;
  std::vector<double> col_cost_;
  std::vector<double> col_lower_;
  std::vector<double> col_upper_;
  std::vector<VarType> col_type_;

  std::vector<std::string> row_names_;
  std::vector<double> row_lower_;
  std::vector<double> row_upper_;
  std::vector<Index> tri_row_;
  std::vector<Index> tri_col_;
  std::vector<double> tri_value_;
  Count unnamed_rows_ = 0;
};

std::string LpParser::at(std::size_t i, const std::string& message) const {
  const Count line = i < tokens_.size() ? tokens_[i].line
                                        : (tokens_.empty() ? 0 : tokens_.back().line);
  return fmt::format("{}:{}: {}", path_, line, message);
}

Index LpParser::intern_column(const std::string& name) {
  const auto it = col_index_.find(name);
  if (it != col_index_.end()) return it->second;
  const auto index = static_cast<Index>(col_names_.size());
  col_index_.emplace(name, index);
  col_names_.push_back(name);
  col_cost_.push_back(0.0);
  col_lower_.push_back(0.0);  // the LP-format default, as in MPS
  col_upper_.push_back(kInfinity);
  col_type_.push_back(VarType::kContinuous);
  return index;
}

bool LpParser::lex(const std::string& path, std::string* error) {
  LineReader reader;
  if (!reader.open(path, error)) return false;

  std::string line;
  while (reader.next(&line)) {
    const Count line_number = reader.line_number();
    bool first = true;
    std::size_t i = 0;
    while (i < line.size()) {
      const char c = line[i];
      if (is_space(c)) { ++i; continue; }
      if (c == '\\') break;  // comment to end of line

      Tok tok;
      tok.line = line_number;
      tok.first_on_line = first;

      if (std::isdigit(static_cast<unsigned char>(c)) != 0 ||
          (c == '.' && i + 1 < line.size() &&
           std::isdigit(static_cast<unsigned char>(line[i + 1])) != 0)) {
        const char* begin = line.c_str() + i;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin) {
          *error = fmt::format("{}:{}: cannot parse a number", path, line_number);
          return false;
        }
        tok.kind = TokKind::kNumber;
        tok.value = value;
        tok.text.assign(begin, static_cast<std::size_t>(end - begin));
        i += static_cast<std::size_t>(end - begin);
      } else if (ident_start(c)) {
        const std::size_t start = i;
        while (i < line.size() && ident_body(line[i])) ++i;
        tok.kind = TokKind::kIdent;
        tok.text = line.substr(start, i - start);
      } else {
        tok.kind = TokKind::kOp;
        // Normalise every spelling of a relation to "<=", ">=" or "=".
        if ((c == '<' || c == '>' || c == '=') && i + 1 < line.size() &&
            (line[i + 1] == '=' || line[i + 1] == '<' || line[i + 1] == '>')) {
          const char second = line[i + 1];
          if (c == '<' || second == '<') {
            tok.text = "<=";
          } else if (c == '>' || second == '>') {
            tok.text = ">=";
          } else {
            tok.text = "=";
          }
          i += 2;
        } else if (c == '<') {
          tok.text = "<="; ++i;
        } else if (c == '>') {
          tok.text = ">="; ++i;
        } else {
          tok.text = std::string(1, c);
          ++i;
        }
      }

      first = false;
      tokens_.push_back(std::move(tok));
    }
  }

  Tok eof;
  eof.kind = TokKind::kEof;
  eof.line = reader.line_number();
  tokens_.push_back(std::move(eof));
  return true;
}

bool LpParser::split_sections(std::string* error) {
  // A marker records both where its keyword starts and where the section body begins, so
  // that the previous section's body can end exactly at the next keyword rather than at
  // some token counted back from it.
  struct Marker {
    LpSection kind;
    std::size_t keyword_start;
    std::size_t body_begin;
    bool maximize;
  };
  std::vector<Marker> markers;

  for (std::size_t i = 0; i + 1 < tokens_.size(); ++i) {
    const Tok& t = tokens_[i];
    if (t.kind != TokKind::kIdent || !t.first_on_line) continue;
    const std::string k = to_upper(t.text);
    std::size_t keyword_tokens = 1;
    LpSection kind = LpSection::kEnd;
    bool maximize = false;

    if (k == "MAXIMIZE" || k == "MAXIMISE" || k == "MAX" || k == "MAXIMUM") {
      kind = LpSection::kObjective;
      maximize = true;
    } else if (k == "MINIMIZE" || k == "MINIMISE" || k == "MIN" || k == "MINIMUM") {
      kind = LpSection::kObjective;
    } else if (k == "ST" || k == "ST." || k == "S.T." || k == "S.T") {
      kind = LpSection::kConstraints;
    } else if (k == "SUBJECT" || k == "SUCH") {
      // "subject to" / "such that": the following token completes the keyword.
      const std::string next = to_upper(tokens_[i + 1].text);
      if (next != "TO" && next != "THAT") continue;
      kind = LpSection::kConstraints;
      keyword_tokens = 2;
    } else if (k == "BOUNDS" || k == "BOUND") {
      kind = LpSection::kBounds;
    } else if (k == "GENERAL" || k == "GENERALS" || k == "GEN" || k == "INTEGER" ||
               k == "INTEGERS") {
      kind = LpSection::kGeneral;
    } else if (k == "BINARY" || k == "BINARIES" || k == "BIN") {
      kind = LpSection::kBinary;
    } else if (k == "END") {
      kind = LpSection::kEnd;
    } else if (k == "SOS" || k == "SEMI" || k == "SEMIS" || k == "SEMI-CONTINUOUS") {
      *error = at(i, fmt::format("the {} section is not supported", t.text));
      return false;
    } else {
      continue;
    }

    markers.push_back({kind, i, i + keyword_tokens, maximize});
    if (kind == LpSection::kEnd) break;
  }

  if (markers.empty() || markers.front().kind != LpSection::kObjective) {
    *error = fmt::format("{}: no Maximize or Minimize section", path_);
    return false;
  }

  const std::size_t last_token = tokens_.size() - 1;  // the kEof sentinel
  for (std::size_t s = 0; s < markers.size(); ++s) {
    if (markers[s].kind == LpSection::kEnd) break;
    const std::size_t body_end =
        (s + 1 < markers.size()) ? markers[s + 1].keyword_start : last_token;
    sections_.push_back(
        {markers[s].kind, markers[s].body_begin, body_end, markers[s].maximize});
  }
  return true;
}

bool LpParser::parse_expression(std::size_t* i, std::size_t end,
                                std::unordered_map<Index, double>* terms, double* constant,
                                std::string* error) {
  double sign = 1.0;
  bool expect_term = true;

  while (*i < end) {
    const Tok& t = tokens_[*i];

    if (t.kind == TokKind::kOp && (t.text == "+" || t.text == "-")) {
      sign = (t.text == "-") ? -sign : sign;
      ++(*i);
      expect_term = true;
      continue;
    }
    if (t.kind == TokKind::kOp && t.text == "[") {
      *error = at(*i, "quadratic objective terms are not supported by this reader yet");
      return false;
    }
    if (!expect_term && (t.kind == TokKind::kNumber || t.kind == TokKind::kIdent)) break;
    if (t.kind == TokKind::kOp || t.kind == TokKind::kEof) break;

    double coefficient = 1.0;
    bool had_number = false;
    if (t.kind == TokKind::kNumber) {
      coefficient = t.value;
      had_number = true;
      ++(*i);
      // An optional '*' between the coefficient and the variable.
      if (*i < end && tokens_[*i].kind == TokKind::kOp && tokens_[*i].text == "*") ++(*i);
    }

    if (*i < end && tokens_[*i].kind == TokKind::kIdent) {
      const Index col = intern_column(tokens_[*i].text);
      (*terms)[col] += sign * coefficient;
      ++(*i);
    } else if (had_number) {
      *constant += sign * coefficient;
    } else {
      *error = at(*i, "expected a variable name or a number");
      return false;
    }

    sign = 1.0;
    expect_term = false;
  }
  return true;
}

bool LpParser::parse_objective(const SectionSpan& span, std::string* error) {
  model_->sense = span.maximize ? ObjSense::kMaximize : ObjSense::kMinimize;

  std::size_t i = span.begin;
  // An optional "name:" label.
  if (i + 1 < span.end && tokens_[i].kind == TokKind::kIdent &&
      tokens_[i + 1].kind == TokKind::kOp && tokens_[i + 1].text == ":") {
    i += 2;
  }

  std::unordered_map<Index, double> terms;
  double constant = 0.0;
  if (!parse_expression(&i, span.end, &terms, &constant, error)) return false;
  if (i != span.end) {
    *error = at(i, fmt::format("unexpected '{}' in the objective", tokens_[i].text));
    return false;
  }

  model_->objective_offset = constant;
  for (const auto& [col, coefficient] : terms) {
    col_cost_[static_cast<std::size_t>(col)] = coefficient;
  }
  return true;
}

bool LpParser::parse_constraints(const SectionSpan& span, std::string* error) {
  std::size_t i = span.begin;

  while (i < span.end) {
    std::string name;
    if (i + 1 < span.end && tokens_[i].kind == TokKind::kIdent &&
        tokens_[i + 1].kind == TokKind::kOp && tokens_[i + 1].text == ":") {
      name = tokens_[i].text;
      i += 2;
    } else {
      name = fmt::format("R{}", ++unnamed_rows_);
    }

    // A range constraint opens with "constant relop": -3 <= x - y <= 5. The sign is a
    // separate token, so this has to look past one or more leading '+'/'-' before deciding
    // whether the statement really begins with a bound. It is a LOOKAHEAD: nothing is
    // consumed unless the whole pattern matches, otherwise a plain constraint that happens
    // to start with a negative coefficient ("-3 x + y >= 2") would lose its first term.
    bool have_left = false;
    double left = 0.0;
    std::string left_op;
    {
      std::size_t probe = i;
      double sign = 1.0;
      while (probe < span.end && tokens_[probe].kind == TokKind::kOp &&
             (tokens_[probe].text == "+" || tokens_[probe].text == "-")) {
        if (tokens_[probe].text == "-") sign = -sign;
        ++probe;
      }
      if (probe + 1 < span.end && tokens_[probe].kind == TokKind::kNumber &&
          is_relop(tokens_[probe + 1])) {
        left = sign * tokens_[probe].value;
        left_op = tokens_[probe + 1].text;
        have_left = true;
        i = probe + 2;
      }
    }

    std::unordered_map<Index, double> terms;
    double constant = 0.0;
    if (!parse_expression(&i, span.end, &terms, &constant, error)) return false;

    if (i >= span.end || !is_relop(tokens_[i])) {
      *error = at(i, "constraint has no relational operator");
      return false;
    }
    const std::string op = tokens_[i].text;
    ++i;
    // The right-hand side carries its sign as its own token too: "x + y >= -5".
    double right_sign = 1.0;
    while (i < span.end && tokens_[i].kind == TokKind::kOp &&
           (tokens_[i].text == "+" || tokens_[i].text == "-")) {
      if (tokens_[i].text == "-") right_sign = -right_sign;
      ++i;
    }
    if (i >= span.end || tokens_[i].kind != TokKind::kNumber) {
      *error = at(i, "expected a number on the right of the relational operator");
      return false;
    }
    const double right = right_sign * tokens_[i].value;
    ++i;

    // A constant that appeared inside the expression moves to the right-hand side.
    double lo = -kInfinity;
    double hi = kInfinity;
    if (op == "<=") {
      hi = right - constant;
    } else if (op == ">=") {
      lo = right - constant;
    } else {
      lo = right - constant;
      hi = lo;
    }

    if (have_left) {
      const double bound = left - constant;
      if (left_op == "<=") {
        lo = std::max(lo, bound);
      } else if (left_op == ">=") {
        hi = std::min(hi, bound);
      } else {
        lo = std::max(lo, bound);
        hi = std::min(hi, bound);
      }
    }

    if (lo > hi) {
      *error = at(i, fmt::format("constraint '{}' has an empty range [{}, {}]", name, lo, hi));
      return false;
    }

    const auto row = static_cast<Index>(row_names_.size());
    row_names_.push_back(name);
    row_lower_.push_back(lo);
    row_upper_.push_back(hi);
    for (const auto& [col, coefficient] : terms) {
      if (coefficient == 0.0) continue;
      tri_row_.push_back(row);
      tri_col_.push_back(col);
      tri_value_.push_back(coefficient);
    }
  }
  return true;
}

bool LpParser::parse_bounds(const SectionSpan& span, std::string* error) {
  // A bound value may be written as "inf", "infinity", "-inf", "+infinity".
  const auto read_bound = [&](std::size_t* i, double* out) -> bool {
    double sign = 1.0;
    while (*i < span.end && tokens_[*i].kind == TokKind::kOp &&
           (tokens_[*i].text == "+" || tokens_[*i].text == "-")) {
      if (tokens_[*i].text == "-") sign = -sign;
      ++(*i);
    }
    if (*i < span.end && tokens_[*i].kind == TokKind::kNumber) {
      *out = sign * tokens_[*i].value;
      ++(*i);
      return true;
    }
    if (*i < span.end && tokens_[*i].kind == TokKind::kIdent) {
      const std::string k = to_upper(tokens_[*i].text);
      if (k == "INF" || k == "INFINITY") {
        *out = sign * kInfinity;
        ++(*i);
        return true;
      }
    }
    return false;
  };

  std::size_t i = span.begin;
  while (i < span.end) {
    // "x free"
    if (tokens_[i].kind == TokKind::kIdent && i + 1 < span.end &&
        tokens_[i + 1].kind == TokKind::kIdent && to_upper(tokens_[i + 1].text) == "FREE") {
      const Index col = intern_column(tokens_[i].text);
      col_lower_[static_cast<std::size_t>(col)] = -kInfinity;
      col_upper_[static_cast<std::size_t>(col)] = kInfinity;
      i += 2;
      continue;
    }

    double lower = 0.0;
    const std::size_t start = i;
    const bool leading_value = read_bound(&i, &lower);

    if (leading_value) {
      // "lo <= x [<= hi]"
      if (i >= span.end || !is_relop(tokens_[i]) || i + 1 >= span.end ||
          tokens_[i + 1].kind != TokKind::kIdent) {
        *error = at(start, "malformed bound statement");
        return false;
      }
      const std::string op = tokens_[i].text;
      const Index col = intern_column(tokens_[i + 1].text);
      const auto u = static_cast<std::size_t>(col);
      i += 2;

      if (op == "<=") {
        col_lower_[u] = lower;
      } else if (op == ">=") {
        col_upper_[u] = lower;
      } else {
        col_lower_[u] = lower;
        col_upper_[u] = lower;
      }

      if (i < span.end && is_relop(tokens_[i])) {
        const std::string op2 = tokens_[i].text;
        ++i;
        double upper = 0.0;
        if (!read_bound(&i, &upper)) {
          *error = at(i, "expected a bound value");
          return false;
        }
        if (op2 == "<=") {
          col_upper_[u] = upper;
        } else if (op2 == ">=") {
          col_lower_[u] = upper;
        } else {
          col_lower_[u] = upper;
          col_upper_[u] = upper;
        }
      }
      continue;
    }

    // "x relop value"
    if (tokens_[i].kind != TokKind::kIdent) {
      *error = at(i, "malformed bound statement");
      return false;
    }
    const Index col = intern_column(tokens_[i].text);
    const auto u = static_cast<std::size_t>(col);
    ++i;
    if (i >= span.end || !is_relop(tokens_[i])) {
      *error = at(i, "expected a relational operator in a bound statement");
      return false;
    }
    const std::string op = tokens_[i].text;
    ++i;
    double value = 0.0;
    if (!read_bound(&i, &value)) {
      *error = at(i, "expected a bound value");
      return false;
    }
    if (op == "<=") {
      col_upper_[u] = value;
    } else if (op == ">=") {
      col_lower_[u] = value;
    } else {
      col_lower_[u] = value;
      col_upper_[u] = value;
    }
  }
  return true;
}

bool LpParser::parse_integrality(const SectionSpan& span, bool binary, std::string* error) {
  for (std::size_t i = span.begin; i < span.end; ++i) {
    if (tokens_[i].kind != TokKind::kIdent) {
      *error = at(i, "expected a variable name");
      return false;
    }
    const Index col = intern_column(tokens_[i].text);
    const auto u = static_cast<std::size_t>(col);
    col_type_[u] = VarType::kInteger;
    if (binary) {
      col_lower_[u] = 0.0;
      col_upper_[u] = 1.0;
    }
  }
  return true;
}

ReadResult LpParser::parse(const std::string& path) {
  path_ = path;
  *model_ = Model{};
  model_->source_path = path;

  std::string error;
  if (!lex(path, &error)) return ReadResult::failure(error);
  if (!split_sections(&error)) return ReadResult::failure(error);

  for (const SectionSpan& span : sections_) {
    bool ok = true;
    switch (span.kind) {
      case LpSection::kObjective: ok = parse_objective(span, &error); break;
      case LpSection::kConstraints: ok = parse_constraints(span, &error); break;
      case LpSection::kBounds: ok = parse_bounds(span, &error); break;
      case LpSection::kGeneral: ok = parse_integrality(span, false, &error); break;
      case LpSection::kBinary: ok = parse_integrality(span, true, &error); break;
      case LpSection::kEnd: break;
    }
    if (!ok) return ReadResult::failure(error);
  }

  const auto n = static_cast<Index>(col_names_.size());
  const auto m = static_cast<Index>(row_names_.size());
  model_->col_cost = col_cost_;
  model_->col_lower = col_lower_;
  model_->col_upper = col_upper_;
  model_->col_type = col_type_;
  model_->col_names = col_names_;
  model_->row_names = row_names_;
  model_->row_lower = row_lower_;
  model_->row_upper = row_upper_;

  model_->matrix.reset(m, n);
  model_->matrix.reserve(tri_row_.size());
  for (std::size_t k = 0; k < tri_row_.size(); ++k) {
    model_->matrix.add_entry(tri_row_[k], tri_col_[k], tri_value_[k]);
  }
  model_->matrix.finalize();
  model_->hessian.reset(n, n);
  model_->hessian.finalize();

  const std::string problem = model_->validate();
  if (!problem.empty()) {
    return ReadResult::failure(fmt::format("{}: model failed validation: {}", path, problem));
  }
  return ReadResult::success();
}

}  // namespace

ReadResult read_lp(const std::string& path, Model* model) {
  LpParser parser(model);
  return parser.parse(path);
}

ReadResult read_model(const std::string& path, Model* model) {
  std::string lowered = path;
  for (char& c : lowered) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (lowered.size() >= 3 && lowered.compare(lowered.size() - 3, 3, ".gz") == 0) {
    lowered.resize(lowered.size() - 3);
  }
  if (lowered.size() >= 3 && lowered.compare(lowered.size() - 3, 3, ".lp") == 0) {
    return read_lp(path, model);
  }
  return read_mps(path, model);
}

}  // namespace sankhya::io

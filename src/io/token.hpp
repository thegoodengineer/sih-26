// SPDX-License-Identifier: Apache-2.0
// SANKHYA - lexical helpers shared by the MPS and LP readers.
//
// parse_double is the one that matters. std::stod throws, std::atof cannot report failure,
// and a silently-zero coefficient in a constraint matrix is precisely the bug class that
// produces a confident wrong optimum. This wrapper reports failure explicitly, rejects
// trailing garbage, and normalises the Fortran 'D' exponent that older MPS writers emit
// (1.5D+02), which strtod does not recognise and would otherwise truncate to 1.5.
#pragma once

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace sankhya::io {

[[nodiscard]] inline bool is_space(char c) noexcept {
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

[[nodiscard]] inline std::string_view trim(std::string_view s) noexcept {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && is_space(s[b])) ++b;
  while (e > b && is_space(s[e - 1])) --e;
  return s.substr(b, e - b);
}

[[nodiscard]] inline std::string to_upper(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return out;
}

/// Split on runs of whitespace. Empty tokens are never produced.
inline void tokenize(std::string_view line, std::vector<std::string_view>* out) {
  out->clear();
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && is_space(line[i])) ++i;
    if (i >= line.size()) break;
    const std::size_t start = i;
    while (i < line.size() && !is_space(line[i])) ++i;
    out->push_back(line.substr(start, i - start));
  }
}

/// Strip a single pair of surrounding single or double quotes, as MPS MARKER records use.
[[nodiscard]] inline std::string_view unquote(std::string_view s) noexcept {
  if (s.size() >= 2 && (s.front() == '\'' || s.front() == '"') && s.back() == s.front()) {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

/// Parse a floating point number. Returns false on an empty field, an unparsable field, or
/// trailing non-space characters. Out-of-range magnitudes are accepted as +/-HUGE_VAL,
/// which the callers then normalise through normalize_infinity().
[[nodiscard]] inline bool parse_double(std::string_view text, double* out) {
  const std::string_view t = trim(text);
  if (t.empty()) return false;

  std::string buffer(t);
  for (char& c : buffer) {
    if (c == 'D' || c == 'd') c = 'E';  // Fortran-style exponent from older MPS writers
  }

  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(buffer.c_str(), &end);
  if (end == buffer.c_str()) return false;
  while (end != nullptr && *end != '\0' && is_space(*end)) ++end;
  if (end == nullptr || *end != '\0') return false;

  // strtod happily accepts "nan", "NaN" and "-nan". No field of an MPS or LP file can
  // usefully hold one: a NaN bound, right-hand side or coefficient makes the model
  // meaningless rather than merely extreme. Rejecting it here means the reader reports the
  // offending line instead of the value travelling on to be diagnosed later, or worse
  // vanishing - a NaN fails every comparison, so a zero-dropping pass deletes it and leaves
  // a well formed model of a different problem.
  //
  // Infinity is NOT rejected here. It is meaningless in a coefficient but legitimate in a
  // bound, where MPS spells it 1e30 and where readers normalise anything that large to a
  // true infinity. The distinction is per-field, so it is enforced at the call sites that
  // read coefficients rather than in this shared parser.
  if (std::isnan(value)) return false;

  *out = value;
  return true;
}

/// Parse a signed integer with the same strictness as parse_double.
[[nodiscard]] inline bool parse_int(std::string_view text, long long* out) {
  const std::string_view t = trim(text);
  if (t.empty()) return false;
  const std::string buffer(t);
  errno = 0;
  char* end = nullptr;
  const long long value = std::strtoll(buffer.c_str(), &end, 10);
  if (end == buffer.c_str()) return false;
  while (end != nullptr && *end != '\0' && is_space(*end)) ++end;
  if (end == nullptr || *end != '\0') return false;
  *out = value;
  return true;
}

}  // namespace sankhya::io

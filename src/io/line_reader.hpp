// SPDX-License-Identifier: Apache-2.0
// SANKHYA - buffered line input for the model readers.
//
// One class serves both readers and both compression states. When the build has zlib we go
// through gzread unconditionally: zlib reads an uncompressed file transparently, so there
// is no format sniffing and no second code path to keep in step. Without zlib we fall back
// to std::ifstream and reject .gz input with a clear message rather than producing garbage.
//
// Lines are delivered without their terminator and with a trailing '\r' removed, so a
// Windows-authored MPS file parses identically to a Unix one. That matters more than it
// sounds: a stray '\r' glued to the last token of a BOUNDS line turns "UP" into "UP\r",
// which a naive reader silently treats as an unknown bound type.
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya::io {

class LineReader {
 public:
  LineReader() = default;
  ~LineReader();

  LineReader(const LineReader&) = delete;
  LineReader& operator=(const LineReader&) = delete;

  /// Open `path`. Returns false and fills `error` when the file cannot be read.
  [[nodiscard]] bool open(const std::string& path, std::string* error);

  /// Fetch the next line into `line`, stripped of its terminator. Returns false at EOF.
  [[nodiscard]] bool next(std::string* line);

  /// 1-based number of the line most recently returned by next().
  [[nodiscard]] Count line_number() const noexcept { return line_number_; }

  /// Format "path:line: message", the error shape every reader emits.
  [[nodiscard]] std::string error_at(const std::string& message) const;

  void close();

 private:
  /// Refill `buffer_` from the underlying stream. Returns bytes read, 0 at EOF.
  [[nodiscard]] std::size_t fill();

  std::string path_;
  Count line_number_ = 0;

  void* gz_ = nullptr;  ///< gzFile when built with zlib, else always null
  std::ifstream stream_;

  std::vector<char> buffer_;
  std::size_t begin_ = 0;  ///< first unconsumed byte in buffer_
  std::size_t end_ = 0;    ///< one past the last valid byte in buffer_
  bool eof_ = false;
};

}  // namespace sankhya::io

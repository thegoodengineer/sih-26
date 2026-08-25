// SPDX-License-Identifier: Apache-2.0
// SANKHYA - buffered line input, implementation.

#include "line_reader.hpp"

#include <algorithm>
#include <cstring>

#include <fmt/format.h>

#ifdef SANKHYA_WITH_ZLIB
#include <zlib.h>
#endif

namespace sankhya::io {
namespace {

/// 64 KiB. Large enough that the syscall cost disappears against the parse cost, small
/// enough to stay in L2 while the reader tokenizes out of it.
constexpr std::size_t kChunkBytes = 64 * 1024;

#ifndef SANKHYA_WITH_ZLIB
[[nodiscard]] bool has_gz_suffix(const std::string& path) {
  return path.size() >= 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
}
#endif

}  // namespace

LineReader::~LineReader() {
  close();
}

bool LineReader::open(const std::string& path, std::string* error) {
  close();
  path_ = path;
  line_number_ = 0;
  begin_ = 0;
  end_ = 0;
  eof_ = false;
  buffer_.assign(kChunkBytes, '\0');

#ifdef SANKHYA_WITH_ZLIB
  gz_ = gzopen(path.c_str(), "rb");
  if (gz_ == nullptr) {
    if (error != nullptr) *error = fmt::format("{}: cannot open file", path);
    return false;
  }
  gzbuffer(static_cast<gzFile>(gz_), static_cast<unsigned>(kChunkBytes));
  return true;
#else
  if (has_gz_suffix(path)) {
    if (error != nullptr) {
      *error =
          fmt::format("{}: gzip input needs a build with zlib (-DSANKHYA_WITH_ZLIB=ON)", path);
    }
    return false;
  }
  stream_.open(path, std::ios::binary);
  if (!stream_.is_open()) {
    if (error != nullptr) *error = fmt::format("{}: cannot open file", path);
    return false;
  }
  return true;
#endif
}

void LineReader::close() {
#ifdef SANKHYA_WITH_ZLIB
  if (gz_ != nullptr) {
    gzclose(static_cast<gzFile>(gz_));
    gz_ = nullptr;
  }
#endif
  if (stream_.is_open()) stream_.close();
  buffer_.clear();
  begin_ = 0;
  end_ = 0;
  eof_ = true;
}

std::size_t LineReader::fill() {
  begin_ = 0;
  end_ = 0;
  if (eof_) return 0;

#ifdef SANKHYA_WITH_ZLIB
  const int got =
      gzread(static_cast<gzFile>(gz_), buffer_.data(), static_cast<unsigned>(buffer_.size()));
  if (got <= 0) {
    eof_ = true;
    return 0;
  }
  end_ = static_cast<std::size_t>(got);
#else
  stream_.read(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
  const std::streamsize got = stream_.gcount();
  if (got <= 0) {
    eof_ = true;
    return 0;
  }
  end_ = static_cast<std::size_t>(got);
#endif
  return end_;
}

bool LineReader::next(std::string* line) {
  line->clear();
  bool any = false;

  for (;;) {
    if (begin_ == end_ && fill() == 0) break;

    const char* base = buffer_.data();
    const char* start = base + begin_;
    const std::size_t remaining = end_ - begin_;
    const char* nl = static_cast<const char*>(std::memchr(start, '\n', remaining));

    if (nl == nullptr) {
      line->append(start, remaining);
      begin_ = end_;
      any = true;
      continue;
    }

    const auto take = static_cast<std::size_t>(nl - start);
    line->append(start, take);
    begin_ += take + 1;
    any = true;
    break;
  }

  if (!any && line->empty()) return false;

  // A file authored on Windows and read as binary leaves the CR attached to the last token.
  if (!line->empty() && line->back() == '\r') line->pop_back();
  ++line_number_;
  return true;
}

std::string LineReader::error_at(const std::string& message) const {
  return fmt::format("{}:{}: {}", path_, line_number_, message);
}

}  // namespace sankhya::io

// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a scratch file that deletes itself, for the reader tests.
//
// The reader tests are table-driven over dozens of small MPS and LP fragments. Keeping
// those fragments as string literals inside the test file rather than as checked-in data
// files means the expected parse and the input sit on the same screen, which is what makes
// a wrong RANGES row obvious during review instead of during a benchmark run three weeks
// later.
#pragma once

#include <cstdio>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

namespace sankhya::testing {

class TempFile {
 public:
  /// Write `contents` to a uniquely named file with the given extension.
  TempFile(std::string_view contents, const char* extension = ".mps") {
    static int counter = 0;
    path_ = std::string("sankhya_test_") + std::to_string(counter++) + extension;
    std::FILE* out = std::fopen(path_.c_str(), "wb");
    if (out == nullptr) {
      ADD_FAILURE() << "cannot create scratch file " << path_;
      return;
    }
    std::fwrite(contents.data(), 1, contents.size(), out);
    std::fclose(out);
  }

  ~TempFile() {
    if (!path_.empty()) std::remove(path_.c_str());
  }

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  std::string path_;
};

}  // namespace sankhya::testing

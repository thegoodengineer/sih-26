// SPDX-License-Identifier: Apache-2.0
// SANKHYA - build identification. The macros are supplied by CMake.

#include "sankhya/version.hpp"

#include <string>

#include <fmt/format.h>

#ifndef SANKHYA_VERSION
#define SANKHYA_VERSION "0.0.0-unconfigured"
#endif
#ifndef SANKHYA_GIT_COMMIT
#define SANKHYA_GIT_COMMIT "unknown"
#endif
#ifndef SANKHYA_BUILD_TYPE
#define SANKHYA_BUILD_TYPE "unknown"
#endif
#ifndef SANKHYA_COMPILER
#define SANKHYA_COMPILER "unknown"
#endif

namespace sankhya {

const char* version_string() noexcept {
  return SANKHYA_VERSION;
}
const char* git_commit() noexcept {
  return SANKHYA_GIT_COMMIT;
}
const char* build_type() noexcept {
  return SANKHYA_BUILD_TYPE;
}
const char* compiler_string() noexcept {
  return SANKHYA_COMPILER;
}

bool cuda_enabled() noexcept {
#ifdef SANKHYA_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

const char* banner() noexcept {
  static const std::string text =
      fmt::format("SANKHYA {} ({}, {}, {}, CUDA {})", version_string(), git_commit(),
                  build_type(), compiler_string(), cuda_enabled() ? "on" : "off");
  return text.c_str();
}

}  // namespace sankhya

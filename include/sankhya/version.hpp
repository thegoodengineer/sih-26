// SPDX-License-Identifier: Apache-2.0
// SANKHYA - build identification.
//
// Every benchmark CSV records the git commit (CLAUDE.md), so the commit has to be
// reachable from inside the binary rather than from whatever shell produced the CSV.
#pragma once

namespace sankhya {

/// Semantic version of the library, e.g. "0.1.0".
[[nodiscard]] const char* version_string() noexcept;

/// Short git commit the binary was built from, or "unknown" outside a git checkout.
[[nodiscard]] const char* git_commit() noexcept;

/// CMake build type: Release, Debug, RelWithDebInfo.
[[nodiscard]] const char* build_type() noexcept;

/// Compiler identification, e.g. "GNU 13.2.0".
[[nodiscard]] const char* compiler_string() noexcept;

/// True when the binary was compiled with the CUDA backend available. Note this says
/// nothing about whether a device is present at run time.
[[nodiscard]] bool cuda_enabled() noexcept;

/// One-line banner: name, version, commit, build type.
[[nodiscard]] const char* banner() noexcept;

}  // namespace sankhya

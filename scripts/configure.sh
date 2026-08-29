#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SANKHYA - configure a build tree with a C++20-capable compiler.
#
# Windows dev boxes vary: one carries a MinGW 6.3.0 on PATH that cannot do C++20 at all,
# another has MSYS2 UCRT64 GCC 16. This script probes known-good locations and verifies the
# major version rather than leaving the choice to PATH order. It also puts the toolchain's
# own bin directory on PATH so the cmake/ninja that ship beside the compiler are found.
#
# Usage: scripts/configure.sh [build-dir] [Release|Debug] [extra cmake args...]
set -euo pipefail

BUILD_DIR="${1:-build}"
BUILD_TYPE="${2:-Release}"
shift 2 2>/dev/null || true

pick_compiler() {
  local candidates=(
    "/c/msys64/ucrt64/bin/g++.exe"
    "/c/msys64/mingw64/bin/g++.exe"
    "/c/Strawberry/c/bin/g++.exe"
    "$(command -v g++-13 2>/dev/null || true)"
    "$(command -v g++-12 2>/dev/null || true)"
    "$(command -v g++ 2>/dev/null || true)"
  )
  for cxx in "${candidates[@]}"; do
    [ -n "$cxx" ] && [ -x "$cxx" ] || continue
    local major
    major="$("$cxx" -dumpversion 2>/dev/null | cut -d. -f1)"
    if [ -n "$major" ] && [ "$major" -ge 10 ]; then
      echo "$cxx"
      return 0
    fi
  done
  echo "error: no C++20-capable g++ found (need GCC 10 or newer)" >&2
  return 1
}

CXX_BIN="$(pick_compiler)"
CC_BIN="${CXX_BIN%g++*}gcc${CXX_BIN##*g++}"
echo "sankhya: using ${CXX_BIN} ($("$CXX_BIN" -dumpversion))"

# MSYS2 and Strawberry ship cmake/ninja next to the compiler. Prepending the toolchain bin
# directory means we pick those up instead of an unrelated cmake that targets another ABI.
TOOLCHAIN_BIN="$(dirname "$CXX_BIN")"
case ":$PATH:" in
  *":$TOOLCHAIN_BIN:"*) ;;
  *) PATH="$TOOLCHAIN_BIN:$PATH"; export PATH ;;
esac

command -v cmake >/dev/null 2>&1 || {
  echo "error: cmake not found on PATH or in ${TOOLCHAIN_BIN}" >&2
  echo "       MSYS2: pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja" >&2
  exit 1
}

# REUSE DEPENDENCY SOURCES FROM ANY EXISTING BUILD TREE.
#
# FetchContent clones fmt, CLI11, nlohmann_json, googletest and sometimes zlib into
# <build>/_deps - about 209 MB, and several minutes on a slow link. That cost is paid again
# for every new build directory, which would be a footnote except that creating a new build
# directory is the standard remedy for Windows Smart App Control blocking a freshly linked
# binary by hash. The recovery path a teammate needs on a demo day should not be the one that
# takes ten minutes and a network.
#
# Only the SOURCE directories are shared. They are configuration-independent - the same tag
# checked out from the same repository - so a Release tree and a Debug tree can read the same
# checkout safely. The per-dependency build outputs stay inside each tree, which is why this
# uses FETCHCONTENT_SOURCE_DIR_<name> rather than FETCHCONTENT_BASE_DIR: pointing the base
# directory at a shared location would also share the build trees, and Release and Debug
# would then fight over them.
FETCH_ARGS=()
for existing in "$(dirname "$BUILD_DIR")"/*/_deps; do
  [ -d "$existing" ] || continue
  [ "$existing" = "${BUILD_DIR%/}/_deps" ] && continue
  for src in "$existing"/*-src; do
    [ -d "$src" ] || continue
    name="$(basename "$src")"; name="${name%-src}"
    # CMake upper-cases the declared name for this variable.
    upper="$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')"
    already=0
    for arg in ${FETCH_ARGS[@]+"${FETCH_ARGS[@]}"}; do
      case "$arg" in "-DFETCHCONTENT_SOURCE_DIR_${upper}="*) already=1 ;; esac
    done
    [ "$already" = 1 ] && continue
    FETCH_ARGS+=("-DFETCHCONTENT_SOURCE_DIR_${upper}=$(cd "$src" && pwd)")
  done
done
if [ "${#FETCH_ARGS[@]}" -gt 0 ]; then
  echo "sankhya: reusing ${#FETCH_ARGS[@]} dependency source(s) already on disk"
fi

CC="$CC_BIN" CXX="$CXX_BIN" cmake -G Ninja -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ${FETCH_ARGS[@]+"${FETCH_ARGS[@]}"} "$@"

echo "sankhya: configured ${BUILD_DIR} (${BUILD_TYPE})"
echo "         build with: cmake --build ${BUILD_DIR} -j"
echo "         test  with: ctest --test-dir ${BUILD_DIR} --output-on-failure"

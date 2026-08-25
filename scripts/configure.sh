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

CC="$CC_BIN" CXX="$CXX_BIN" cmake -G Ninja -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "$@"

echo "sankhya: configured ${BUILD_DIR} (${BUILD_TYPE})"
echo "         build with: cmake --build ${BUILD_DIR} -j"
echo "         test  with: ctest --test-dir ${BUILD_DIR} --output-on-failure"

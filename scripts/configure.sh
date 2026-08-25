#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SANKHYA - configure a build tree with a C++20-capable compiler.
#
# On this project's Windows dev box the default `g++` on PATH is MinGW 6.3.0, which does
# not support C++20 at all. Strawberry Perl ships MinGW-W64 GCC 13.2.0, which does. This
# script picks a usable compiler rather than leaving it to PATH order.
#
# Usage: scripts/configure.sh [build-dir] [Release|Debug] [extra cmake args...]
set -euo pipefail

BUILD_DIR="${1:-build}"
BUILD_TYPE="${2:-Release}"
shift 2 2>/dev/null || true

pick_compiler() {
  local candidates=(
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

CC="$CC_BIN" CXX="$CXX_BIN" cmake -G Ninja -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "$@"

echo "sankhya: configured ${BUILD_DIR} (${BUILD_TYPE})"
echo "         build with: cmake --build ${BUILD_DIR} -j"
echo "         test  with: ctest --test-dir ${BUILD_DIR} --output-on-failure"

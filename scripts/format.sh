#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SANKHYA - run the pinned clang-format over the tree.
#
# CI gates formatting with clang-format 22.1.8 installed from pip, and clang-format's output
# changes between major versions - so a system clang-format from a distro package will
# happily "fix" the tree into a state CI then rejects. This script provisions the exact
# pinned version into a throwaway virtualenv, which is the only way to get a local result
# that agrees with the CI job.
#
# Usage:
#   scripts/format.sh            rewrite every file in place
#   scripts/format.sh --check    report violations and exit non-zero, exactly as CI does
set -euo pipefail

CLANG_FORMAT_VERSION="22.1.8"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Outside the repository on purpose. Keeping it in build/ would put a virtualenv inside the
# CMake build tree, where a `cmake --build` or a `rm -rf build` would churn it, and some
# Windows environments refuse to execute a freshly written .exe from a working directory.
# Keyed by version so bumping the pin provisions a fresh one instead of reusing a stale one.
VENV_DIR="${SANKHYA_FORMAT_VENV:-${TMPDIR:-/tmp}/sankhya-clang-format-${CLANG_FORMAT_VERSION}}"

MODE="write"
if [ "${1:-}" = "--check" ]; then
  MODE="check"
elif [ $# -gt 0 ]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

# Probe by RUNNING each candidate, not by asking whether the name resolves. Windows puts an
# App Execution Alias named python3.exe on PATH that exists, is executable, and does nothing
# but print an advert for the Microsoft Store - so `command -v python3` succeeds on a machine
# with no python3 at all.
pick_python() {
  local candidates=("${PYTHON:-}" python3 python py)
  for candidate in "${candidates[@]}"; do
    [ -n "$candidate" ] || continue
    if "$candidate" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  echo "error: no working Python interpreter found (tried: ${candidates[*]})" >&2
  return 1
}

PYTHON="$(pick_python)"

if [ ! -x "${VENV_DIR}/bin/clang-format" ] && [ ! -x "${VENV_DIR}/Scripts/clang-format.exe" ]; then
  echo "sankhya: provisioning clang-format ${CLANG_FORMAT_VERSION} in ${VENV_DIR}"
  "$PYTHON" -m venv "$VENV_DIR"
  if [ -x "${VENV_DIR}/bin/python" ]; then
    "${VENV_DIR}/bin/python" -m pip install --quiet "clang-format==${CLANG_FORMAT_VERSION}"
  else
    "${VENV_DIR}/Scripts/python.exe" -m pip install --quiet "clang-format==${CLANG_FORMAT_VERSION}"
  fi
fi

if [ -x "${VENV_DIR}/bin/clang-format" ]; then
  CLANG_FORMAT="${VENV_DIR}/bin/clang-format"
else
  CLANG_FORMAT="${VENV_DIR}/Scripts/clang-format.exe"
fi

"$CLANG_FORMAT" --version

cd "$REPO_ROOT"
# The same file set the CI job walks. Keep the two in step.
FIND_ARGS=(include src apps tests -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.h' \))

if [ "$MODE" = "check" ]; then
  find "${FIND_ARGS[@]}" -print0 | xargs -0 "$CLANG_FORMAT" --dry-run --Werror
  echo "sankhya: formatting is clean"
else
  find "${FIND_ARGS[@]}" -print0 | xargs -0 "$CLANG_FORMAT" -i
  echo "sankhya: formatted in place; re-run with --check to confirm"
fi

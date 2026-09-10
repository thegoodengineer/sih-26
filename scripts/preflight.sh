#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SANKHYA - check that this machine can build, run and reproduce our claims.
#
# WHY THIS EXISTS. The evidence in this repository is only worth what a reader can
# regenerate, and every failure this script tests for has actually happened on a developer
# box here - each one silently, and each one producing either a demo that stops halfway or,
# worse, one that skips a section and still exits zero. A judge running the demo for the
# first time should find out about a missing dependency from a line that names the fix, not
# from a table that quietly has fewer rows than the README promised.
#
# It distinguishes two severities and nothing else:
#
#   BLOCKER   nothing can run until this is fixed
#   DEGRADED  the demo runs but a specific section will be missing, named here
#
# Exit status is 1 only for blockers, so this is safe to run at the top of a pipeline that
# should still proceed on a machine without, say, highspy.
#
# Usage: scripts/preflight.sh [--quiet]
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

BLOCKERS=0
DEGRADED=0

pass()    { [ "$QUIET" = 1 ] || printf '  \033[32mok\033[0m       %s\n' "$1"; }
degrade() { printf '  \033[33mDEGRADED\033[0m %s\n' "$1"; printf '           %s\n' "$2"; DEGRADED=$((DEGRADED + 1)); }
blocker() { printf '  \033[31mBLOCKER\033[0m  %s\n' "$1"; printf '           %s\n' "$2"; BLOCKERS=$((BLOCKERS + 1)); }

[ "$QUIET" = 1 ] || printf '\nSANKHYA preflight\n=================\n\n'

# ---- Compiler -----------------------------------------------------------------------------
# scripts/configure.sh does the real probing; this only reports what it would find, so that a
# box with nothing but the ancient MinGW says so here rather than at the first -std=c++20.
CXX_FOUND=""
for cxx in /c/msys64/ucrt64/bin/g++.exe /c/msys64/mingw64/bin/g++.exe \
           /c/Strawberry/c/bin/g++.exe "$(command -v g++ 2>/dev/null || true)"; do
  [ -n "$cxx" ] && [ -x "$cxx" ] || continue
  major="$("$cxx" -dumpversion 2>/dev/null | cut -d. -f1)"
  if [ -n "$major" ] && [ "$major" -ge 10 ] 2>/dev/null; then
    CXX_FOUND="$cxx ($("$cxx" -dumpversion))"
    break
  fi
done
if [ -n "$CXX_FOUND" ]; then
  pass "C++20 compiler        $CXX_FOUND"
else
  blocker "no C++20 compiler found" \
    "install MSYS2 UCRT64 GCC, or see the toolchain note in CLAUDE.md. Never let PATH
           order choose - one box here carries a MinGW 6.3.0 that predates C++20 entirely."
fi

for tool in cmake ninja; do
  if command -v "$tool" >/dev/null 2>&1; then
    pass "$tool                 $("$tool" --version 2>/dev/null | head -1)"
  else
    blocker "$tool not on PATH" \
      "pacman -S --needed mingw-w64-ucrt-x86_64-$tool, or use the one beside the compiler -
           scripts/configure.sh prepends the toolchain's own bin directory for this reason."
  fi
done

# ---- Python -------------------------------------------------------------------------------
# THE STORE STUB. On Windows, ~/AppData/Local/Microsoft/WindowsApps/python3 exists as an App
# Execution Alias that only prints "Python was not found". It satisfies `command -v`, which is
# exactly how the demo used to select it, and then every Python step failed with a message
# about the Microsoft Store rather than about the solver.
PYTHON=""
for candidate in "${PYTHON:-}" python3 python py; do
  [ -n "$candidate" ] || continue
  command -v "$candidate" >/dev/null 2>&1 || continue
  if "$candidate" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  blocker "no working Python" \
    "install Python 3.9+. If 'python3' exists but does nothing, it is the Microsoft Store
           stub: Settings > Apps > Advanced app settings > App execution aliases, and turn
           off python.exe and python3.exe."
else
  pass "Python                $("$PYTHON" --version 2>&1) as '$PYTHON'"
  if command -v python3 >/dev/null 2>&1 && ! python3 -c "" >/dev/null 2>&1; then
    degrade "'python3' on PATH is the Microsoft Store stub" \
      "harmless now that this script picked '$PYTHON', but pass PYTHON=$PYTHON to any
           script that defaults to python3, including demo/run_sih_demo.sh."
  fi
fi

# ---- Python packages ----------------------------------------------------------------------
# Both are optional and both silently remove a section of the demo, so they are named with the
# section they take with them rather than as a bare import failure.
if [ -n "$PYTHON" ]; then
  if "$PYTHON" -c "import numpy" >/dev/null 2>&1; then
    pass "numpy                 present"
  else
    degrade "numpy is missing" \
      "demo section 4(b) prints 'needs an SVD; install numpy to see it' instead of the
           condition number.  pip install numpy"
  fi
  if "$PYTHON" -c "import highspy" >/dev/null 2>&1; then
    pass "highspy               present"
  else
    degrade "highspy is missing" \
      "demo section 5 - the solver comparison PS26119 explicitly asks for - does not run
           at all.  pip install highspy   (a separate process; never linked into SANKHYA,
           so it does not affect the sovereignty claim)"
  fi
fi

# ---- Can a freshly built binary actually RUN? ---------------------------------------------
# SMART APP CONTROL. On Windows 11 this blocks unsigned executables by hash and reputation,
# and a binary you just linked has neither. The failure is "Permission denied" from bash or
# "An Application Control policy has blocked this file" from PowerShell, and it hits the test
# runner too - ctest reports sankhya_tests_NOT_BUILT, which reads like a build failure.
# It cost several hours here before it was recognised, so it is checked explicitly.
if [ -x build/sankhya.exe ] || [ -x build/sankhya ]; then
  BIN="build/sankhya.exe"
  [ -x "$BIN" ] || BIN="build/sankhya"
  if "$BIN" version >/dev/null 2>&1; then
    pass "binary executes       $BIN"
  else
    # DEGRADED rather than BLOCKER, deliberately. This is a property of the binary sitting in
    # the tree right now, and the usual next step - rebuilding - is exactly what fixes it.
    # Blocking here would stop scripts/reproduce.sh before the build that resolves the
    # problem, so the hard check lives there instead, applied to the binary that run just
    # linked. Standalone, this is still the warning you want before a demonstration.
    degrade "the binary already in build/ will not execute" \
      "on Windows 11 this is Smart App Control, blocking an unsigned executable that has
           no reputation yet. Rebuilding into a new directory changes the hash and
           sometimes clears it - but only sometimes: two builds of the same commit minutes
           apart have been observed here giving opposite results. Worth one retry, not a
           procedure. Do NOT disable Smart App Control; on Windows 11 that is one-way.
           The lesson that always holds: trust is per-binary, so a build/ that works stops
           working the moment you rebuild it. Do not rebuild before a demonstration."
  fi
else
  [ "$QUIET" = 1 ] || printf '  \033[2m--\033[0m       no build yet; run scripts/reproduce.sh\n'
fi

# ---- Data ---------------------------------------------------------------------------------
COMMITTED=$(ls data/netlib/*.mps 2>/dev/null | wc -l | tr -d ' ')
if [ "$COMMITTED" -gt 0 ]; then
  pass "committed instances   $COMMITTED Netlib .mps files, no network needed"
else
  degrade "no committed Netlib instances" \
    "python bench/runners/fetch_data.py    (needs network)"
fi

# ---- Verdict ------------------------------------------------------------------------------
printf '\n'
if [ "$BLOCKERS" -gt 0 ]; then
  printf '\033[31m%d blocker(s)\033[0m' "$BLOCKERS"
  [ "$DEGRADED" -gt 0 ] && printf ', %d degradation(s)' "$DEGRADED"
  printf '. Fix the blockers above before running the demo.\n\n'
  exit 1
fi
if [ "$DEGRADED" -gt 0 ]; then
  printf '\033[33mReady, with %d degradation(s).\033[0m The demo will run and will SAY what it\n' "$DEGRADED"
  printf 'skipped, but the sections named above will be missing.\n\n'
else
  printf '\033[32mReady.\033[0m Everything the demo needs is present.\n\n'
fi
exit 0

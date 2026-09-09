#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SANKHYA - regenerate every claim this repository makes, from a fresh clone, in one command.
#
# WHAT THIS PROTECTS. Everything here rests on a promise that a reader can reproduce our
# numbers. Until this script existed that promise required knowing which of eight scripts to
# run, in which order, with which flags - which is a treasure hunt, not reproducibility
# (issue #73).
#
# IT RUNS OFFLINE BY DEFAULT. Nine Netlib instances are committed to data/netlib/ with their
# published optima, so the headline benchmark needs no network. That matters more than it
# sounds: the one occasion this has to work without fail is a demonstration on someone
# else's machine, on conference wifi. Pass --fetch-medium to additionally download and run
# the 50-instance medium tier, which is where the honest pass rate lives.
#
# IT SAYS WHAT IT SKIPPED, AND DOES NOT CONFUSE THAT WITH FINDING FAILURES. A step that
# cannot run prints why and the pipeline continues; the summary lists every skip. A
# reproduction script that silently produces fewer results than the README claims is worse
# than one that fails loudly - and one that claims it produced fewer than it did is the same
# failure wearing the other hat. bench/runners/netlib.py exits `0 if passes == total else 1`,
# so on any tier with known failures a completely successful run exits non-zero. That is a
# result, not a skip, and the summary keeps them apart.
#
# Usage: scripts/reproduce.sh [--fetch-medium] [--debug] [--build-dir DIR]
#
#   --fetch-medium  also download and run the 50-instance medium tier (needs network)
#   --debug         build Debug instead of Release. This is the Windows Smart App Control
#                   escape hatch: different bytes, different hash, so it can run where a
#                   blocked Release binary cannot. Same answers; the timing comparison is
#                   skipped under it because it would no longer mean anything.
#   --build-dir     build somewhere other than build/
#   --miplib        also run the MIPLIB set (needs the instances; fetch them first with
#                   bench/runners/fetch_miplib.py --count 30). Off by default because the
#                   30 instances take about twenty minutes at the 60 s limit, which is
#                   longer than everything else here put together.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUILD_DIR="build"
BUILD_TYPE="Release"
FETCH_MEDIUM=0
RUN_MIPLIB=0
while [ $# -gt 0 ]; do
  case "$1" in
    --fetch-medium) FETCH_MEDIUM=1 ;;
    --miplib) RUN_MIPLIB=1 ;;
    # Debug exists here as the Smart App Control escape hatch, not as a developer
    # convenience: it produces different bytes, so it can run where a blocked Release
    # binary cannot. Answers and iteration counts are identical; only timings change.
    --debug) BUILD_TYPE="Debug" ;;
    --build-dir) BUILD_DIR="${2:?--build-dir needs a directory}"; shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

SKIPPED=()
INCOMPLETE=()
STEP=0

rule() { STEP=$((STEP + 1)); printf '\n\033[1m[%d] %s\033[0m\n%s\n' "$STEP" "$1" \
         "$(printf '%.0s-' $(seq 1 78))"; }
skip() { printf '\n  \033[33mSKIPPED: %s\033[0m\n  %s\n' "$1" "$2"; SKIPPED+=("$1"); }
# The step RAN and produced results; it exited non-zero because what it measured has known
# failures, and its own output above names them.
ran_with_failures() {
  printf '\n  \033[33m%s: ran, and reported failures\033[0m\n  %s\n' "$1" "$2"
  INCOMPLETE+=("$1")
}
# Run a benchmark step. A non-zero exit that WROTE a results file is a result; a non-zero
# exit that wrote nothing is a step that could not run. Which file gets written is the
# runner's business - it names them by tier and commit - so this looks for any CSV newer
# than a marker taken just before the run, rather than duplicating that naming here.
run_benchmark() {
  local label="$1"; shift
  local marker; marker="$(mktemp)"
  if "$@"; then rm -f "$marker"; return 0; fi
  local produced
  produced="$(find bench/results "${TMPDIR:-/tmp}" -maxdepth 1 -name '*.csv' -newer "$marker" \
              -print -quit 2>/dev/null)"
  rm -f "$marker"
  if [ -n "$produced" ]; then
    ran_with_failures "$label" "the runner exits non-zero when any instance fails; they are
  named in its output above, and $produced was written"
  else
    skip "$label" "see the output above"
  fi
}

START=$(date +%s)
printf '\n\033[1mSANKHYA - reproducing every claim\033[0m\n'
printf 'commit %s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
# THE TOOLCHAIN IS PART OF THE RESULT. Every number below depends on which compiler built the
# binary and which Python drove the harness, and a reproduction that does not say so cannot be
# compared with docs/PROVENANCE.md section 2b, where the tested toolchains are recorded.
printf 'machine %s\n' "$(uname -s -m 2>/dev/null || echo unknown)"

# ---- 0. Preflight -------------------------------------------------------------------------
rule "Preflight: can this machine build and run at all?"
if ! bash scripts/preflight.sh; then
  printf '\n\033[31mStopping: preflight found a blocker.\033[0m Nothing below would produce a\n'
  printf 'trustworthy number on this machine.\n\n'
  exit 1
fi

# Pick the interpreter preflight validated, rather than trusting python3 (see preflight.sh
# for why that name is not safe on Windows).
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in python3 python py; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    "$candidate" -c "import sys" >/dev/null 2>&1 && { PYTHON="$candidate"; break; }
  done
fi

# ---- 1. Build -----------------------------------------------------------------------------
rule "Build ($BUILD_TYPE)"
if ! bash scripts/configure.sh "$BUILD_DIR" "$BUILD_TYPE" >/dev/null; then
  printf '\033[31mconfigure failed\033[0m\n'; exit 1
fi
if ! cmake --build "$BUILD_DIR" -j; then
  printf '\033[31mbuild failed\033[0m\n'; exit 1
fi

BIN="$BUILD_DIR/sankhya.exe"
[ -x "$BIN" ] || BIN="$BUILD_DIR/sankhya"
if ! "$BIN" version >/dev/null 2>&1; then
  # Re-checked HERE and not only in preflight, because the binary that matters is the one
  # this run just linked - and on Windows that is exactly the one Smart App Control blocks.
  printf '\n\033[31mThe binary built but will not execute.\033[0m\n\n'
  printf 'On Windows 11 this is Smart App Control: it blocks unsigned executables that have\n'
  printf 'no reputation, and one you just linked has none.\n\n'
  printf 'There is no reliable fix from inside this repository, and it would be dishonest to\n'
  printf 'print one. Rebuilding into a new directory changes the bytes and therefore the\n'
  printf 'hash, and that SOMETIMES clears it - but measured on this project, two Debug\n'
  printf 'builds of the same commit minutes apart gave opposite results, one running and one\n'
  printf 'blocked. Treat a retry as worth one attempt, not as a procedure:\n\n'
  printf '    scripts/reproduce.sh --build-dir build-retry\n\n'
  printf 'If that fails too, the realistic options are a machine without Smart App Control\n'
  printf '(CI builds and runs this on Linux every push), or a decision about Smart App\n'
  printf 'Control that is yours to make and not one a script should make for you. Turning it\n'
  printf 'off on Windows 11 is one-way and cannot be undone without reinstalling.\n\n'
  printf 'THE PRACTICAL LESSON, which costs nothing to follow: the trust is per-binary. A\n'
  printf 'build/ that has been working stops working the moment you rebuild it. Do not\n'
  printf 'rebuild before a demonstration - keep the binary that already runs.\n\n'
  exit 1
fi
"$BIN" version

# ---- 2. Tests -----------------------------------------------------------------------------
rule "Test suite, including the rational-arithmetic oracle"
if ! ctest --test-dir "$BUILD_DIR" --output-on-failure; then
  printf '\n\033[31mTests failed. Every number below would be suspect, so stopping.\033[0m\n\n'
  exit 1
fi

# ---- 3. Netlib ----------------------------------------------------------------------------
rule "Netlib: our answers against the optima published by netlib.org"
if [ "$FETCH_MEDIUM" = 1 ]; then
  printf 'Fetching the 50-instance medium tier (needs network)...\n'
  "$PYTHON" bench/runners/fetch_data.py --set medium || \
    skip "medium-tier fetch" "network unavailable; the committed set below still ran"
fi
printf 'Solving %s committed instance(s), each answer checked by tools/verify_solution.py\n' \
  "$(ls data/netlib/*.mps 2>/dev/null | wc -l | tr -d ' ')"
printf 'which shares no code with the solver.\n\n'
# A DEBUG RUN MUST NOT WRITE INTO bench/results/. The runner records wall time per instance,
# make_benchmarks_doc.py regenerates docs/BENCHMARKS.md from the newest matching CSV, and it
# PREFERS the netlib-small-*.csv glob - so one --debug reproduction would silently republish
# the documented timings as those of an unoptimised build. Correctness is identical either
# way; the numbers that would become documentation are not.
NETLIB_OUT=()
if [ "$BUILD_TYPE" != "Release" ]; then
  NETLIB_OUT=(--out "${TMPDIR:-/tmp}/netlib-$BUILD_TYPE-scratch.csv")
  printf '(%s build: results go to a scratch file, not bench/results/)\n\n' "$BUILD_TYPE"
fi
run_benchmark "Netlib benchmark" \
  "$PYTHON" bench/runners/netlib.py --binary "$BIN" --time-limit 60 \
  ${NETLIB_OUT[@]+"${NETLIB_OUT[@]}"}

# ---- 3b. Robustness: where the solver stops working (#71) ----------------------------------
rule "Robustness sweep: each numerical hazard pushed until the answer or its certificate moves"
printf 'Every instance has an optimum known by construction; a pass needs the status, the\n'
printf 'objective and the independent verifier. docs/BENCHMARKS.md section 5 is this table.\n\n'
ROBUST_OUT=()
if [ "$BUILD_TYPE" != "Release" ]; then
  ROBUST_OUT=(--out "${TMPDIR:-/tmp}/robustness-$BUILD_TYPE-scratch.csv")
fi
run_benchmark "robustness sweep" \
  "$PYTHON" bench/runners/robustness.py --binary "$BIN" \
  ${ROBUST_OUT[@]+"${ROBUST_OUT[@]}"}

# ---- 3c. MIPLIB, when asked ----------------------------------------------------------------
if [ "$RUN_MIPLIB" = 1 ]; then
  rule "MIPLIB 2017: the mixed-integer side, which is the weakest evidence here"
  printf 'Two questions, kept apart: how many instances reach the published optimum, and how\n'
  printf 'many the search PROVES. docs/BENCHMARKS.md section 2 is this table.\n\n'
  MIPLIB_OUT=()
  if [ "$BUILD_TYPE" != "Release" ]; then
    MIPLIB_OUT=(--out "${TMPDIR:-/tmp}/miplib-$BUILD_TYPE-scratch.csv")
  fi
  run_benchmark "MIPLIB benchmark" \
    "$PYTHON" bench/runners/miplib.py --binary "$BIN" --time-limit 60 \
    ${MIPLIB_OUT[@]+"${MIPLIB_OUT[@]}"}
fi

# ---- 4. Comparison against an established solver -------------------------------------------
rule "Compared against HiGHS, as PS26119 requires"
if [ "$BUILD_TYPE" != "Release" ]; then
  skip "HiGHS comparison" "this is a $BUILD_TYPE build. The objectives would still
  agree, but the timings would be measuring our missing optimiser rather than our
  solver, and a number that unfair to us is still a number someone could quote. Build
  Release for the comparison."
elif "$PYTHON" -c "import highspy" >/dev/null 2>&1; then
  "$PYTHON" bench/runners/compare.py --sankhya-binary "$BIN" --time-limit 60 || \
    skip "HiGHS comparison" "see the output above"
else
  skip "HiGHS comparison" "highspy not installed.  pip install highspy   - it runs as a
  separate process and is never linked into SANKHYA, so it does not touch the sovereignty
  claim that section 1 of the demo prints live."
fi

# ---- 5. The walkthrough --------------------------------------------------------------------
rule "The PS26119 walkthrough"
printf 'Everything above, in the problem statement order, plus the case studies, the\n'
printf 'robustness hazards, and the list of what we do NOT have.\n'
# SANKHYA_BIN, not just PYTHON. The demo searches the usual build locations on its own, so
# without this a run with --build-dir would test the binary this script just built and then
# demonstrate a DIFFERENT one - or, as happened here, find a stale build/ that Smart App
# Control had blocked and skip the whole walkthrough for a reason unrelated to the run.
SANKHYA_BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")" PYTHON="$PYTHON" \
  bash demo/run_sih_demo.sh || ran_with_failures "demo" "the walkthrough runs the same
  benchmark runners, which exit non-zero when any instance fails; its own sections above say
  which, and section 6 lists what the project does not have"

# ---- 6. The document the numbers become -----------------------------------------------------
# THE POINT OF THE WHOLE SCRIPT. Everything above wrote a CSV; docs/BENCHMARKS.md is generated
# from those CSVs so it cannot drift from them, and a reproduction that stops before this step
# leaves the reader comparing screen output against a document nobody regenerated. A Debug run
# is excluded because its CSVs went to a scratch file on purpose (see step 3) and the
# generator would republish whatever it found instead.
rule "Regenerating docs/BENCHMARKS.md from the CSVs this run just wrote"
if [ "$BUILD_TYPE" != "Release" ]; then
  skip "docs/BENCHMARKS.md" "this is a $BUILD_TYPE build, whose results deliberately did not
  go into bench/results/. Regenerating would republish older numbers as though this run had
  produced them."
elif "$PYTHON" bench/runners/make_benchmarks_doc.py; then
  if git diff --quiet -- docs/BENCHMARKS.md 2>/dev/null; then
    printf '\ndocs/BENCHMARKS.md is unchanged: the committed document already states what this\n'
    printf 'machine just measured.\n'
  else
    printf '\n\033[33mdocs/BENCHMARKS.md CHANGED.\033[0m That is the honest outcome of a run on a\n'
    printf 'different machine - the pass rates should match, the timings will not. Inspect with\n'
    printf '  git diff -- docs/BENCHMARKS.md\n'
  fi
else
  skip "docs/BENCHMARKS.md" "see the output above"
fi

# ---- Summary --------------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
printf '\n%s\n' "$(printf '%.0s=' $(seq 1 78))"
printf '\033[1mReproduction complete in %dm %ds\033[0m\n' $((ELAPSED / 60)) $((ELAPSED % 60))
if [ ${#INCOMPLETE[@]} -gt 0 ]; then
  printf '\n%d step(s) ran and reported failures:\n' "${#INCOMPLETE[@]}"
  for s in "${INCOMPLETE[@]}"; do printf '  - %s\n' "$s"; done
  printf 'Those are RESULTS, not gaps: these runners exit non-zero when any instance fails,\n'
  printf 'and every failing instance is named in the output above. A tier with known\n'
  printf 'failures - the medium tier is 48 of 50 - exits non-zero on a perfect run.\n'
fi
if [ ${#SKIPPED[@]} -eq 0 ]; then
  printf '\nNothing was skipped. Every claim above was regenerated on this machine.\n'
else
  printf '\n\033[33m%d step(s) skipped:\033[0m\n' "${#SKIPPED[@]}"
  for s in "${SKIPPED[@]}"; do printf '  - %s\n' "$s"; done
  printf '\nThe results above are real; they are simply not the complete set. Anything the\n'
  printf 'README claims that is not printed above was NOT reproduced in this run.\n'
fi
printf '\nWhat we do not have is in demo/run_sih_demo.sh section 6 and in issue #54, which\n'
printf 'tracks every PS26119 requirement against what exists on main.\n\n'

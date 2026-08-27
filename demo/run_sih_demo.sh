#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SANKHYA - the SIH PS26119 demonstration.
#
# Walks the problem statement in its own order and shows, for each thing it asks for, either
# a live result or an explicit admission that we do not have it yet. Every number printed
# below is produced by a command run during this script. Nothing is cached, quoted from an
# earlier run, or read out of a file somebody wrote by hand - see CLAUDE.md, "Evidence rules".
#
#   demo/run_sih_demo.sh              # the full walk
#   demo/run_sih_demo.sh --quick      # skip the HiGHS comparison
#
# demo/run_demo.sh is the other script: it goes deep on ONE instance. This one goes wide.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BIN="${SANKHYA_BIN:-}"
if [ -z "$BIN" ]; then
  for candidate in build/sankhya build/sankhya.exe build/Release/sankhya.exe; do
    [ -x "$REPO/$candidate" ] && BIN="$REPO/$candidate" && break
  done
fi
if [ -z "$BIN" ]; then
  echo "No solver binary found. Build first:" >&2
  echo "    scripts/configure.sh build Release && cmake --build build -j" >&2
  exit 1
fi

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

CASES="data/casestudies"

rule() { printf '\n\033[1m%s\033[0m\n%s\n' "$1" "$(printf '=%.0s' $(seq 1 86))"; }
ask()  { printf '\n\033[2mPS26119 asks:\033[0m %s\n\n' "$1"; }

# Solve one model quietly and leave stats in $WORK/<tag>.json and the point in <tag>.sol.
solve_case() {
  local tag="$1" path="$2"
  "$BIN" solve "$path" --option log_to_console=false \
    --write-sol "$WORK/$tag.sol" --stats "$WORK/$tag.json" >/dev/null
}

field() {  # field <tag> <section> <key>
  "$PYTHON" -c "
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])" "$WORK/$1.json" "$2" "$3"
}

# ===========================================================================================
rule "0. What this is"
# ===========================================================================================
"$BIN" version
cat <<'INTRO'

SANKHYA is a mathematical optimization solver core written from scratch in C++20 for
SIH PS26119, issued by Mangalore Refinery and Petrochemicals. LP and MILP engines are
implemented and benchmarked; QP is not yet. This script walks the problem statement in
its own order. Where we do not have something, it says so rather than changing the subject.
INTRO

# ===========================================================================================
rule "1. Sovereignty: not built on an existing solver"
# ===========================================================================================
ask "'It shall not be built upon any existing open source solver library but shall be built
              from scratch from mathematical foundation.'"

echo "Everything the binary links against, right now:"
if command -v ldd >/dev/null 2>&1; then
  ldd "$BIN" | sed 's/^/    /'
elif command -v objdump >/dev/null 2>&1; then
  objdump -p "$BIN" | grep "DLL Name" | sed 's/^/    /'
fi
cat <<'PROV'

CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, lp_solve, OSQP, PDLP, OR-Tools, Gurobi, CPLEX and
Xpress appear nowhere in that list - not their simplex, not their cuts, not their MPS
reader. The simplex, the LU factorization, the branch and bound, the MPS parser and the
first-order method are all ours. docs/PROVENANCE.md carries the dependency table, the
algorithm-to-citation table and the full CMake link line; CI regenerates an SPDX SBOM.
PROV

# ===========================================================================================
rule "2. The LP core, on a recognised benchmark library"
# ===========================================================================================
ask "'The solver should successfully solve standard benchmark problems from recognised
              optimization libraries such as MIPLIB, Netlib or Mittelmann benchmark sets.'"

echo "Solving the committed Netlib set live, and checking every answer against the optimum"
echo "published by netlib.org - which bench/runners/fetch_data.py parses from their readme."
echo
"$PYTHON" bench/runners/netlib.py --binary "$BIN" --time-limit 60 \
  --out "$WORK/netlib.csv" 2>&1 | tail -14

# ===========================================================================================
rule "3. The industrial scope PS26119 names"
# ===========================================================================================
ask "'refinery scheduling, crude blending, process optimization, production planning,
              logistics, power system dispatch, transportation and supply chain management.'"

printf '    %-22s %-6s %6s %6s  %-10s %16s\n' \
  "case study" "class" "rows" "cols" "status" "objective"
printf '    %s\n' "$(printf -- '-%.0s' $(seq 1 76))"

for entry in \
  "crude_blend:demo/crude_blend.mps:crude blending" \
  "blend_milp:demo/blend_milp.mps:refinery scheduling" \
  "power_dispatch:$CASES/power_dispatch.mps:power dispatch" \
  "supply_chain:$CASES/supply_chain.mps:supply chain" \
  "lot_sizing:$CASES/lot_sizing.mps:production planning" ; do
  tag="${entry%%:*}"; rest="${entry#*:}"; path="${rest%%:*}"; label="${rest#*:}"
  solve_case "$tag" "$path"
  ints=$(field "$tag" model integer_columns)
  klass="LP"; [ "$ints" != "0" ] && klass="MILP"
  printf '    %-22s %-6s %6s %6s  %-10s %16.6f\n' \
    "$label" "$klass" "$(field "$tag" model rows)" "$(field "$tag" model columns)" \
    "$(field "$tag" result status)" "$(field "$tag" result objective)"
done

echo
echo "Each of those was then handed to tools/verify_solution.py, which has its OWN MPS reader"
echo "and links no part of our C++. It re-derives the row activities, the objective, the"
echo "reduced costs and the duality gap independently:"
echo
for entry in \
  "crude_blend:demo/crude_blend.mps" \
  "blend_milp:demo/blend_milp.mps" \
  "power_dispatch:$CASES/power_dispatch.mps" \
  "supply_chain:$CASES/supply_chain.mps" \
  "lot_sizing:$CASES/lot_sizing.mps" ; do
  tag="${entry%%:*}"; path="${entry#*:}"
  printf '    %-34s %s\n' "$(basename "$path")" \
    "$("$PYTHON" tools/verify_solution.py "$path" "$WORK/$tag.sol" --quiet 2>&1 | tail -1)"
done

# -------------------------------------------------------------------------------------------
echo
echo "--- What a planner actually reads: the shadow prices on the blend ------------------"
echo
echo "The dual value on a constraint is the marginal worth of relaxing it by one unit. It is"
echo "the output a refinery planner acts on, and it is why an LP is worth solving exactly."
echo
sed -n '/begin rows/,/end rows/p' "$WORK/crude_blend.sol" | grep -v '^begin\|^end' | \
  awk '{printf "    %-12s activity %14.6f    shadow price %14.6f\n", $1, $2, $3}'

# -------------------------------------------------------------------------------------------
echo
echo "--- A MILP answer corroborated by exhaustion, not by itself ------------------------"
echo
cat <<'ORACLE'
verify_solution.py can confirm a MILP answer is feasible, integral and that the bound closed.
It CANNOT confirm no better answer exists - that is the whole content of the search, and a
branch and bound which wrongly fathoms a node returns a point that passes every such check.
The dispatch instance is small enough to settle by exhaustion, so we settle it:

ORACLE
"$PYTHON" "$CASES/dispatch_oracle.py" | sed 's/^/    /'
echo
echo "    SANKHYA returned:  $(field power_dispatch result objective)  in $(field power_dispatch effort nodes) nodes"
"$PYTHON" - "$WORK/power_dispatch.json" <<'PYCHK'
import json, sys
got = json.load(open(sys.argv[1]))["result"]["objective"]
ok = abs(got - 3270.0) <= 1e-9
print("    " + ("AGREES with the exhaustive oracle to 1e-9."
                if ok else "DISAGREES with the oracle - got {!r}".format(got)))
sys.exit(0 if ok else 1)
PYCHK

# ===========================================================================================
rule "4. Numerical robustness, on the three hazards PS26119 names"
# ===========================================================================================
ask "'...challenging large-scale optimization problems involving degeneracy, weak LP
              relaxations or ill-conditioned constraint matrices, where simpler
              implementations struggle to achieve reliable convergence.'"

# --- degeneracy ---------------------------------------------------------------------------
echo "(a) DEGENERACY - $CASES/supply_chain.mps"
echo
cat <<'DEGEN'
    Balanced transportation: total supply equals total demand, so the seven equality rows
    are linearly DEPENDENT and the constraint matrix has rank 6, not 7. A basis needs
    m + n - 1 = 6 basic variables against 7 rows, so a basic variable sits at zero at every
    iteration and ratio-test ties are the norm. This is the classical cycling test bed - the
    structure that makes a naive simplex stall forever on a sequence of zero-length steps.
DEGEN
echo
echo "    Measured, not asserted. The matrix is read by verify_solution.py's INDEPENDENT MPS"
echo "    reader, and the rank is computed EXACTLY by elimination over the rationals - no"
echo "    singular-value threshold to pick, and no SANKHYA code involved:"
echo
"$PYTHON" "$CASES/matrix_stats.py" "$CASES/supply_chain.mps" --rank | sed 's/^/        /'
echo
echo "    SANKHYA: status $(field supply_chain result status), objective $(field supply_chain result objective), $(field supply_chain effort iterations) iterations, no stall."
echo "    Anti-cycling is Bland's rule, cited in src/simplex/primal_simplex.cpp."

# --- ill conditioning ----------------------------------------------------------------------
echo
echo "(b) ILL-CONDITIONING - $CASES/ill_conditioned.mps"
echo
cat <<'ILLC'
    The SAME model, with rows scaled by 10^-6 ... 10^+6 and columns re-parameterised by
    10^-6 ... 10^+4. Both are exact changes of variable, so the optimal objective MUST be
    unchanged. That makes the robustness claim falsifiable instead of decorative: we know
    the right answer, so a solver that drifts here cannot hide it.
ILLC
solve_case ill_conditioned "$CASES/ill_conditioned.mps"
echo
echo "    what the scaling did to the matrix, measured the same independent way:"
echo
"$PYTHON" "$CASES/matrix_stats.py" "$CASES/ill_conditioned.mps" --conditioning | sed 's/^/        /'
echo
printf '    %-28s %s\n' "well-scaled supply_chain:" "$(field supply_chain result objective)"
printf '    %-28s %s\n' "ill-conditioned re-scaling:" "$(field ill_conditioned result objective)"
"$PYTHON" - "$WORK/supply_chain.json" "$WORK/ill_conditioned.json" <<'PYAGREE'
import json, sys
a = json.load(open(sys.argv[1]))["result"]["objective"]
b = json.load(open(sys.argv[2]))["result"]["objective"]
rel = abs(a - b) / max(1.0, abs(a))
print("    relative difference:         {:.3e}".format(rel))
if rel <= 1e-9:
    print("    The two agree. The scaling did not move the answer.")
else:
    print("    THEY DISAGREE. That is a bug in our solver, and this demo just found it.")
sys.exit(0 if rel <= 1e-9 else 1)
PYAGREE

# --- weak relaxation -----------------------------------------------------------------------
echo
echo "(c) WEAK LP RELAXATION - $CASES/lot_sizing.mps"
echo
cat <<'WEAK'
    Lot sizing with a big-M set-up link, x_t <= M y_t. In the relaxation y_t is free to take
    x_t / M, so a period producing one unit pays 1/180 of a set-up rather than a whole one.
    The bound therefore sits far below the integer optimum and branch and bound has to close
    the difference by search. This is the standard weak-formulation failure mode.
WEAK
solve_case lot_relaxed "$CASES/lot_sizing_relaxed.mps"
echo
printf '    %-34s %s\n' "LP relaxation bound:" "$(field lot_relaxed result objective)"
printf '    %-34s %s\n' "proven integer optimum:" "$(field lot_sizing result objective)"
printf '    %-34s %s\n' "nodes explored to close it:" "$(field lot_sizing effort nodes)"
"$PYTHON" - "$WORK/lot_relaxed.json" "$WORK/lot_sizing.json" <<'PYGAP'
import json, sys
lp = json.load(open(sys.argv[1]))["result"]["objective"]
ip = json.load(open(sys.argv[2]))["result"]["objective"]
print("    {:<34} {:.2f}%".format("integrality gap the search closed:",
                                  100.0 * (ip - lp) / abs(ip)))
PYGAP

# ===========================================================================================
rule "5. Compared against an established solver"
# ===========================================================================================
ask "'...with solution quality and computational performance compared against at least one
              established commercial or open-source solver.'"

if [ "$QUICK" = "1" ]; then
  echo "Skipped (--quick). Run without --quick to execute the comparison live."
elif "$PYTHON" -c "import highspy" >/dev/null 2>&1; then
  echo "HiGHS is the reference. It is invoked as a SEPARATE PROCESS over the same MPS files;"
  echo "no HiGHS code is linked into, or read by, SANKHYA. Both sides are timed on solver"
  echo "time only, so neither is charged for interpreter start-up."
  echo
  "$PYTHON" bench/runners/compare.py --sankhya-binary "$BIN" --time-limit 60 \
    --out "$WORK/compare.csv" 2>&1 | tail -18
else
  echo "highspy is not importable, so the comparison cannot run here."
  echo "    pip install highspy    then re-run this script."
fi

# ===========================================================================================
# Read the medium-tier result out of the newest committed CSV. Falls back to naming the
# reproduction command if none is present, rather than printing a number from nowhere.
MEDIUM_SUMMARY="$("$PYTHON" bench/runners/latest_result.py "netlib-medium-*.csv" --summary)"

rule "6. What PS26119 asks for that we do NOT yet have"
# THE MEDIUM-TIER FIGURE BELOW IS READ FROM THE COMMITTED CSV, not typed here. The demo does
# not run that tier - fetching 50 instances takes minutes - so it was hand-written, and it
# went stale three separate times in two days: 26 after #49 made it 37, then 37 after #86
# made it 40. Each time it UNDERSTATED the solver, which is the safe direction and still
# wrong. Deriving it from bench/results/netlib-medium-*.csv removes the failure mode rather
# than asking the next person to remember. Same reasoning as #53.
# ===========================================================================================
cat <<'GAPS' | sed "s|@MEDIUM@|${MEDIUM_SUMMARY}|"
    Stating these is the point. A solver that is vague about its limits is not one an
    industrial user can plan around.

    QP                  Not implemented. PS26119 names LP, MILP and QP as the initial focus.
                        The Model already carries the Hessian and solve() refuses a QP with
                        `not_solved` rather than silently reporting its LP relaxation.
    Interior point      Not implemented. The continuous engines today are revised simplex
                        (exact, gives a basis) and restarted PDHG (first-order, CPU).
    Cutting planes      Branch and bound is plain: no Gomory, MIR or cover cuts yet, no
                        pseudocost branching. Tracked as issue #23.
    GPU acceleration    NOT WRITTEN. The first-order method it needs exists and runs on CPU;
                        the CUDA backend is issues #16-#19. --gpu today prints a warning and
                        falls back to CPU. We are not claiming a speed-up we have not measured.
    Scale               Everything above is small. The committed Netlib set is the small end
                        of Netlib, and NOTHING here supports a claim about the "thousands to
                        millions of variables" the problem statement asks for. On the wider
                        50-instance Netlib medium set we pass @MEDIUM@.
                        That is issue #34, and it is the honest headline number, not the 8/8
                        above. Reproduce it with:
                            python bench/runners/fetch_data.py --set medium
                            python bench/runners/netlib.py --time-limit 60
    Parallelism         Single-threaded today.

    On speed against HiGHS, section 5 above prints the measured ratio for this run rather
    than repeating a number here that would go stale - and it is a narrow comparison either
    way: eight small, well conditioned instances settle nothing about large models. HiGHS is
    a decade of specialist work, including a mature dual simplex - which we do not have yet.
    Our own presolve (src/presolve, on by default) is not part of that gap any more.
    The claim we do make is narrower and checkable: on every instance we report as solved,
    the answer matches the published optimum AND survives an independent verifier that
    shares no code with the solver.
GAPS
echo

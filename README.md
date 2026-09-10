**SIH26119** — Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to CPLEX / Xpress)  
Smart India Hackathon 2026 · Mangalore Refinery and Petrochemicals Limited (MRPL)

# SANKHYA

**Indigenous GPU-accelerated optimization solver — LP, MILP, convex QP, written from
mathematical foundations.**

Smart India Hackathon 2026, problem statement **SIH26119**, issued by Mangalore Refinery and
Petrochemicals Limited.

> Not built on top of any open-source solver. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md)
> for the dependency table, the full link line, the linked-library dump, and the CI job that
> fails the build if a solver library ever appears in the binary.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Foundations: model, options, sparse linear algebra, CI | **done** |
| 2 | MPS/LP readers, revised primal simplex, CLI | **done** |
| 3 | Verification spine: rational oracle, independent checker, Netlib harness | **done** |
| 4 | Restarted PDHG — **CPU done**, CUDA backend not started (no GPU available) | partial |
| 5 | Branch & bound → MILP | **done** (MIPLIB benchmarked; root cuts landed in #159 and are off by default, see below) |
| 6–10 | Performance, branch & cut, IPM/QP, robustness, packaging | convex QP **done** (Phase 8, `src/qp/`); interior point **done, opt-in** (`src/ipm/`, no basis); robustness sweep **done** (`bench/runners/robustness.py`); root cuts **done, off by default** (#159, `src/mip/cuts.cpp`); packaging **done** apart from the human items on #73 |

LP is solved by a bounded-variable revised primal simplex (or restarted PDHG), MILP by
branch and bound, and convex QP by a Condat-Vu primal-dual method — and MIQP by branch and
bound over QP relaxations — all end to end from an MPS file through to an independently
verified answer. **Non-convex** QP is the one class still refused, and refused deliberately
rather than approximated: it is decided by an LDL^T semidefiniteness test before any
arithmetic starts, and a negative pivot is returned as the certificate. Reporting a local
optimum as a global one is the single most damaging thing this dispatcher could do, so it
does not — see the Evidence rules in [`CLAUDE.md`](CLAUDE.md).

Benchmark results against Netlib, headline first: **78 of 89** on the full set — matched
to the published optimum to a relative 1e-6 *and* passed independent verification —
measured on `main` at `53cbe16` (`bench/results/netlib-full-53cbe16.csv`). The narrower
tiers read higher (**48 of 50** on the medium tier, **9 of 9** on the small set the demo
runs) because both are defined by a row cap, which makes them the easier half by
construction; the full set is the number Phase 6's ">= 95% of Netlib" criterion is
measured against, so it is the one quoted here. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), generated from the CSVs in `bench/results/` so
it cannot drift.

The 11 non-passes are worth naming, and most of them are not wrong answers. Every one that
produces an answer was cross-checked against **HiGHS**, a mature third-party solver run as a
separate process, by `bench/runners/cross_check_highs.py`
(`bench/results/cross-check-highs-adcee1b.csv`):

| what it is | count | instances |
|---|---|---|
| our answer verifies as optimal and agrees with HiGHS; Netlib's published table is the outlier (`e226` by its objective constant, the rest by up to 1.3e-03) | **7** | `80bau3b`, `e226`, `ganges`, `greenbea`, `greenbeb`, `nesm`, `scrs8` |
| ran out of time at 120 s | **2** | `dfl001`, `pilot87` |
| the answer agrees with HiGHS to 3.0e-07 and verifies; our own dual-feasibility check downgrades the status to `feasible`, and Netlib's table is off by 1.5e-04 | **1** | `pilot` |
| the solver declined to answer: its phase-1 ratio test found no blocking variable and it reported a numerical error rather than a claim it could not stand behind | **1** | `maros-r7` |

So: **on every Netlib instance where this solver produces a final answer, that answer
agrees with HiGHS** - nine of the twelve rows in that CSV agree to 3.0e-07 or better. The
three that differ are `dfl001` and `pilot87`, whose rows there are unfinished iterates
rather than answers, and `maros-r7`, which has none. What remains is speed on two
instances, our own status reporting on one, and one instance without an answer.

The machine's speed state is part of the evidence, so it is stated, and on this laptop it
decides exactly one instance. `pilot87` needs **26,226 iterations** to reach its optimum -
run it under `--option iteration_limit=27000` and it gets there every time, at objective
301.710691459, with one basis repair - and whether those iterations fit inside the 120 s
limit depends on how fast the machine is that minute. It fitted at `adcee1b` (55.9 s) and
did not at `53cbe16` (the run above), which is the whole of the difference between 79 of
89 and 78 of 89 on this hardware. Nothing else moves: 86 of the 89 rows are identical in
status, iteration count and objective between the two runs, and the other two are
`dfl001`, truncated wherever the clock leaves it, and `fit2p`, which takes the scaled or
the unscaled route depending on the same clock (#172). The table above is the slower, more
conservative run.

This mattered because our own verifier could not settle it — it re-derives the answer from
the same file we read, so agreeing with it shows only that our two readers agree, and both
were written by this project (#75). Two independent solvers landing on the same number is a
different order of evidence. The pass rate above is still measured against Netlib's table,
unchanged: a project cannot grade itself against a solver of its own choosing.

The failure class that *was* the largest is gone from Netlib. `basis became singular` was
13 of 23 failures at `ca6fde9` (`bench/results/netlib-full-ca6fde9.csv`, 2026-08-31) and
has been **zero on the full Netlib set** since #144 found the pivot search treating "none
of my first four candidates was admissible" as proof of singularity and #147 repaired the
genuine rank defects that remained. It is not zero everywhere: on Mittelmann's `qap15` the
unscaled retry went singular at iteration 13,954 (`bench/results/mittelmann-592aea3.csv`,
issue #174), the first reappearance in the evidence and on the Mittelmann instance closest
to Netlib's size. Since then the dual simplex became the automatic engine (#165),
reliability branching landed (#166), presolve's postsolve runs its dual passes to a fixed
point (#162), and the FTRAN went hyper-sparse (#169): between them the full set went from
71 to 78 verified passes, and `degen3`, which took 123.6 s, takes 0.9 s
(`bench/results/netlib-full-53cbe16.csv`).

That is a better class of problem to have, and a different roadmap: speed on the three
largest instances rather than robustness. Tracked in #198.

MIPLIB 2017 is benchmarked too: **13 of 30** easy instances reach the published optimum,
**6 of 30** also prove it (`bench/results/miplib-53cbe16.csv`, 60 s) —
branch and bound has reliability branching and warm-started node LPs, and root cutting
planes that are off by default because they were measured to cost proofs at this limit
(#159; `docs/BENCHMARKS.md` section 2), so it finds good incumbents far more often than it
closes the bound. For
scale beyond what Netlib tests, `bench/runners/generate_large_lp.py` builds sparse LPs of
any size with an exactly known analytic optimum, and `bench/runners/mittelmann.py` runs
Mittelmann's LP set: on its eight smallest instances (6,330 to 376,500 rows) the result is
**0 of 8** inside 300 s - seven time limits and one singular basis (`qap15`, #174) - every
one named in section 1d of `docs/BENCHMARKS.md` (`bench/results/mittelmann-592aea3.csv`,
run on `main` with the machine awake and on AC throughout).

## Reproduce everything

One command takes a fresh clone to every claim on this page - build, tests, the Netlib
benchmark with independent verification, the HiGHS comparison, and the full PS26119
walkthrough:

```bash
scripts/reproduce.sh
```

It runs **offline**: the Netlib instances it benchmarks are committed, with their published
optima. Add `--fetch-medium` to also download and run the 50-instance medium tier; the
full 89-instance set, where the headline number above comes from, is
`bench/runners/fetch_data.py --set full` followed by
`bench/runners/netlib.py --time-limit 120`. Any step that cannot run on your machine prints why and is
listed again in the summary, so a shorter run is never mistaken for a passing one.

To check the machine without running anything:

```bash
scripts/preflight.sh
```

It names the two things that most often go wrong quietly - a `python3` that is the Microsoft
Store stub, and Windows Smart App Control refusing to execute a freshly linked binary - and
prints the fix for each.

## Build

Requires CMake 3.20+, Ninja, and a C++20 compiler (GCC 10+ / Clang 12+ / MSVC 19.30+).

```bash
scripts/configure.sh build Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`scripts/configure.sh` picks a C++20-capable compiler rather than trusting PATH order,
which matters on Windows boxes carrying an old MinGW. It also reuses dependency sources
from any build tree already on disk, so a second build directory costs seconds rather than
re-cloning 200 MB. 285 tests, all passing.

## Use

```bash
./build/sankhya version
./build/sankhya options
./build/sankhya info  demo/crude_blend.mps
./build/sankhya solve demo/crude_blend.mps --write-sol blend.sol --stats blend.json
./build/sankhya solve demo/crude_blend.mps --progress-out progress.jsonl
```

`demo/crude_blend.mps` is a small crude-blending LP: three crudes into a diesel pool, with a
CDU throughput window, a diesel commitment and a sulphur specification. It ships as both MPS
and LP so the two readers can be checked against each other, and it exercises the format
features most likely to be misread - a `RANGES` entry on a `G` row, an equality row,
`OBJSENSE MAX`, and `LO`/`UP` bounds.

`solve` returns a meaningful exit code: `0` optimal, `1` a limit or a proven
infeasible/unbounded model, `3` the file could not be read, `5` a numerical or model error.

Beyond the objective, the solution file carries the **shadow price of every row**. On the
blending model those are the numbers a refinery planner acts on: what one more unit of
diesel commitment costs, and what the sulphur specification is worth.

`--progress-out` appends one JSON line per logged iteration or node to a file as the solve
runs, flushed immediately - an operator can `tail -f` it during a long solve to watch the
bound close in on the answer without waiting for the final report.

## Demo

```bash
demo/run_sih_demo.sh         # the full PS26119 walkthrough, in the problem statement's order
demo/run_demo.sh --list      # or pick a single instance
demo/run_demo.sh share2b     # solve it live, then verify it independently
```

The nine Netlib instances are committed, so the demo needs no network. Every number it prints
comes from a command it just ran.

On Windows, run it as `PYTHON=python demo/run_sih_demo.sh` if `python3` on your PATH is the
Microsoft Store stub; `scripts/preflight.sh` tells you whether it is.

## Layout

```
include/sankhya/  public headers — Model, Solution, Options, tolerances, sparse containers
                  plus sankhya.h, the C API
src/api           C API — an FFI-safe surface over the core, no C++ types crossing
src/core          Model/Solution implementation and the solve() dispatcher
src/util          logging, timers, arena allocator, option registry
src/io            MPS + LP readers (including QPS QUADOBJ), solution and JSON writers
src/presolve      reductions + postsolve               (on by default)
src/simplex       primal and dual revised simplex (the dual is the branch-and-bound node engine)
src/la            sparse containers, sparse Markowitz LU (hyper-sparse FTRAN), sparse LDL^T, dense LU (test oracle only)
src/pdhg          restarted PDHG, CPU                  (CUDA backend: not started)
src/mip           branch and bound + diving heuristic + root cuts (cuts off by default, #159)
src/qp            convex QP, Condat-Vu primal-dual     (done)
src/ipm           Mehrotra interior point, sparse LDL^T (opt-in: algorithm=ipm, no basis)
bindings/python   Python bindings — ctypes over the C API, nothing to compile
tests/  bench/  tools/  docs/  demo/
```

## What this does NOT do

Stated here rather than only in the demo, because a solver that is vague about its limits is
not one an industrial user can plan around. [Issue #54](https://github.com/thegoodengineer/sih-26/issues/54)
tracks every PS26119 requirement against what exists on `main`; section 6 of
`demo/run_sih_demo.sh` prints this list at the end of every run.

| not implemented | note |
|---|---|
| **GPU acceleration** | The first-order method it needs exists and runs on CPU - restarted PDHG, `--option algorithm=pdhg`, 8 of 9 committed instances to `optimal` at 1e-8 (`docs/BENCHMARKS.md` section 1e). The CUDA backend is unwritten (#16-#19); `--gpu` warns and falls back. No speed-up is claimed. |
| **Scale** | Measured, and the answer depends on the engine (#198, `docs/BENCHMARKS.md` section 1f). On generated instances whose optimum is exact by construction, the first-order engine reaches it at **100,000 x 100,000** to a relative 1.0e-07 in 121 s but cannot certify it inside the limit; the dual simplex and the interior point reach it at 1,000 and time out above that. On real models: the largest Netlib instance solved is `fit2d`, 25x10500 with 129018 nonzeros, in 0.3 s, and on Mittelmann's eight smallest LPs, 6,330 to 376,500 rows, the result is 0 of 8 inside 300 s. Nothing here is evidence about a million-variable industrial model. |
| **Interior point as a default** | An interior-point method exists (#56, `--option algorithm=ipm`, Mehrotra predictor-corrector over a from-scratch sparse LDL^T) and is opt-in: it produces no basis, so it cannot warm-start branch and bound and cannot certify infeasibility, and on the full Netlib set it verifies fewer instances than the dual simplex (`docs/PS26119_COVERAGE.md`). The default continuous engine is the simplex. |
| **Cutting planes by default** | Root Gomory mixed-integer and lifted knapsack cover cuts exist (#159, `--option enable_root_cuts=true`) and are off by default: on the 30-instance MIPLIB set at 60 s they take the node count to 0.887x over the 28 instances that end the same way and cost two proofs, because a cut row makes every node LP dearer (`bench/results/miplib-cuts-{off,on}.csv`; `docs/BENCHMARKS.md` section 2). No MIR cuts, and none below the root. Branch and bound itself has reliability branching (#69) and warm-started dual node LPs (#65). This is why MIPLIB proves few optima. |
| **Non-convex QP** | Refused deliberately, with an LDL^T certificate. A local optimum reported as a global one is not something this solver will do. |
| **MIQP bound quality** | MIQP is implemented, but its node bound comes from a first-order method and is only accurate to the tolerance it converged to, so pruning is deliberately kept on the conservative side and costs nodes. With root cuts off by default too, expect incumbents more often than proofs. |
| **Parallelism** | Single-threaded by default. `--option threads=N` runs the column loops of an iteration under OpenMP, deterministically - results are bit-identical at 1 and 8 threads - and at Netlib scale it is measured to buy nothing, because an iteration is too short to amortize the fork (#57). It is a correctness-preserving switch, not a speed claim. |

On speed against HiGHS: on the medium tier the objectives agree on all 50 instances, and
the speed comparison is **indistinguishable rather than a result**. These models solve in
single-digit milliseconds, 20 of the 50 timing envelopes overlap outright, and the median
per-instance ratio came out 1.38x on one committed run and 2.11x on the next, three days
apart on this machine (`bench/results/compare-highs-medium-adcee1b.csv` and
`compare-highs-medium-53cbe16.csv`) - while our total solve time on the second run was
0.68x the first's. A number that unstable is not a speed claim in either direction.
The reproducible comparison is iteration count, where the gap narrowed by a third when
devex pricing became the default (#66); HiGHS's devex still takes fewer.

## Licence

Apache-2.0.

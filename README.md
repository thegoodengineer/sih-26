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
| 5 | Branch & bound → MILP | **done** (cuts still deferred, see #23; MIPLIB now benchmarked) |
| 6–10 | Performance, branch & cut, IPM/QP, robustness, packaging | convex QP **done** (Phase 8, `src/qp/`); IPM, cuts, packaging remain |

LP is solved by a bounded-variable revised primal simplex (or restarted PDHG), MILP by
branch and bound, and convex QP by a Condat-Vu primal-dual method — all end to end from an
MPS file through to an independently verified answer. Only MIQP (mixed-integer QP) is
**refused**, not approximated: handing it to the LP or QP engine and reporting a relaxation
as optimal is the single most damaging thing this dispatcher could do, so it does not — see
the Evidence rules in [`CLAUDE.md`](CLAUDE.md).

Benchmark results against Netlib: **9 of 9** on the small set the demo runs, **41 of 50** on
the medium tier — see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), generated from the CSVs in
`bench/results/` so it cannot drift. MIPLIB 2017 is now benchmarked too: **10 of 30** easy
instances reach the published optimum, **5 of 30** also prove it (branch and bound has no
cutting planes yet, see #23) — same source. For scale beyond what Netlib's committed set
tests (it tops out around 500 rows), `bench/runners/generate_large_lp.py` builds sparse LPs
of any size with an exactly known analytic optimum.

## Reproduce everything

One command takes a fresh clone to every claim on this page - build, tests, the Netlib
benchmark with independent verification, the HiGHS comparison, and the full PS26119
walkthrough:

```bash
scripts/reproduce.sh
```

It runs **offline**: the Netlib instances it benchmarks are committed, with their published
optima. Add `--fetch-medium` to also download and run the 50-instance medium tier, which is
where the honest pass rate lives. Any step that cannot run on your machine prints why and is
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
src/core          Model/Solution implementation and the solve() dispatcher
src/util          logging, timers, arena allocator, option registry
src/io            MPS + LP readers (including QPS QUADOBJ), solution and JSON writers
src/presolve      reductions + postsolve               (on by default)
src/simplex       primal revised simplex               (dual simplex: Phase 6)
src/la            sparse containers, sparse Markowitz LU, dense LU (test oracle only)
src/pdhg          restarted PDHG, CPU                  (CUDA backend: not started)
src/mip           branch and bound + diving heuristic  (cutting planes: Phase 7)
src/qp            convex QP, Condat-Vu primal-dual     (done)
src/ipm           interior point                       (Phase 8, not started)
tests/  bench/  tools/  docs/  demo/
```

## What this does NOT do

Stated here rather than only in the demo, because a solver that is vague about its limits is
not one an industrial user can plan around. [Issue #54](https://github.com/thegoodengineer/sih-26/issues/54)
tracks every PS26119 requirement against what exists on `main`; section 6 of
`demo/run_sih_demo.sh` prints this list at the end of every run.

| not implemented | note |
|---|---|
| **GPU acceleration** | The first-order method it needs exists and runs on CPU. The CUDA backend is unwritten (#16-#19); `--gpu` warns and falls back. No speed-up is claimed. |
| **Scale** | One generated 5000x5000 instance is demonstrated with an optimum known by construction. Nothing here supports the *"millions of variables"* end of the problem statement. |
| **Interior point** | Not started (#56). The continuous engines are revised simplex and restarted PDHG. |
| **Cutting planes** | Branch and bound is plain - no Gomory, MIR or cover cuts, no pseudocost branching (#23). This is why MIPLIB proves few optima. |
| **MIQP** | Convex QP and MILP each work; joining them is not written. `solve()` returns `not_solved` rather than reporting either relaxation. |
| **Non-convex QP** | Refused deliberately, with an LDL^T certificate. A local optimum reported as a global one is not something this solver will do. |
| **Parallelism** | Single-threaded. |

On speed against HiGHS: on the committed instances the two are **indistinguishable**, not
faster. They solve in single-digit milliseconds and the timing envelopes overlap, so
`bench/runners/compare.py` marks the rows it cannot separate and says so. The reproducible
comparison is iteration count, where we are behind - Dantzig pricing against HiGHS's devex
(#66).

## Licence

Apache-2.0.

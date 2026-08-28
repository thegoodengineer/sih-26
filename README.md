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

Benchmark results against Netlib: **8 of 8** on the small set the demo runs, **41 of 50** on
the medium tier — see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), generated from the CSVs in
`bench/results/` so it cannot drift. MIPLIB 2017 is now benchmarked too: **10 of 30** easy
instances reach the published optimum, **5 of 30** also prove it (branch and bound has no
cutting planes yet, see #23) — same source. For scale beyond what Netlib's committed set
tests (it tops out around 500 rows), `bench/runners/generate_large_lp.py` builds sparse LPs
of any size with an exactly known analytic optimum.

## Build

Requires CMake 3.20+, Ninja, and a C++20 compiler (GCC 10+ / Clang 12+ / MSVC 19.30+).

```bash
scripts/configure.sh build Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`scripts/configure.sh` picks a C++20-capable compiler rather than trusting PATH order,
which matters on Windows boxes carrying an old MinGW. 284 tests, all passing.

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
demo/run_demo.sh --list      # the instances a judge can pick from
demo/run_demo.sh share2b     # solve it live, then verify it independently
```

The eight Netlib instances are committed, so the demo needs no network. Every number it
prints comes from a command it just ran.

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

## Licence

Apache-2.0.

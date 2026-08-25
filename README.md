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
| 5 | Branch & bound → MILP | |
| 6–10 | Performance, branch & cut, IPM/QP, robustness, packaging | |

LP is solved by a bounded-variable revised primal simplex. MILP and QP are **refused**, not
approximated: handing a MILP to the LP engine and reporting its fractional relaxation as
optimal is the single most damaging thing this dispatcher could do, so it does not — see the
Evidence rules in [`CLAUDE.md`](CLAUDE.md).

Benchmark results against Netlib are **not** claimed yet. That is Phase 3, and it is gated on
an open provenance question recorded in [`docs/PROVENANCE.md`](docs/PROVENANCE.md) section 5.

## Build

Requires CMake 3.20+, Ninja, and a C++20 compiler (GCC 10+ / Clang 12+ / MSVC 19.30+).

```bash
scripts/configure.sh build Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`scripts/configure.sh` picks a C++20-capable compiler rather than trusting PATH order,
which matters on Windows boxes carrying an old MinGW.

## Use

```bash
./build/sankhya version
./build/sankhya options
./build/sankhya info  demo/crude_blend.mps
./build/sankhya solve demo/crude_blend.mps --write-sol blend.sol --stats blend.json
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

## Layout

```
include/sankhya/  public headers — Model, Solution, Options, tolerances, sparse containers
src/core          Model/Solution implementation and the solve() dispatcher
src/util          logging, timers, arena allocator, option registry
src/la            sparse linear algebra
src/io            MPS + LP readers, solution and JSON writers
src/simplex       primal revised simplex, dense LU     (dual simplex: Phase 6)
src/pdhg src/gpu  first-order method, CUDA backend     (Phase 4)
src/mip           branch and cut                       (Phase 5)
src/ipm src/qp    interior point, convex QP            (Phase 8)
tests/  bench/  tools/  docs/  demo/
```

## Licence

Apache-2.0.

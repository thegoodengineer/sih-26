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
| 2 | MPS/LP readers, revised primal simplex, CLI | next |
| 3 | Verification spine: rational oracle, independent checker, Netlib harness | |
| 4 | Restarted PDHG on CPU and CUDA | |
| 5 | Branch & bound → MILP | |
| 6–10 | Performance, branch & cut, IPM/QP, robustness, packaging | |

There is no solver engine yet. `solve()` classifies the model and reports
`not_solved` rather than returning a plausible-looking zero — see the Evidence rules in
[`CLAUDE.md`](CLAUDE.md).

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
```

## Layout

```
include/sankhya/  public headers — Model, Solution, Options, tolerances, sparse containers
src/core          Model/Solution implementation and the solve() dispatcher
src/util          logging, timers, arena allocator, option registry
src/la            sparse linear algebra
src/io            model readers and writers            (Phase 2)
src/simplex       primal and dual revised simplex      (Phase 2, 6)
src/pdhg src/gpu  first-order method, CUDA backend     (Phase 4)
src/mip           branch and cut                       (Phase 5)
src/ipm src/qp    interior point, convex QP            (Phase 8)
tests/  bench/  tools/  docs/  demo/
```

## Licence

Apache-2.0.

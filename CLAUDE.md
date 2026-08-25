# SANKHYA — Claude Code operating rules

## What this is

A mathematical optimization solver core (LP, MILP, convex QP) written from scratch in C++20
for Smart India Hackathon 2026 PS26119, issued by Mangalore Refinery and Petrochemicals.
Judges are industrial optimization people. Deliverable = solver engine + C API + CLI +
Python bindings + benchmark evidence. No GUI.

Two LP engines: revised simplex (CPU, exact, produces a basis, is the node solver for
branch & cut) and restarted PDHG (CPU + CUDA, first-order, this is the GPU story).
Architecture must stay modular enough to add MIQP, NLP and MINLP later.

## Deadline

20 September 2026. When choosing between "more correct" and "more features", choose more
correct. A wrong answer scores zero; a missing feature scores partial.

## Dependency and provenance policy

### Red line — the only hard rule

No source code from an optimization SOLVER may be copied, vendored, linked, or read:
CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, lp_solve, OSQP, PDLP, cuPDLP / cuPDLP-C / cuPDLPx,
OR-Tools GLOP or CP-SAT, Gurobi, CPLEX, Xpress. Not their simplex, not their cuts, not
even their MPS reader. If you land on one of those repos, stop and back out.

### Explicitly ALLOWED — use freely, don't ask

1. **Your own knowledge.** You know these algorithms. Write them from what you know plus
   the literature. Cite the paper or textbook above each implementation for the judges,
   not because you needed to look it up.
2. Papers, theses, textbooks, lecture notes, solver user manuals.
3. Non-solver libraries, licence permitting: zlib, fmt, spdlog, CLI11, nlohmann/json,
   GoogleTest, pybind11, cuSPARSE / cuBLAS / cuSOLVER (vendor BLAS, not solvers).
   Graph ordering routines (AMD, COLAMD, RCM) if permissively licensed — check the
   per-module licence, SuiteSparse modules differ from each other.
   Eigen in TESTS ONLY as a reference oracle, never in src/.
4. Copy freely from non-solver repos: CMake patterns, CI YAML, benchmark and plotting
   scripts, docs site config, .clang-format, issue templates.
5. API shape. Imitating the surface of a known solver API (create/set/solve/query, string
   options, callback signatures) is interface compatibility, not derivation. It makes us
   drop-in adoptable. Use public header docs and manuals, not source.
6. Benchmark instances and published reference optima: MIPLIB, Netlib, QPLIB, Mittelmann —
   plus Mittelmann's reporting methodology.

### Provenance discipline

Maintain docs/PROVENANCE.md continuously, not at the end:
- dependency table: name, version, licence, one line on why it is not a solver
- algorithm table: every major algorithm, its citation, the file it lives in
- `ldd build/sankhya-cli` output and the full CMake link line
- SPDX SBOM generated in CI

If unsure whether something crosses the red line, log it under "Judgement calls" with your
reasoning and tell me. Never silently decide either way.

## Evidence rules — the most important section

- NEVER report a benchmark number, pass rate, or speedup you did not just produce by
  running a command in this session. Paste the real terminal output.
- If a test fails, say it failed. Do not "fix" it by loosening a tolerance without saying
  so explicitly and giving the numerical justification.
- Every benchmark run writes a CSV to bench/results/ with: instance, sha256 of the instance
  file, our objective, published reference objective, absolute and relative gap, status,
  wall time, iterations or nodes, git commit, machine tag. No CSV, no claim.
- docs/BENCHMARKS.md is generated from those CSVs by a script so it cannot drift.
- Never mark a task done on "it should work". Compile it and run it.
- This is numerical code. Wrong answers compile, run, print, and look completely correct.
  There is no stack trace. Assume output is wrong until a benchmark or the rational oracle
  says otherwise.

## Definition of done for any unit of work

- [ ] `cmake --build build -j` clean with -Wall -Wextra -Werror
- [ ] `ctest --test-dir build --output-on-failure` passes
- [ ] ASan/UBSan clean on the touched path
- [ ] Netlib pass-rate did not drop
- [ ] Rational oracle fuzz still reports 0 mismatches
- [ ] Algorithm cited in a code comment
- [ ] docs/PROVENANCE.md updated if a dependency or algorithm was added
- [ ] Conventional commit, one logical change

## Numerical conventions — no magic numbers, all in include/sankhya/tolerances.hpp

primal feasibility 1e-7 · dual feasibility 1e-7 · integrality 1e-6 · MIP relative gap 1e-4,
absolute 1e-6 · pivot/zero drop 1e-11 · Markowitz threshold 0.01 · PDHG reported at both
1e-4 and 1e-8. Double precision everywhere. No `float` in the numerical core.

## Frozen interfaces — never change without saying so explicitly

- `sankhya::Model` — what readers produce and every solver consumes
- `sankhya::Solution` — what every solver produces
- `solve(const Model&, const Options&) -> Solution` — the one entry point both engines
  implement, and the seam where QP/MIQP/NLP engines plug in later

tools/verify_solution.py consumes the written .sol file only. It never links our C++.

## Layout

    include/sankhya/   public headers, model.hpp, tolerances.hpp, sankhya.h (C API)
    src/core           Model/Solution implementation and the solve() dispatcher (the seam)
    src/util           logging, timers, arena allocator, options table
    src/la             sparse CSC/CSR, LU, FTRAN/BTRAN
    src/io             MPS + LP readers, solution and JSON writers
    src/presolve       reductions + postsolve stack
    src/simplex        primal & dual revised simplex
    src/pdhg           restarted PDHG (CPU)
    src/ipm            interior point
    src/qp             convex QP
    src/mip            branch & cut
    src/gpu            CUDA kernels, guarded by SANKHYA_ENABLE_CUDA
    src/api            C API
    apps/sankhya-cli · bindings/python · tools/ · tests/ · bench/ · data/casestudies/ · docs/

## Workflow

Branch per task, PR into main, squash merge. No file over ~600 lines.

CI gates formatting with clang-format **22.1.8** from pip, and clang-format's output changes
between major versions, so a distro clang-format will "fix" the tree into a state CI then
rejects. Run the pinned one before pushing - it provisions itself on first use:

    scripts/format.sh            # rewrite in place
    scripts/format.sh --check    # exactly what CI runs

The CPU build must work with zero CUDA installed — all GPU code behind
`#ifdef SANKHYA_ENABLE_CUDA` plus a runtime `--gpu` flag with silent CPU fallback.

## Local toolchain note (Windows dev boxes)

Do not trust PATH order for the compiler. Boxes seen so far:

- MSYS2 UCRT64, `C:\msys64\ucrt64\bin\g++.exe` (GCC 16.1.0) - current primary
- Strawberry Perl MinGW-W64, `C:\Strawberry\c\bin\g++.exe` (GCC 13.2.0)
- a stale MinGW 6.3.0 that predates C++20 entirely and must never be selected

`scripts/configure.sh` probes these in order, checks `-dumpversion >= 10`, and prepends the
chosen toolchain's bin directory to PATH so the cmake/ninja shipped beside the compiler win
over any unrelated one. Always configure through it:

    scripts/configure.sh build Release

If cmake or ninja are missing on an MSYS2 box:

    pacman -S --needed mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja

CI is ubuntu-latest and is the authority on `-Werror` cleanliness; Windows is the
convenience build.

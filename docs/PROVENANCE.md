# SANKHYA — provenance

This document exists so that the claim "built from mathematical foundations, not wrapped
around an existing solver" can be **checked** rather than believed. It is maintained
continuously, not written at the end.

Last updated: **Phase 6** (sparse LU), with the HiGHS comparison run. Every number and every command output below was
produced by running the command shown, on the machine described, at the commit recorded.

---

## 1. The red line

No source code from an optimization **solver** is copied, vendored, linked, or read. That
list explicitly includes CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, lp_solve, OSQP, PDLP,
cuPDLP / cuPDLP-C / cuPDLPx, OR-Tools GLOP and CP-SAT, Gurobi, CPLEX and Xpress — not their
simplex, not their cuts, not their MPS readers.

HiGHS appears in this project **only** as the comparison baseline in
`bench/runners/compare.py`. It runs either as an externally installed command-line binary or,
where none is present, through the `highspy` pip package in a separate Python process. It is
never linked into SANKHYA, never a build dependency, and no part of its source informs ours.
The comparison results are in `docs/BENCHMARKS.md` section 3.

---

## 2. Dependency table

Every dependency is a general-purpose library. None of them solves an optimization problem.

| Dependency | Version | Licence | Linked? | Why it is not a solver |
|---|---|---|---|---|
| **fmt** | 10.2.1 | MIT | yes, static | String formatting. Produces text from values. |
| **CLI11** | 2.4.2 | BSD-3-Clause | header-only | Command-line argument parsing. |
| **nlohmann/json** | 3.11.3 | MIT | header-only | JSON serialisation for the `--stats` result blob. |
| **OpenMP runtime** (libgomp with GCC, libomp with Clang) | as shipped by the compiler | GPL-3.0 with the GCC Runtime Library Exception / Apache-2.0 with LLVM exception | optional, found by CMake | The compiler's own thread runtime for `#pragma omp parallel for`. Schedules loops; contains no numerical code of any kind. |
| **zlib** | 1.3.1 | zlib | yes, static | DEFLATE decompression, for `.mps.gz` inputs (Phase 2). |
| **GoogleTest** | 1.14.0 | BSD-3-Clause | test binary only | Unit test framework. Never linked into `sankhya_core`. |
| **highspy** | pip, benchmark only | MIT | **never linked** | The HiGHS solver, used ONLY as the comparison baseline in `bench/runners/compare.py`. It runs in a separate Python process, is not a build dependency, and nothing in `src/` knows it exists. Its source does not inform ours - see the red line in section 1. |

Considered, and not present. Each was allowed by the policy above; none turned out to be
needed, and none is linked, vendored or fetched:

| Dependency | Licence | Status |
|---|---|---|
| pybind11 | BSD-3-Clause | Not used. The Python bindings (#59, #129) are `ctypes` over the C API and compile nothing. |
| cuSPARSE / cuBLAS | NVIDIA SDK | Not used yet. The CUDA backend is #16-#19; when it lands these supply SpMV and dense kernels, which are matrix arithmetic, not an optimization algorithm. |
| AMD / COLAMD ordering | per-module | Not used. The sparse LDL^T (#70) carries its own minimum-degree ordering, and the LU orders by Markowitz counts. |
| Eigen | MPL-2.0 | Not used. `DenseLu` (`src/simplex/dense_lu.cpp`) is the reference oracle the sparse LU is tested against. |

---

## 2b. Toolchains this has been built and tested with

PS26119 asks for a foundation someone else can pick up, and "it builds here" is not that.
These are the toolchains the evidence in `docs/BENCHMARKS.md` was actually produced with, and
the ones CI exercises on every push. `scripts/preflight.sh` reports what it finds on the
machine it runs on, and `scripts/configure.sh` chooses among them rather than trusting the
order of `PATH`.

| Where | Compiler | Build system | Python | What it is the authority on |
|---|---|---|---|---|
| CI, `ubuntu-latest` | the runner's default GCC | CMake >= 3.20, Ninja | 3.x as shipped | `-Wall -Wextra -Werror` cleanliness, the ASan/UBSan run, and the Netlib gates. A portability claim rests on the Linux build. |
| Development, Windows 11 | GCC 16.1.0, MSYS2 UCRT64 (`C:\msys64\ucrt64\bin\g++.exe`) | CMake 4.3.3, Ninja 1.13.2 | 3.11.9 | Every benchmark CSV in `bench/results/` whose `machine` column reads `Windows-AMD64`. |
| Development, Windows 11 (fallback) | GCC 13.2.0, Strawberry Perl MinGW-W64 | as shipped | 3.x | A second Windows compiler the configure script accepts; no committed CSV comes from it. |

**One toolchain must never be selected**: a MinGW 6.3.0 that predates C++20 and sits on
`PATH` on at least one development box. `scripts/configure.sh` rejects any `g++` reporting a
major version below 10, which is why it exists rather than the build simply calling `cmake`.

Third-party sources are pinned by tag in `CMakeLists.txt`, and `docs/sbom.spdx.json` - the
SPDX 2.3 bill of materials - is generated from those pins by `tools/make_sbom.py`. CI runs
`tools/make_sbom.py --check`, so a dependency added to the build without an entry in the
table above turns the tree red instead of passing quietly.

## 3. Algorithm citation table

Every algorithm we implement is cited at its implementation site as well as here. The
citation records where the *mathematics* came from; the code is written from the
mathematics, not transcribed from anyone's implementation.

| Algorithm | Citation | File |
|---|---|---|
| Compressed-column storage, counting-sort transpose | Davis, *Direct Methods for Sparse Linear Systems* (SIAM, 2006), ch. 2 | `src/la/sparse.cpp` |
| Dense-accumulator sparse vector (FTRAN/BTRAN result pattern) | Davis, ibid.; Hall & McKinnon on hyper-sparsity | `include/sankhya/sparse.hpp`, `src/la/sparse.cpp` |
| Markowitz threshold pivoting (constant only, so far) | Suhl & Suhl, *Computing sparse LU factorizations for large-scale linear programming bases* | `include/sankhya/tolerances.hpp` |
| MPS format: sections, RANGES and BOUNDS semantics | IBM MPS specification; Maros, *Computational Techniques of the Simplex Method* (Kluwer, 2003), appendix A | `src/io/mps_reader.cpp` |
| CPLEX LP format | Public CPLEX and Gurobi reference manuals (documentation only) | `src/io/lp_reader.cpp` |
| Dense LU with partial pivoting; transposed triangular solves | Golub & Van Loan, *Matrix Computations* (4th ed.), sections 3.2 and 3.4 | `src/simplex/dense_lu.cpp` |
| Sparse LU with Markowitz pivoting and threshold stability | Markowitz, *The elimination form of the inverse and its application to linear programming*, Management Science 3 (1957); Suhl & Suhl, *Computing sparse LU factorizations for large-scale linear programming bases*, ORSA J. Computing 2 (1990); Duff, Erisman & Reid, *Direct Methods for Sparse Matrices* (2nd ed., 2017), ch. 7-8 | `src/la/lu.cpp` |
| Basis update, product form of the inverse | Dantzig & Orchard-Hays, *The product form for the inverse in the simplex method*, Mathematical Tables and Other Aids to Computation 8 (1954) | `src/la/lu.cpp` |
| Bounded-variable revised primal simplex | Dantzig, *Linear Programming and Extensions* (1963); Chvátal, *Linear Programming* (1983), ch. 3 and 8 | `src/simplex/primal_simplex.cpp` |
| Piecewise-linear (composite) phase 1, no artificial variables | Maros, *Computational Techniques of the Simplex Method*, ch. 9 | `src/simplex/primal_simplex.cpp` |
| Bounded-variable revised dual simplex with the bound-flipping ratio test, dual devex pricing, artificial bounds for dual infeasibility, warm start from a basis (#65; `--option algorithm=dual-simplex`, and the branch-and-bound node engine) | Lemke, *The dual method of solving the linear programming problem*, Naval Research Logistics Quarterly 1 (1954); Maros, *Computational Techniques of the Simplex Method*, ch. 10; Koberstein, *The dual simplex method, techniques for a fast and stable implementation*, PhD thesis, Paderborn (2005), sec. 3.3 and 4.5; Forrest & Goldfarb (1992), above, sec. 3 for the dual weights | `src/simplex/dual_simplex.cpp`, shared state in `src/simplex/simplex_core.hpp` |
| Bland's anti-cycling rule | Chvátal, *Linear Programming*, ch. 3 | `src/simplex/primal_simplex.cpp` |
| Devex pricing (default since #66 was re-measured; `--option pricing=dantzig` selects the old rule) | Forrest & Goldfarb, *Steepest-edge simplex algorithms for linear programming*, Mathematical Programming 57 (1992); the approximation is Harris (1973), below | `src/simplex/primal_simplex.cpp` |
| Harris two-pass ratio test, long-step bound flipping (opt-in; `--option ratio_test=harris`, #67) | Harris, P.M.J., *Pivot selection methods of the Devex LP code*, Mathematical Programming 5 (1973), 1-28; long-step generalises the piecewise-linear phase 1 already cited to Maros below | `src/simplex/primal_simplex.cpp` |
| Exact rational tableau simplex (test oracle) | Chvátal, *Linear Programming*, ch. 2–3 | `tests/oracles/rational_simplex.cpp` |
| Primal-dual hybrid gradient (the base iteration) | Chambolle & Pock, *A first-order primal-dual algorithm for convex problems with applications to imaging*, JMIV 40(1), 2011, Algorithm 1 | `src/pdhg/pdhg.cpp` |
| Adaptive step size, primal weight, restarts | Applegate et al., *Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient* (PDLP), NeurIPS 2021, sections 3.1, 3.2, 4.3 | `src/pdhg/pdhg.cpp` |
| GPU-oriented restarted PDHG (design reference) | Lu & Yang, *cuPDLP.jl*, arXiv:2311.12180 | `src/pdhg/pdhg.cpp` |
| Ruiz equilibration | Ruiz, *A scaling algorithm to equilibrate both rows and columns norms in matrices*, RAL-TR-2001-034 | `src/la/scaling.cpp` |
| Diagonal preconditioning, alpha = 1 | Pock & Chambolle, *Diagonal preconditioning for first order primal-dual algorithms*, ICCV 2011, section 4 | `src/la/scaling.cpp` |
| Moreau decomposition for the support-function prox | Rockafellar, *Convex Analysis*, theorem 31.5 | `src/pdhg/pdhg.cpp` |
| Branch and bound | Land & Doig, *An automatic method of solving discrete programming problems*, Econometrica 28(3), 1960; Wolsey, *Integer Programming*, ch. 7 | `src/mip/branch_and_bound.cpp` |
| Node propagation from row activities | Savelsbergh, *Preprocessing and probing for MIP*, ORSA J. Computing 6(4), 1994 | `src/mip/branch_and_bound.cpp` |
| Search shape: propagation at nodes, incumbent as cutoff | Achterberg, *Constraint Integer Programming* (thesis, 2007), ch. 5–6 | `src/mip/branch_and_bound.cpp` |
| Hyper-sparse back-substitution through U in the FTRAN, U stored by column as well as by row, zero results skipped; the row-wise gather kept as the reference the push form is tested against (#68) | Gilbert & Peierls, *Sparse partial pivoting in time proportional to arithmetic operations*, SIAM J. Sci. Stat. Comput. 9 (1988) | `src/la/lu.cpp` (`solve`, `solve_reference`) |
| Sparse symmetric LDL^T: minimum degree ordering, elimination tree, symbolic pattern once, up-looking numeric factorization with diagonal regularization (#70) | Davis, *Direct Methods for Sparse Linear Systems*, SIAM (2006), ch. 4; Liu, *The role of elimination trees in sparse factorization*, SIAM J. Matrix Anal. Appl. 11 (1990); Tinney & Walker, Proc. IEEE 55 (1967); Altman & Gondzio, Optim. Methods Softw. 11 (1999) | `src/la/ldl.cpp` |
| Primal-dual interior-point method, Mehrotra predictor-corrector on the normal equations of the bounded-variable form; no basis, no infeasibility certificate (#56; `--option algorithm=ipm`) | Mehrotra, SIAM J. Optim. 2 (1992); Wright, *Primal-Dual Interior-Point Methods*, SIAM (1997), ch. 10–11; Altman & Gondzio (1999) for the regularization | `src/ipm/ipm.cpp` |
| Reliability branching: pseudocosts, strong branching on unreliable candidates with capped warm-started dual probes, product score (#69; `--option mip_branching=most-fractional` keeps the old rule) | Achterberg, Koch & Martin, *Branching rules revisited*, Operations Research Letters 33 (2005), 42–54 | `src/mip/branch_and_bound.cpp` |
| Iterative refinement of the final basis, primal and dual, residual in compensated arithmetic (#72) | Wilkinson, *Rounding Errors in Algebraic Processes* (1963), ch. 4; Ogita, Rump & Oishi, *Accurate sum and dot product*, SIAM J. Sci. Comput. 26 (2005) | `src/simplex/primal_simplex.cpp` (`refine_final_basis`) |
| Cost perturbation on a dual-degenerate stall, nonbasic costs only so dual feasibility is preserved by construction | Maros, *Computational Techniques of the Simplex Method*, ch. 9 (the bound-shifting scheme, applied to the dual's costs); Koberstein (2005), above, sec. 6.2 | `src/simplex/dual_simplex.cpp` |
| Robustness suite: the classic cycling examples, adversarial families judged by the exact oracle, and the conditioning / near-parallel / cost-ratio / redundancy / degeneracy sweeps with an optimum known by construction (#71) | Beale, *Cycling in the dual simplex algorithm*, Naval Research Logistics Quarterly 2 (1955); Kuhn's example as given in Chvátal, *Linear Programming* (1983), ch. 3; the KKT construction is the oracle fuzz's (above) | `tests/robustness/test_robustness.cpp`, `bench/runners/robustness.py` |
| MIQP: branch and bound over convex QP node relaxations | Gupta & Ravindran, *Branch and bound experiments in convex nonlinear integer programming*, Management Science 31(12), 1985 | `src/mip/branch_and_bound.cpp` |
| Exact rational branch and bound (test oracle) | as above, in exact arithmetic | `tests/oracles/rational_simplex.cpp` |
| Shifted geometric mean benchmark reporting | Mittelmann, plato.asu.edu benchmark methodology | `bench/runners/make_benchmarks_doc.py` |
| LP duality checks (feasibility, complementary slackness, strong duality) | Chvátal, *Linear Programming*, ch. 5 | `tools/verify_solution.py` |
| Lifted knapsack cover cuts at the root, exact sequential lifting through a 0/1 knapsack dynamic programme (#159; `--option enable_root_cuts=true`, off by default by measurement) | Balas, *Facets of the knapsack polytope*, Math. Programming 8 (1975); Wolsey, *Faces for a linear inequality in 0-1 variables*, ibid.; Zemel, *Easily computable facets of the knapsack polytope*, Math. Oper. Res. 14 (1989); Crowder, Johnson & Padberg, Oper. Res. 31 (1983); Gu, Nemhauser & Savelsbergh, INFORMS J. Computing 10 (1998) | `src/mip/cuts.cpp` |
| Gomory mixed-integer cuts at the root, from tableau rows reconstructed off the final basis (#159) | Gomory, *An algorithm for the mixed integer problem*, RAND RM-2597 (1960); Balas, Ceria, Cornuéjols & Natraj, *Gomory cuts revisited*, Oper. Res. Letters 19 (1996); Marchand & Wolsey, *Aggregation and mixed integer rounding to solve MIPs*, Oper. Res. 49 (2001) | `src/mip/cuts.cpp` |
| Bound rounding on integral rows; cut filtering by density, coefficient range and violation, duplicates dropped | Chvátal, *Edmonds polytopes and a hierarchy of combinatorial problems*, Discrete Math. 4 (1973); Achterberg, *Constraint Integer Programming* (thesis, 2007), ch. 8 | `src/mip/cuts.cpp` |
| Cut validity gate: every family checked in exact arithmetic against the rational oracle's optimum, with a deliberately invalid cut as the negative control | as above, in exact arithmetic | `tests/unit/test_cuts.cpp` |
| Convex QP by a primal-dual proximal method with a forward step for the smooth 0.5 x'Qx term (#55; the LP engine's iteration with the gradient added) | Condat, *A primal-dual splitting method for convex optimization involving Lipschitzian, proximable and linear composite terms*, J. Optim. Theory Appl. 158 (2013); Vũ, *A splitting algorithm for dual monotone inclusions involving cocoercive operators*, Adv. Comput. Math. 38 (2013); Chambolle & Pock (2011) for the Q = 0 case | `src/qp/qp_condat_vu.cpp` |
| Convexity decided before any arithmetic: LDL^T without interchanges on sense · Q, a negative pivot returned as the certificate; dense, and above 2000 columns it reports unverified rather than guessing | Golub & Van Loan, *Matrix Computations* (4th ed.), section 4.1; Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed.), ch. 10 | `src/qp/convexity.cpp` |
| Presolve: empty, singleton, redundant and forcing rows, fixed and empty columns (#43), free column singletons and doubleton equations (#92); every reduction pushes a record and postsolve replays the stack in reverse, with the round-trip asserted against the rational oracle | Brearley, Mitra & Williams, *Analysis of mathematical programming problems prior to applying the simplex algorithm*, Math. Programming 8 (1975); Andersen & Andersen, *Presolving in linear programming*, Math. Programming 71 (1995); Achterberg et al., *Presolve reductions in mixed integer programming*, INFORMS J. Computing 32 (2020) | `src/presolve/presolve.cpp` |
| Root diving heuristic: fix the least-fractional integer column, re-solve, to a bounded depth, so the search starts with an incumbent (#25) | Achterberg, *Constraint Integer Programming* (thesis, 2007), ch. 6; Berthold, *Primal heuristics for mixed integer programs* (diploma thesis, TU Berlin, 2006) | `src/mip/branch_and_bound.cpp` |
| Singular-basis repair: the LU reports dependent columns by position, the simplex evicts them for their logicals, and a budget of stalled repairs stops a solve that is being carried by repair alone (#34, #174) | Maros, *Computational Techniques of the Simplex Method*, section 9.4; Suhl & Suhl, *Computing sparse LU factorizations for large-scale linear programming bases*, ORSA J. Computing 2 (1990) | `src/la/lu.cpp`, `src/simplex/simplex_core.hpp` |

Of the work once listed here as "Phases 6 onwards", restarted PDHG, the Mehrotra
predictor-corrector, Gomory mixed-integer and lifted cover cuts, Devex pricing (#66, the
default) and the Harris ratio test (#67, opt-in) have all landed and are in the table
above. Not landed: the Forrest-Tomlin update (Forrest & Tomlin 1972; what ships is the
product form of the inverse with refactorization forced by an accuracy check),
mixed-integer rounding cuts, and cuts below the root.

---

## 4. Machine-checkable evidence

### 4.1 What the binary actually links against

Regenerated at **Phase 2**, commit `12fa98b`. The Phase 1 text predicted that zlib would
appear here once the gzip MPS path became real; it now does, on both platforms, which is
what that prediction was for.

Windows 11, **GCC 16.1.0 (MSYS2 UCRT64), Release**:

```
$ objdump -p build/sankhya.exe | grep "DLL Name" | sort -u
        DLL Name: KERNEL32.dll
        DLL Name: api-ms-win-crt-convert-l1-1-0.dll
        DLL Name: api-ms-win-crt-environment-l1-1-0.dll
        DLL Name: api-ms-win-crt-filesystem-l1-1-0.dll
        DLL Name: api-ms-win-crt-heap-l1-1-0.dll
        DLL Name: api-ms-win-crt-locale-l1-1-0.dll
        DLL Name: api-ms-win-crt-math-l1-1-0.dll
        DLL Name: api-ms-win-crt-private-l1-1-0.dll
        DLL Name: api-ms-win-crt-runtime-l1-1-0.dll
        DLL Name: api-ms-win-crt-stdio-l1-1-0.dll
        DLL Name: api-ms-win-crt-string-l1-1-0.dll
        DLL Name: api-ms-win-crt-time-l1-1-0.dll
        DLL Name: api-ms-win-crt-utility-l1-1-0.dll
        DLL Name: zlib1.dll
```

The Windows C runtime plus **zlib1.dll**. Note that the `-static` flag added in Phase 1 does
not cover this one: MSYS2 provides zlib as an import library (`libz.dll.a`), so the Windows
`.exe` is no longer self-contained. That is recorded rather than fixed. Windows is the
convenience build; Linux is what CI gates and what the claim rests on. It is logged as
judgement call 7 below.

The Linux build is the authoritative one. Produced by the `provenance` job on
**ubuntu-latest, GCC (runner default), Release**, run 32867892122:

```
$ ldd build/sankhya
        linux-vdso.so.1 (0x00007f2012bf2000)
        libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f2012b07000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f2012800000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f2012717000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f2012ad9000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f2012400000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f2012bf4000)

$ ldd build/sankhya | grep -Ei 'cbc|clp|highs|scip|soplex|glpk|lpsolve|osqp|ortools|gurobi|cplex|xpress'
OK: no solver library is linked
```

zlib, the C++ runtime, libc and libm - and, on a build with `SANKHYA_WITH_OPENMP` on (the
default, #57), the compiler's own OpenMP runtime, `libgomp` with GCC, which the listing
above predates. Nothing else.

That grep is a **gate**, not a comment: the `provenance` job in `.github/workflows/ci.yml`
exits non-zero if any linked library name matches a known solver. The claim cannot silently
rot.

### 4.2 The full link line

Asked of Ninja directly so it cannot drift from what was really executed.

Linux, from the CI `provenance` job (run 32867892122):

```
$ ninja -C build -t commands sankhya | tail -1
/usr/bin/c++ -O3 -DNDEBUG -Wl,--dependency-file=CMakeFiles/sankhya-cli.dir/link.d     CMakeFiles/sankhya-cli.dir/apps/sankhya-cli/main.cpp.o     -o sankhya     libsankhya_core.a     _deps/fmt-build/libfmt.a     /usr/lib/x86_64-linux-gnu/libz.so
```

Windows, same command on the local tree:

```
$ ninja -C build -t commands sankhya-cli | tail -1
g++ -O3 -DNDEBUG -static-libgcc -static-libstdc++ -static     CMakeFiles/sankhya-cli.dir/apps/sankhya-cli/main.cpp.obj     -o sankhya.exe     -Wl,--out-implib,libsankhya.dll.a     libsankhya_core.a     _deps/fmt-build/libfmt.a     C:/msys64/ucrt64/lib/libz.dll.a
```

Three artifacts on each platform: our own core, fmt, and the system zlib. CLI11 and
nlohmann/json are header-only and appear as includes rather than archives; they are in the
dependency table above and visible in `CMakeLists.txt`.

### 4.3 SBOM

An SPDX SBOM is generated in CI by the `provenance` job and attached as a build artifact
(`sankhya-sbom.spdx.json`).

---

## 5. Judgement calls

Decisions where the dependency policy needed interpretation. Per `CLAUDE.md`, these are
recorded rather than silently made.

| # | Question | Decision | Reasoning |
|---|---|---|---|
| 1 | Does imitating a familiar solver *log layout* (iteration / objective / primal inf / dual inf / time) cross the line? | Allowed | Interface familiarity, not derivation. The layout is visible in published user manuals and screenshots; no source was consulted. It makes the output readable to the industrial audience that reads such logs daily. Same reasoning as the API-shape allowance in `CLAUDE.md`. |
| 2 | Does a string-keyed option table with typed values imitate a solver's API? | Allowed | Explicitly permitted by `CLAUDE.md` item 5 (API shape). The shape is create/set/solve/query with string options; the registry, parser and storage are our own. |
| 3 | System zlib is picked up in preference to a fetched copy. Does that weaken the provenance claim? | Acceptable, and recorded | zlib is a compression library under a permissive licence, present on essentially every system. The exact resolved path appears in the link line above, so which copy was used is always visible. |
| 4 | The Windows build links libstdc++ and libgcc **statically** (`-static`). | Deliberate | It removes a start-up failure caused by an older MinGW earlier on PATH, and it makes the dependency list above shorter and easier to audit, not longer. It has no effect on the Linux build, which CI treats as authoritative. |
| 5 | The MPS reader imitates the *file format* of CPLEX/Xpress inputs, and the LP reader imitates the CPLEX LP format. | Allowed | A file format is an interface, not an implementation. `CLAUDE.md` item 5 covers API shape for the same reason, and reading a format everyone's solver reads is what makes us drop-in adoptable. Both readers were written from the published format specification and from the textbook reference above; **no solver's reader source was consulted**, which `CLAUDE.md` calls out by name as forbidden. |
| 6 | Netlib distributes its LP test set in a packed `emps` encoding, not as plain MPS, and expanding it requires the `emps` decoder Netlib ships alongside the data. Does fetching that decoder cross the red line? | **RESOLVED — allowed, and not vendored.** See row 8, which records how it is actually done. | Logged as OPEN while Phase 2 was in progress, and Phase 2 correctly claimed no Netlib result at the time. It was settled when `bench/runners/fetch_data.py` was written: the decoder is DOWNLOADED AND COMPILED at fetch time rather than committed, so this repository contains no third-party source, and the sha256 of everything downloaded is recorded. `emps` is a file-format converter, not a solver, so it sits outside the red line, and `CLAUDE.md` item 6 permits the Netlib dataset of which this is the delivery mechanism. Nothing it produces is linked into SANKHYA. The Netlib results in `docs/BENCHMARKS.md` rest on this decision. See issue #62. |
| 7 | The Windows `.exe` dynamically links `zlib1.dll`, so the Phase 1 claim that it is self-contained no longer holds. | Recorded, not fixed | MSYS2 ships zlib only as an import library, and `-static` cannot statically link what has no static archive. Fixing it would mean forcing the bundled zlib build on Windows, which trades an audit-surface improvement for a divergence between the two platforms' dependency sets. Linux is what CI gates and what the provenance claim rests on; Windows is the convenience build. Revisit at Phase 10 packaging if we ship a Windows binary. |
| 8 | `bench/runners/fetch_data.py` downloads Netlib's `emps.c` decoder and COMPILES it at fetch time. Is that third-party source in the project? | Allowed, and not vendored | Netlib distributes its LP set in a custom compressed encoding, and `emps.c` is the decoder they publish beside it. It is a file-format converter, not a solver, so it is outside the red line. It is downloaded at fetch time rather than committed, so this repository contains no third-party source; its sha256 is recorded in `data/netlib/reference.json`. Nothing it produces is linked into SANKHYA - it runs once, offline, to turn an archive format into plain MPS. |
| 9 | MPS says a negative `UP` bound with no explicit lower bound implies `lower = -inf`. That is documented for continuous columns and **implementation-defined for integer ones**, where established readers disagree. Which reading do we take? | Apply the convention to integer columns too, and warn | Declining to choose was tried first and was worse than either choice: leaving `lower = 0` produces `[0, -5]`, an empty interval, so `Model::validate()` rejected the model and the file could not be loaded **at all** - and the resulting error named crossed bounds, which is the symptom rather than the cause. It also put the C++ reader at odds with `tools/verify_solution.py`, which already applies the convention; two components disagreeing about what the same bytes mean is exactly what that verifier exists to catch, so the disagreement sitting inside the pair weakened the check against every instance carrying such a bound. The warning is kept so the ambiguity stays visible in the log. See issue #8. |

---

## 6. Reproducing this

```bash
scripts/configure.sh build Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On Linux, `ldd build/sankhya`. On Windows, `objdump -p build/sankhya.exe | grep "DLL Name"`.

To reproduce the Phase 2 end-to-end result:

```bash
build/sankhya info demo/crude_blend.mps
build/sankhya solve demo/crude_blend.mps --write-sol blend.sol --stats blend.json
build/sankhya solve demo/crude_blend.lp
```

The MPS and LP files describe the same model; the two solutions must be identical.

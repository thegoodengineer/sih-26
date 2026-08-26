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
| **zlib** | 1.3.1 | zlib | yes, static | DEFLATE decompression, for `.mps.gz` inputs (Phase 2). |
| **GoogleTest** | 1.14.0 | BSD-3-Clause | test binary only | Unit test framework. Never linked into `sankhya_core`. |
| **highspy** | pip, benchmark only | MIT | **never linked** | The HiGHS solver, used ONLY as the comparison baseline in `bench/runners/compare.py`. It runs in a separate Python process, is not a build dependency, and nothing in `src/` knows it exists. Its source does not inform ours - see the red line in section 1. |

Planned, not yet present:

| Dependency | Phase | Licence | Why it is not a solver |
|---|---|---|---|
| pybind11 | 10 | BSD-3-Clause | C++/Python binding glue. |
| cuSPARSE / cuBLAS | 4, 8 | NVIDIA SDK | Vendor BLAS-level kernels (SpMV, dense linear algebra). They provide matrix arithmetic, not an optimization algorithm. |
| AMD / COLAMD ordering | 6, 8 | per-module, to be checked before use | Graph fill-reducing permutations. Combinatorics on a sparsity pattern, not optimization. |
| Eigen | 6+ | MPL-2.0 | **Tests only**, as a dense reference oracle. Never in `src/`. |

---

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
| Bounded-variable revised primal simplex | Dantzig, *Linear Programming and Extensions* (1963); Chvátal, *Linear Programming* (1983), ch. 3 and 8 | `src/simplex/primal_simplex.cpp` |
| Piecewise-linear (composite) phase 1, no artificial variables | Maros, *Computational Techniques of the Simplex Method*, ch. 9 | `src/simplex/primal_simplex.cpp` |
| Bland's anti-cycling rule | Chvátal, *Linear Programming*, ch. 3 | `src/simplex/primal_simplex.cpp` |
| Exact rational tableau simplex (test oracle) | Chvátal, *Linear Programming*, ch. 2–3 | `tests/oracles/rational_simplex.cpp` |
| Primal-dual hybrid gradient (the base iteration) | Chambolle & Pock, *A first-order primal-dual algorithm for convex problems with applications to imaging*, JMIV 40(1), 2011, Algorithm 1 | `src/pdhg/pdhg.cpp` |
| Adaptive step size, primal weight, restarts | Applegate et al., *Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient* (PDLP), NeurIPS 2021, sections 3.1, 3.2, 4.3 | `src/pdhg/pdhg.cpp` |
| GPU-oriented restarted PDHG (design reference) | Lu & Yang, *cuPDLP.jl*, arXiv:2311.12180 | `src/pdhg/pdhg.cpp` |
| Ruiz equilibration | Ruiz, *A scaling algorithm to equilibrate both rows and columns norms in matrices*, RAL-TR-2001-034 | `src/pdhg/scaling.cpp` |
| Diagonal preconditioning, alpha = 1 | Pock & Chambolle, *Diagonal preconditioning for first order primal-dual algorithms*, ICCV 2011, section 4 | `src/pdhg/scaling.cpp` |
| Moreau decomposition for the support-function prox | Rockafellar, *Convex Analysis*, theorem 31.5 | `src/pdhg/pdhg.cpp` |
| Branch and bound | Land & Doig, *An automatic method of solving discrete programming problems*, Econometrica 28(3), 1960; Wolsey, *Integer Programming*, ch. 7 | `src/mip/branch_and_bound.cpp` |
| Node propagation from row activities | Savelsbergh, *Preprocessing and probing for MIP*, ORSA J. Computing 6(4), 1994 | `src/mip/branch_and_bound.cpp` |
| Search shape: propagation at nodes, incumbent as cutoff | Achterberg, *Constraint Integer Programming* (thesis, 2007), ch. 5–6 | `src/mip/branch_and_bound.cpp` |
| Exact rational branch and bound (test oracle) | as above, in exact arithmetic | `tests/oracles/rational_simplex.cpp` |
| Shifted geometric mean benchmark reporting | Mittelmann, plato.asu.edu benchmark methodology | `bench/runners/make_benchmarks_doc.py` |
| LP duality checks (feasibility, complementary slackness, strong duality) | Chvátal, *Linear Programming*, ch. 5 | `tools/verify_solution.py` |

Phases 6 onwards add: dual revised simplex (Maros; Huangfu & Hall), Forrest–Tomlin
update (Forrest & Tomlin 1972), Devex pricing (Forrest & Goldfarb 1992), Harris two-pass
ratio test (Harris 1973), restarted PDHG (Applegate et al.; Lu & Yang, arXiv:2311.12180;
arXiv:2507.14051), Mehrotra predictor–corrector (Nocedal & Wright; Gondzio), Gomory MIR and
cover cuts (Marchand & Wolsey; Wolsey), and branch-and-cut search (Achterberg).

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

zlib, the C++ runtime, libc and libm. Nothing else.

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
| 6 | **OPEN — needs a decision before Phase 3.** Netlib distributes its LP test set in a packed `emps` format, not as plain MPS. Expanding it requires the `emps.f` / `emps.c` expander that Netlib ships alongside the data (`https://www.netlib.org/lp/data/readme` directs users to it as the only way to obtain the instances). Does fetching that expander, or writing our own decoder from its format description, cross the red line? | **Not yet decided — flagged, not acted on** | The argument for *allowed*: `emps` is a data-format expander distributed with the benchmark dataset itself, it appears nowhere in the red line's enumerated list of solvers, and `CLAUDE.md` item 6 explicitly permits "benchmark instances and published reference optima: MIPLIB, Netlib, QPLIB, Mittelmann" — of which this is the delivery mechanism. The argument for *caution*: it is still third-party code associated with the LP ecosystem, and reverse-engineering the packed format blind risks a decoder that produces a well-formed MPS for the **wrong problem**, which is the exact silent-failure mode this project is built to avoid. Nothing has been downloaded or written. **Phase 2 therefore does not claim a Netlib result.** |
| 7 | The Windows `.exe` dynamically links `zlib1.dll`, so the Phase 1 claim that it is self-contained no longer holds. | Recorded, not fixed | MSYS2 ships zlib only as an import library, and `-static` cannot statically link what has no static archive. Fixing it would mean forcing the bundled zlib build on Windows, which trades an audit-surface improvement for a divergence between the two platforms' dependency sets. Linux is what CI gates and what the provenance claim rests on; Windows is the convenience build. Revisit at Phase 10 packaging if we ship a Windows binary. |

| 7 | `bench/runners/fetch_data.py` downloads Netlib's `emps.c` decoder and COMPILES it at fetch time. Is that third-party source in the project? | Allowed, and not vendored | Netlib distributes its LP set in a custom compressed encoding, and `emps.c` is the decoder they publish beside it. It is a file-format converter, not a solver, so it is outside the red line. It is downloaded at fetch time rather than committed, so this repository contains no third-party source; its sha256 is recorded in `data/netlib/reference.json`. Nothing it produces is linked into SANKHYA - it runs once, offline, to turn an archive format into plain MPS. |
| 8 | MPS says a negative `UP` bound with no explicit lower bound implies `lower = -inf`. That is documented for continuous columns and **implementation-defined for integer ones**, where established readers disagree. Which reading do we take? | Apply the convention to integer columns too, and warn | Declining to choose was tried first and was worse than either choice: leaving `lower = 0` produces `[0, -5]`, an empty interval, so `Model::validate()` rejected the model and the file could not be loaded **at all** - and the resulting error named crossed bounds, which is the symptom rather than the cause. It also put the C++ reader at odds with `tools/verify_solution.py`, which already applies the convention; two components disagreeing about what the same bytes mean is exactly what that verifier exists to catch, so the disagreement sitting inside the pair weakened the check against every instance carrying such a bound. The warning is kept so the ambiguity stays visible in the log. See issue #8. |

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

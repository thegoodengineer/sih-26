# SANKHYA — provenance

This document exists so that the claim "built from mathematical foundations, not wrapped
around an existing solver" can be **checked** rather than believed. It is maintained
continuously, not written at the end.

Last updated: **Phase 2** (I/O and the primal simplex). Every number and every command output below was
produced by running the command shown, on the machine described, at the commit recorded.

---

## 1. The red line

No source code from an optimization **solver** is copied, vendored, linked, or read. That
list explicitly includes CBC, Clp, HiGHS, SCIP, SoPlex, GLPK, lp_solve, OSQP, PDLP,
cuPDLP / cuPDLP-C / cuPDLPx, OR-Tools GLOP and CP-SAT, Gurobi, CPLEX and Xpress — not their
simplex, not their cuts, not their MPS readers.

HiGHS appears later in this project **only** as an externally installed comparison binary
invoked as a subprocess by `bench/runners/compare.py` (Phase 3). It is never linked, never
a build dependency, and no part of its source informs ours.

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
| Bounded-variable revised primal simplex | Dantzig, *Linear Programming and Extensions* (1963); Chvátal, *Linear Programming* (1983), ch. 3 and 8 | `src/simplex/primal_simplex.cpp` |
| Piecewise-linear (composite) phase 1, no artificial variables | Maros, *Computational Techniques of the Simplex Method*, ch. 9 | `src/simplex/primal_simplex.cpp` |
| Bland's anti-cycling rule | Chvátal, *Linear Programming*, ch. 3 | `src/simplex/primal_simplex.cpp` |

Phases 4 onwards add: dual revised simplex (Maros; Huangfu & Hall), Forrest–Tomlin
update (Forrest & Tomlin 1972), Devex pricing (Forrest & Goldfarb 1992), Harris two-pass
ratio test (Harris 1973), restarted PDHG (Applegate et al.; Lu & Yang, arXiv:2311.12180;
arXiv:2507.14051), Mehrotra predictor–corrector (Nocedal & Wright; Gondzio), Gomory MIR and
cover cuts (Marchand & Wolsey; Wolsey), and branch-and-cut search (Achterberg).

---

## 4. Machine-checkable evidence

### 4.1 What the binary actually links against

Produced on **Windows 11, GCC 13.2.0 (MinGW-W64, Strawberry), Release**, Phase 1:

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
```

The Windows C runtime and nothing else.

The Linux build is the one CI treats as authoritative. Produced by the `provenance` job on
**ubuntu-latest, GCC (runner default), Release**, commit `b6b1a80`:

```
$ ldd build/sankhya
        linux-vdso.so.1 (0x00007f7e571a0000)
        libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f7e56e00000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f7e570e4000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f7e56a00000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f7e56d17000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f7e571a2000)

$ ldd build/sankhya | grep -Ei 'cbc|clp|highs|scip|soplex|glpk|lpsolve|osqp|ortools|gurobi|cplex|xpress'
OK: no solver library is linked
```

The C++ runtime, libc and libm. Nothing else. zlib does not appear because no Phase 1 code
references a zlib symbol yet, so `--as-needed` dropped it; it will appear once the gzip MPS
path lands in Phase 2, and this section will be regenerated then.

That grep is a **gate**, not a comment: the `provenance` job in `.github/workflows/ci.yml`
exits non-zero if any linked library name matches a known solver, on every pull request. The
claim cannot silently rot.

### 4.2 The full link line

Verbatim, from the build system:

```
g++ -O3 -DNDEBUG -static-libgcc -static-libstdc++ -static
    CMakeFiles/sankhya-cli.dir/apps/sankhya-cli/main.cpp.obj
    -o sankhya.exe
    libsankhya_core.a
    _deps/fmt-build/libfmt.a
    C:/Strawberry/c/lib/libz.a
    -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32
    -loleaut32 -luuid -lcomdlg32 -ladvapi32
```

Three archives: our own core, fmt, and zlib. CLI11 and nlohmann/json are header-only and so
appear as includes rather than archives — they are listed in the dependency table above and
are visible in `CMakeLists.txt`.

The Linux link line, asked of Ninja directly in CI (run 32846138507) so it cannot drift from
what was really executed:

```
$ ninja -C build -t commands sankhya | tail -1
/usr/bin/c++ -O3 -DNDEBUG -Wl,--dependency-file=CMakeFiles/sankhya-cli.dir/link.d \
    CMakeFiles/sankhya-cli.dir/apps/sankhya-cli/main.cpp.o \
    -o sankhya \
    libsankhya_core.a \
    _deps/fmt-build/libfmt.a \
    /usr/lib/x86_64-linux-gnu/libz.so
```

Our own core, fmt, and the system zlib. That is the entire list.

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

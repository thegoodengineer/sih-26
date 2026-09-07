# SANKHYA — architecture

How the solver is put together, where each algorithm lives, and where the next engine plugs
in. PS26119 asks for a "transparent, extensible foundation"; this document is the map that
claim is checked against. Every citation for an algorithm named here is in
`docs/PROVENANCE.md`; every number is in `docs/BENCHMARKS.md`, generated from CSVs.

## 1. The shape in one paragraph

A `Model` goes in, a `Solution` comes out, through one function:

```
solve(const Model&, const Options&) -> Solution        (src/core/solve.cpp)
```

`solve()` classifies the model by what it contains — integrality, a quadratic objective,
neither — and dispatches to an engine. Every engine consumes the same `Model`, produces the
same `Solution`, and is judged by the same code afterwards: `Solution::recompute_quality()`
re-measures feasibility, the reduced costs and the duality gap against the original model,
and the dispatcher's status guard downgrades any claim the measurement does not support.
That seam is what makes a new engine a bounded piece of work: it has to produce a
`Solution`, and it gets the same audit as the ones that exist.

```
                MPS / LP file            C API            Python (pybind11)
                       │                    │                    │
                       └──────── readers (src/io) ───────────────┘
                                          │
                                        Model
                                          │
                              presolve (src/presolve)          ┐
                                          │                    │  LP path
                   ┌──────────────┬───────┴────────┬────────┐  │
              dual simplex   primal simplex      PDHG      IPM (algorithm=ipm)
              (default LP)   (algorithm=simplex) (pdhg)     │
                   └──────────────┴───────┬────────┴────────┘  │
                                       postsolve               ┘
                                          │
                     branch and bound (src/mip) ── node LP: warm-started dual simplex
                     convex QP (src/qp)        ── Condat–Vũ, MIQP nodes
                                          │
                              Solution ── recompute_quality ── status guard
                                          │
                        .sol / --stats JSON (src/io)   tools/verify_solution.py
```

## 2. Modules and their boundaries

| directory | what lives there | depends on | lines |
|---|---|---|---|
| `include/sankhya/` | the public headers: `model.hpp` (Model, Solution, BasisStatus), `options.hpp`, `tolerances.hpp` (every numerical constant, cited), `sankhya.h` (the C API), `io.hpp`, `mip.hpp`, `qp.hpp`, `pdhg.hpp` | nothing | 1.6k |
| `src/core/` | `Model`/`Solution` implementation, `recompute_quality()`, the `solve()` dispatcher and its status guard | everything below | 0.7k |
| `src/util/` | logging, the options registry (every option has a description and a default, checked for duplicates by a test), the version stamp | – | 0.8k |
| `src/la/` | CSC/CSR sparse matrix with row views, the sparse LU with Markowitz threshold pivoting and the product-form update (FTRAN/BTRAN, eta file), Ruiz + Pock–Chambolle equilibration | – | 1.6k |
| `src/io/` | MPS (fixed and free, RANGES, negative-UP convention, MARKER blocks, gzip) and LP readers, the `.sol` writer and the `--stats` JSON writer | core | 2.2k |
| `src/presolve/` | reductions (empty/fixed/singleton rows and columns, redundant rows, free-column singletons, doubleton equations, integer bound rounding) and the postsolve stack that reconstructs the primal and the DUAL of the original model | core, la | 1.8k |
| `src/simplex/` | `simplex_core.hpp` — the state the two simplex loops share (basis, factors, pricing weights, perturbation, warm start); `primal_simplex.cpp` — bounded-variable revised primal simplex, composite phase 1, Devex pricing, textbook and Harris ratio tests, bound perturbation, basis repair; `dual_simplex.cpp` — bounded dual simplex, bound-flipping ratio test, dual Devex, artificial bounds, cost perturbation, hand-over to the primal loop; `dense_lu` — a dense reference used by tests | core, la | 3.2k |
| `src/pdhg/` | restarted PDHG (PDLP-style), CPU; the GPU backend hangs off this path (`src/gpu/`, behind `SANKHYA_ENABLE_CUDA`, PR #153) | core, la | 0.7k |
| `src/ipm/` | Mehrotra predictor-corrector interior-point method on the normal equations, over the sparse LDLᵀ in `src/la/ldl.cpp`; no basis | core, la | 0.5k |
| `src/qp/` | convexity check (Cholesky of the Hessian), Condat–Vũ first-order convex QP | core, la | 0.5k |
| `src/mip/` | branch and bound: propagation, root diving, reliability branching with strong branching, warm-started dual node LPs, MIQP nodes through the QP engine; root cuts are PR #159 | core, simplex, qp | 1.3k |
| `src/api/` | the C API over `solve()`; the Python bindings (`bindings/python/`) wrap this, not the C++ | core | 0.5k |
| `apps/sankhya-cli/` | `sankhya solve|info|options|version`, `--stats`, `--write-sol`, `--option k=v` | api, io | – |
| `tools/` | `verify_solution.py`: re-parses the model with its own reader and checks the `.sol` file's primal feasibility, reduced costs, dual feasibility, complementary slackness and strong duality. Shares no code with the solver, deliberately | – | 1.5k |
| `tests/` | unit tests per module; `oracles/` — a rational-arithmetic simplex and exact MILP branch and bound that the float engines are fuzzed against; `robustness/` — the sweeps that find where the solver stops working | – | 10k |
| `bench/runners/` | Netlib, MIPLIB and Mittelmann runners and fetchers, the HiGHS comparison (a separate process over the same files), the robustness sweep, `make_benchmarks_doc.py` which generates `docs/BENCHMARKS.md` from the CSVs | – | 4k |

The dependency direction is strictly downward in that table: `la` knows nothing about
models, `simplex` knows nothing about integrality, `mip` knows nothing about file formats.
No file in `src/` reads or links anything from another optimization solver; `docs/PROVENANCE.md`
records the dependency table, the link line and the algorithm citations.

## 3. The two invariants everything else rests on

**The frozen interface.** `Model`, `Solution` and `solve()` do not change without an explicit
note in the PR that changes them. Readers produce a `Model`; every engine consumes one and
produces a `Solution`; the writers, the verifier, the C API and the bindings consume a
`Solution`. A new engine touches `src/<engine>/` and one branch of the dispatcher.

**Nothing is reported that was not measured.** `recompute_quality()` recomputes the row
activities, the objective, the reduced costs and the KKT residuals against the ORIGINAL
model after postsolve, and the status guard turns an engine's "optimal" into "feasible" when
those numbers disagree with the claim. Outside the process, `tools/verify_solution.py`
repeats the audit with its own reader, and the exact oracle in `tests/oracles/` is the
standard the float engines are compared against on instances nobody chose. A wrong answer
in numerical code prints and looks correct; these three layers are how it gets caught.

## 4. How a solve flows

1. **Read.** `src/io` produces a `Model` with column-major storage, bounds, integrality and
   an optional lower-triangular Hessian. Coefficients below `kZeroDrop` are dropped here and
   nowhere later, and `docs/BENCHMARKS.md` §5 records what that costs.
2. **Classify.** Integrality → branch and bound; a Hessian → convex QP (or MIQP nodes);
   otherwise an LP engine chosen by `algorithm`: `auto` is the dual simplex.
3. **Presolve** (LP path). Reductions are recorded on a stack. The reduced model is scaled
   (Ruiz then Pock–Chambolle) inside the simplex entry point; the scaled and unscaled
   attempts share one time budget.
4. **Solve.** The dual simplex starts from the slack basis (or a warm start), boxes any
   column that is dual infeasible, runs the bound-flipping ratio test with dual Devex
   pricing, perturbs costs on a degenerate stall, and hands the basis to the primal loop
   whenever it cannot finish honestly (artificial bound active, pivot disagreement on fresh
   factors, marginal infeasibility). Optimality is declared only on fresh factors.
5. **Postsolve.** Primal values are reconstructed in reverse record order; the duals are
   reconstructed to a fixed point, then every reduced cost of a column presolve could have
   touched is recomputed as `c − Aᵀy`.
6. **Audit and report.** `recompute_quality()`, the status guard, then the `.sol` and JSON
   writers. The `.sol` file carries 17 significant digits so the verifier sees what the
   solver saw.

For a MILP, step 4 becomes the tree: propagation at each node, a root dive for an incumbent,
reliability branching (pseudocosts once observed enough times, strong branching before that),
and every child solved by the dual simplex from its parent's basis, which is dual feasible
there by construction.

## 5. Where the next engines plug in

- **Interior-point method** (#56) — built: `src/ipm/` is one branch of the LP dispatcher
  over the sparse LDLᵀ in `src/la/ldl.cpp` (#70). It produces a `Solution` without a basis,
  which the status guard and the verifier handle as they do for PDHG. It does not serve
  branch and bound: that would need a crossover to a basis, and the dual simplex stays the
  node engine.
- **MIQP** — already present: the branch-and-bound node relaxation is a QP when the model
  has a Hessian, and the node bound is quadratic (`src/mip/branch_and_bound.cpp`).
- **NLP / MINLP** — the frozen interface is the constraint: a `Model` today is linear
  constraints with an optional quadratic objective. A nonlinear engine would extend `Model`
  with constraint functions and gradients (an explicit interface change, per the rule in
  §3) and plug in at the same dispatcher seam; the MILP tree needs no change to search over
  it, since it only reads `col_value` and the bound. Nothing of this exists yet, and
  `docs/PS26119_COVERAGE.md` says so.
- **Cutting planes** — root GMI and lifted cover cuts are PR #159 (`src/mip/cuts.cpp`),
  appended as rows of the working model before the search starts.
- **Parallelism** — the column loops in pricing and in the sparse products are
  embarrassingly parallel and deterministic (no cross-thread reductions); the tree search is
  the larger prize and the harder one, because a race on the incumbent can fathom a node
  that should have been explored (#57).

## 6. Toolchain, as tested

- C++20, CMake ≥ 3.20, Ninja. CI is Ubuntu (GCC), Release and Debug + ASan/UBSan; the
  Windows development boxes use MSYS2 UCRT64 GCC 16.1.0 through `scripts/configure.sh`,
  which refuses a compiler older than GCC 10.
- Dependencies (all non-solver, table in `docs/PROVENANCE.md`): fmt, CLI11, nlohmann/json,
  GoogleTest (tests only), zlib (optional), pybind11 (bindings). No BLAS, no LAPACK.
- `clang-format` 22.1.8 from pip, pinned, because its output differs between majors.
- Python 3.10+ for the runners, `highspy` optional for the comparison.
- `scripts/reproduce.sh` is the one command from a fresh clone: configure, build, test,
  fetch, benchmark, robustness sweep, comparison, and regenerate `docs/BENCHMARKS.md`. It
  says which steps need the network and skips them, named, when it is absent.

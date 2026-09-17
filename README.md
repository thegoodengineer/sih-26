**SIH26119** — Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to CPLEX / Xpress)  
Smart India Hackathon 2026 · Mangalore Refinery and Petrochemicals Limited (MRPL)

# SANKHYA

**Indigenous optimization solver — LP, MILP, convex QP, written from mathematical
foundations. Its first-order engine is built for the GPU; the GPU backend itself is not on
`main` yet (see below).**

*The first line above is the problem statement's title, quoted as issued. The line under
the name is ours, and says only what is on `main`.*

Smart India Hackathon 2026, problem statement **SIH26119**, issued by Mangalore Refinery and
Petrochemicals Limited.

Project site: [sankhya-solver.vercel.app](https://sankhya-solver.vercel.app) — the problem
statement, what is different, the evidence, how to use it, and the team, on one page.

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
| 4 | Restarted PDHG — **CPU done**, its answer finished by the interior point by default since #229; CUDA backend not on `main`: it is PR #274, open, being carved into #16 and #17 on the team's two GPU laptops, and not yet built or measured on either | partial |
| 5 | Branch & bound → MILP | **done** (MIPLIB benchmarked; root cuts landed in #159 and are off by default, see below) |
| 6–10 | Performance, branch & cut, IPM/QP, robustness, packaging | convex QP **done** (Phase 8, `src/qp/`); interior point **done, opt-in** (`src/ipm/`, no basis); robustness sweep **done** (`bench/runners/robustness.py`); root cuts **done, off by default** (#159, `src/mip/cuts.cpp`); machine-checkable certificates for infeasible and unbounded models **done** (#192, `src/core/certificate.cpp`, checked by `tools/verify_solution.py`); MIP gap targets **done** (#188); packaging **done** apart from the human items on #73 |

LP is solved by a bounded-variable revised primal simplex (or restarted PDHG), MILP by
branch and bound, and convex QP by a Condat-Vu primal-dual method — and MIQP by branch and
bound over QP relaxations — all end to end from an MPS file through to an independently
verified answer. **Non-convex** QP is the one class still refused, and refused deliberately
rather than approximated: it is decided by an LDL^T semidefiniteness test before any
arithmetic starts, and a negative pivot is returned as the certificate. Reporting a local
optimum as a global one is the single most damaging thing this dispatcher could do, so it
does not — see the Evidence rules in [`ENGINEERING_RULES.md`](ENGINEERING_RULES.md).

Benchmark results against Netlib, headline first: **78 of 89** on the full set — matched
to the published optimum to a relative 1e-6 *and* passed independent verification —
measured on `main` at `2b4eb6b` (`bench/results/netlib-full-2b4eb6b.csv`). The narrower
tiers read higher (**48 of 50** on the medium tier, **9 of 9** on the small set the demo
runs) because both are defined by a row cap, which makes them the easier half by
construction; the full set is the number Phase 6's ">= 95% of Netlib" criterion is
measured against, so it is the one quoted here. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), generated from the CSVs in `bench/results/` so
it cannot drift.

The 11 non-passes are worth naming, and most of them are not wrong answers. Every one that
produces an answer was cross-checked against **HiGHS**, a mature third-party solver run as a
separate process, by `bench/runners/cross_check_highs.py`
(`bench/results/cross-check-highs-6bec31e.csv`):

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
limit depends on how fast the machine is that minute. It fitted at `6bec31e` (55.9 s) and
did not at `2b4eb6b` (the run above), which is the whole of the difference between 79 of
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
13 of 23 failures at `7b6b5c2` (`bench/results/netlib-full-7b6b5c2.csv`, 2026-08-31) and
has been **zero on the full Netlib set** since #144 found the pivot search treating "none
of my first four candidates was admissible" as proof of singularity and #147 repaired the
genuine rank defects that remained. It is not zero everywhere: on Mittelmann's `qap15` the
unscaled retry went singular at iteration 13,954 (`bench/results/mittelmann-64d1a6d.csv`,
issue #174), the first reappearance in the evidence and on the Mittelmann instance closest
to Netlib's size. Since then the dual simplex became the automatic engine (#165),
reliability branching landed (#166), presolve's postsolve runs its dual passes to a fixed
point (#162), and the FTRAN went hyper-sparse (#169): between them the full set went from
71 to 78 verified passes, and `degen3`, which took 123.6 s, takes 0.9 s
(`bench/results/netlib-full-2b4eb6b.csv`).

That is a better class of problem to have, and a different roadmap: speed on the three
largest instances rather than robustness. Tracked in #214 (the four non-passes that are
ours to fix) and #210 (the dual simplex's iteration rate at size).

MIPLIB 2017 is benchmarked too: **13 of 30** easy instances reach the published optimum,
**9 of 30** also prove it (`bench/results/miplib-bf3df02.csv`, 60 s; it was 6 of 30 at
`2b4eb6b`, before #188 let a search that meets its gap target say `optimal` - three of the
nine are that renamed status, not a better search) - branch and bound has reliability
branching and warm-started node LPs, and root cutting planes that are off by default: on the
same commit they prove the same nine, take the node count to 0.918x, and cost one published
match (`noswot`'s incumbent worsens from -41 to -39 with them on; `bench/results/miplib-
cuts-{off,on}.csv`, `docs/BENCHMARKS.md` section 2), so it finds good incumbents far more
often than it closes the bound. For
scale beyond what Netlib tests, `bench/runners/generate_large_lp.py` builds sparse LPs of
any size with an exactly known analytic optimum, and `bench/runners/mittelmann.py` runs
Mittelmann's LP set: on its eight smallest instances (6,330 to 376,500 rows) the result is
**0 of 8** inside 300 s - eight time limits, now that `qap15` no longer ends in a singular
basis (#174, fixed in #178) - every one named in section 1d of `docs/BENCHMARKS.md`
(`bench/results/mittelmann-e134aeb.csv`, run on `main`, on AC, alone on the machine). The
overrun on `bdry2`, whose time limit could not reach inside the sparse LU (#208), is gone:
303 s against 300, where it was 380 and before that 648.

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
re-cloning 200 MB. `ctest --test-dir build` runs the whole suite and prints its own count;
the count is not typed here, because a typed count goes stale the day a test is added.

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

A verdict of *infeasible* or *unbounded* is not taken on trust either. Since #192 the
solution file carries a **certificate** with it - a Farkas vector for an infeasible model, a
ray for an unbounded one - and `tools/verify_solution.py`, which shares no code with the
solver, checks the certificate against the original model the same way it checks an
optimal point. The header says `certificate farkas`, `ray` or `none`, so a verdict that
arrives without one (a model presolve proved infeasible, say) is visible as such.

A MILP stopped by a gap target (`--option mip_relative_gap=0.01`) reports *optimal within
the gap* and the file records the gap it was asked for (#188); a run that stops on a limit
reports the incumbent as what it is. Time limits reach inside the sparse LDL^T factorization
(#197), so an interior-point solve on a model whose factor is too large stops on time
rather than when the factorization happens to finish; the sparse LU used by the simplex
does not yet honour it the same way (#208).

A MILP also reports a **solution pool** (#225): up to `pool_size` (default 10) integer plans
the search found, each a different integer assignment, the solution first and the rest best
first, written to the `.sol` file's `pool` section (integer columns only) and checked by
`tools/verify_solution.py`. By default it is what the ordinary search happened to find, and
the search is unchanged; `--option pool_complete=true` keeps searching until the pool
provably holds the best plans, at the cost of nodes, and `pool_diversity` / `pool_gap` shape
which ones. The demo's production-planning scene prints the top three.

`--progress-out` appends one JSON line per logged iteration or node to a file as the solve
runs, flushed immediately - an operator can `tail -f` it during a long solve to watch the
bound close in on the answer without waiting for the final report.

The solver supports graceful interruption. Pressing Ctrl-C (`SIGINT`) in the CLI stops the solve at the next safe point and returns the best found point as an `interrupted` result. The Python bindings expose `Model.interrupt()` and support progress callbacks; returning a non-zero value from a Python callback (or via `sankhya_set_callback` in C) interrupts the solve gracefully.

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
src/core          Model/Solution implementation, the solve() dispatcher, certificates (#192)
src/util          logging, timers, arena allocator, option registry
src/io            MPS + LP readers (including QPS QUADOBJ), solution and JSON writers
src/presolve      reductions + postsolve               (on by default)
src/simplex       primal and dual revised simplex (the dual is the branch-and-bound node engine)
src/la            sparse containers, sparse Markowitz LU (hyper-sparse FTRAN), sparse LDL^T, dense LU (test oracle only)
src/pdhg          restarted PDHG, CPU; its answer is finished by the interior point (#229)
                                                       (CUDA backend: not on main, PR #274)
src/mip           branch and bound + diving heuristic + root cuts (cuts off by default, #159)
src/qp            convex QP, Condat-Vu primal-dual     (done)
src/ipm           Mehrotra interior point, sparse LDL^T (opt-in: algorithm=ipm, no basis)
bindings/python   Python bindings — ctypes over the C API, nothing to compile
tests/  bench/  tools/  docs/  demo/
```

## What this does NOT do

Stated here rather than only in the demo, because a solver that is vague about its limits is
not one an industrial user can plan around. [Issue #54](https://github.com/thegoodengineers/SANKHYA/issues/54)
tracks every PS26119 requirement against what exists on `main`; section 6 of
`demo/run_sih_demo.sh` prints this list at the end of every run.

| not implemented | note |
|---|---|
| **GPU acceleration** | The first-order method it needs exists and runs on CPU - restarted PDHG, `--option algorithm=pdhg`, **9 of 9** committed instances to `optimal` at 1e-8 on `main` at `4177ae6` (`bench/results/pdhg-4177ae6.csv`, `docs/BENCHMARKS.md` section 1e). The ninth is `share2b`, which the first-order method alone leaves at its million-iteration limit: since #229 its answer is finished by the interior point by default (`pdhg_polish`), nine more iterations there, and section 1f.1 measures the same thing size by size - 6.1e-04 to 1.9e-10 at 1,000 rows, declined where the factor is not affordable. The CUDA backend is PR #274, open and not yet built on a GPU (#16-#19, assigned to the two laptops that have one); `--gpu` warns and falls back. No speed-up is claimed. |
| **Scale** | Measured to 1,000,000 rows on generated instances only - no industrial model at that size - and the answer depends on the engine (#198, `docs/BENCHMARKS.md` sections 1f and 1f.1). On generated instances whose optimum is exact by construction: under a 120 s clock the first-order engine reaches the optimum at **100,000 x 100,000** to 1.1e-07 without certifying it and certifies it at 5,000 in 54 s, the interior point reaches it at 5,000 since the AMD ordering (#193) and the dual simplex only at 1,000 (`bench/results/scale-e134aeb.csv`, 7 of 12, measured on `main` after #242, #265 and #278). Given a fixed 1,000 iterations instead of a clock - the one accuracy claim here another machine reproduces exactly - the error stays between 6.9e-06 and 6.1e-04 all the way to **1,000,000 x 1,000,000**, so what grows with the model is the cost per iteration, not the number needed. Those are the random family's numbers, and a random sparse matrix is the worst case for anything that factorizes; on a second family shaped like a multi-period planning model (#198, section 1f.2, `bench/results/scale-staircase-e134aeb.csv`) the interior point reaches the optimum at **20,000** rows instead of 1,000 and 8 of 12 solves reach it against 7 of 12. The polish that finishes PDHG's answer (#229) is measured beside the unpolished run in section 1f.1 (`bench/results/scale-iterations-bf3df02.csv`): at 1,000 rows the same 1,000 iterations land at 6.1e-04 unpolished and 1.9e-10 polished, and at every larger size of the random family the polish declines within its 30 s budget because the factor is dense, which the table shows rather than hides. On real models: the largest Netlib instance solved is `fit2d`, 25x10500 with 129018 nonzeros, in 0.3 s, and on Mittelmann's eight smallest LPs, 6,330 to 376,500 rows, the result is 0 of 8 inside 300 s. On an industrial-structured model - a T-period refinery planning LP with crudes, tanks, yields, capacities, quality budgets and delivery commitments, optimum exact by construction (`bench/runners/generate_refinery_lp.py`, #211, section 1f.3, `bench/results/scale-refinery-e134aeb.csv`) - the monthly year (1,068 rows) is solved exactly by all three engines, the daily year (**32,485 rows**) exactly by the interior point (45 iterations, 68 s) and by PDHG finished by it (92 s), and the hourly year (**779,640 rows**) is reached by PDHG to 1.0e-06 inside 120 s and proved by nothing: 6 of 9 solves. That is the first industrial-shaped model at this size the evidence holds, and the number to read is the daily year. |
| **Interior point as a default** | An interior-point method exists (#56, `--option algorithm=ipm`, Mehrotra predictor-corrector over a from-scratch sparse LDL^T) and is opt-in: it produces no basis, so it cannot warm-start branch and bound and cannot certify infeasibility, and on the full Netlib set it verifies fewer instances than the dual simplex (`docs/PS26119_COVERAGE.md`). The default continuous engine is the simplex. It does two things by default now that it did not: it stops the moment its iterate is not a number and returns the best finite one (#205, found on the 20,000-row staircase model), and it finishes PDHG's answers from PDHG's own point (#229), declining within `polish_max_seconds` when the factor is not affordable. |
| **Cutting planes by default** | Root Gomory mixed-integer and lifted knapsack cover cuts exist (#159, `--option enable_root_cuts=true`) and are off by default: on the 30-instance MIPLIB set at 60 s on `main` at `bf3df02` they prove the same 9 instances, take the node count to 0.918x, and cost one published match (`noswot`, whose incumbent is -39 with them and -41 without), because a cut row makes every node LP dearer (`bench/results/miplib-cuts-{off,on}.csv`; `docs/BENCHMARKS.md` section 2). No MIR cuts, and none below the root (#221). Branch and bound itself has reliability branching (#69) and warm-started dual node LPs (#65). This is why MIPLIB proves few optima. |
| **Non-convex QP** | Refused deliberately, with an LDL^T certificate. A local optimum reported as a global one is not something this solver will do. |
| **MIQP bound quality** | MIQP is implemented, but its node bound comes from a first-order method and is only accurate to the tolerance it converged to, so pruning is deliberately kept on the conservative side and costs nodes. With root cuts off by default too, expect incumbents more often than proofs. |
| **Parallelism** | Single-threaded by default. `--option threads=N` runs the column loops of an iteration under OpenMP, deterministically - results are bit-identical at 1 and 8 threads - and at Netlib scale it is measured to buy nothing, because an iteration is too short to amortize the fork (#57). It is a correctness-preserving switch, not a speed claim. |

On speed against HiGHS: on the medium tier the objectives agree on all 50 instances, and
the speed ratio is, for the first time, a number two runs an hour apart agree on. Three
runs of the same binary on `main` at `bf3df02` (`bench/results/compare-highs-medium-
bf3df02.csv`, `-second.csv`, `-third.csv`; 16:17, 16:41 and 17:21 on the same afternoon,
alone on the machine, on mains) put the median per-instance ratio at **2.01x, 2.04x and
2.12x** - SANKHYA slower - with total solve time 2.64x, 2.56x and 2.62x. The first and the
third are an hour apart and agree within 6%, which is the bar #212 set for quoting it;
the earlier committed pair, three days apart, had read 1.38x and 2.11x, and that spread is
why the number was not quoted before. Read it as *about twice HiGHS's time on these
instances, on this laptop*: 9 to 16 of the 50 timing envelopes still overlap outright,
the instances solve in single-digit milliseconds, and the ratio belongs to this machine
state as much as to the solver.
The reproducible comparison is iteration count, where the gap narrowed by a third when
devex pricing became the default (#66); HiGHS's devex still takes fewer.

## What still needs doing

The list above is what the solver does not do; this is the work that would change it, as it
stands on `main` at the time of writing, each item with the issue that carries its
evidence and its acceptance criteria. Three kinds: things to **create** that do not exist,
things to **improve** that exist and are measured short, and work to **finish** that is
started. Nothing here is a claim; every row points at where the claim would have to be
earned.

### Create

| what | issue | why it matters, and what it takes |
|---|---|---|
| **CUDA backend for PDHG** | [#16](https://github.com/thegoodengineers/SANKHYA/issues/16), [#17](https://github.com/thegoodengineers/SANKHYA/issues/17), [#19](https://github.com/thegoodengineers/SANKHYA/issues/19); PR [#274](https://github.com/thegoodengineers/SANKHYA/pull/274) | The "GPU-accelerated" in the problem statement's title. The CPU engine is shaped for it (two sparse matrix-vector products and componentwise projections per iteration, no serial dependency); the build plumbing, the kernels and the CPU-vs-GPU crossover evidence are unwritten. Blocked on a machine with the card: the PR exists and cannot be built or measured here. What to do the day the hardware arrives - intake, the order of the three PRs, the test list, the measurement protocol, and what the result will and will not let us claim - is written down in [`docs/GPU_PLAN.md`](docs/GPU_PLAN.md). |
| **Crossover** | [#219](https://github.com/thegoodengineers/SANKHYA/issues/219) | The interior point is the engine that solves the structured scale models exactly (the 5,000-row staircase, the 32,485-row refinery year), and its answer has no basis, so it cannot certify infeasibility, warm-start branch and bound, or feed sensitivity ranging (#220's ranges need a basis). Push its point to a vertex with a few simplex pivots. Also what would let #209's "feasible" become "optimal" on the 20,000-row staircase model. |
| **Warm start and in-place modification** | [#218](https://github.com/thegoodengineers/SANKHYA/issues/218) | A planner re-solves twenty times before lunch; today every solve is cold from a file. The dual simplex already re-solves from a basis in a handful of pivots inside branch and bound (#65); the API does not expose it. |
| **Branch-and-cut proper** | [#221](https://github.com/thegoodengineers/SANKHYA/issues/221) | Mixed-integer rounding cuts, and cuts below the root. Root Gomory and cover cuts exist and are off by default because they cost a proof on MIPLIB; MIR cuts from original rows are the standard remedy for the loose root bounds that keep the MIPLIB proof count where it is. |
| **Parallel tree search** | [#222](https://github.com/thegoodengineers/SANKHYA/issues/222) | Branch and bound uses one of eight cores. Node LPs are independent; only the incumbent, the bound and the queue are shared. The most mechanical speedup left. |
| **Convex NLP behind the `solve()` seam** | [#226](https://github.com/thegoodengineers/SANKHYA/issues/226) | The one PS26119 line still marked partial for a reason: the seam dispatches four classes cleanly and no nonlinear engine sits behind it. Convex, separable objective by callbacks; not general nonconvex, not MINLP. |
| **Certificates and gap targets in the C API and Python** | [#207](https://github.com/thegoodengineers/SANKHYA/issues/207) | The Farkas vector, the ray, the IIS and the gap tolerances are all in the `.sol` file and checked by the verifier, and none is reachable from C or Python. Accessors only. |
)

### Improve

| what | issue | where it stands |
|---|---|---|
| **The dual simplex's cost per iteration at size** | [#210](https://github.com/thegoodengineers/SANKHYA/issues/210), [#243](https://github.com/thegoodengineers/SANKHYA/issues/243) | #242 removed two O(m)-per-step sweeps from the factorization and the per-iteration recomputation of the basic values and duals; the per-phase clock it added (verbose log) now puts the pivot row - a BTRAN of a unit vector plus a gather over every column - at a quarter to a third of an iteration at 20,000 rows. Both standard remedies are in: the gather runs over rho's support through a row-wise copy of A (#265) and the transposed solve applies L^T in push form (#278), and the second measured to no change, because counters put a transposed solve's cost in the eta file (20,000 to 27,000 entries read per solve against a few hundred pushes through the factors) and in the four full-length passes over m; the next lever is a Forrest-Tomlin update ([#279](https://github.com/thegoodengineers/SANKHYA/issues/279)). None of the four 5,000- and 20,000-row scale models reaches the optimum in 120 s yet, but the iteration count inside those 120 s is up on every one: on `main` at `e134aeb` (`bench/results/scale-e134aeb.csv`, `scale-staircase-e134aeb.csv`) the dual simplex does 33,019 / 42,448 iterations at 5,000 rows (random / staircase) and 34,163 / 40,108 at 20,000, against 24,316 / 13,761 and 28,564 / 10,144 on the last runs before this work (`scale-f545f83.csv`, `scale-staircase-bf3df02.csv`). |
| **The unscaled retry on badly scaled generated models** | [#244](https://github.com/thegoodengineers/SANKHYA/issues/244) | When the scaled dual simplex times out, the unscaled retry repairs singular bases and hands over to the primal on a fresh-factor pivot disagreement. The dual ratio test accepts any pivot above an absolute 1e-9; a relative floor with a Harris pass is what production codes do. |
| **Netlib 78 of 89** | [#214](https://github.com/thegoodengineers/SANKHYA/issues/214) | `maros-r7` is documented rather than solved (#247: the basis the dual hands over decays under the primal, and the run now says so as a numerical error with the numbers instead of running out the clock); `pilot87` and `dfl001` are time limits that #210's speed work is the lever for; `pilot` needs a written numerical argument on its dual residual, or iterative refinement. The full-set re-run on `main` is not done. |
| **Mittelmann 0 of 8** | [#216](https://github.com/thegoodengineers/SANKHYA/issues/216) | The only benchmark line where nothing finishes, re-measured on `main` at `e134aeb` with the same result (`bench/results/mittelmann-e134aeb.csv`; every row a named time limit, `bdry2` now stopped at 303 s). `brazil3` and `qap15` are the two within reach; the rest are size. |
| **MIPLIB: 13 of 30 reach the optimum, 9 prove it** | [#215](https://github.com/thegoodengineers/SANKHYA/issues/215) | The weakest number in the project, and the issue says where each of the other instances stands. The levers are #221 and #222. |
| **Interior point on the largest random model** | [#246](https://github.com/thegoodengineers/SANKHYA/issues/246) | On the 100,000-row random scale model it dies of `std::bad_alloc` 170 s past its 120 s limit with no status and no stats file, on an 8 GB machine: an out-of-memory condition must come back as a status, and the ordering needs a memory budget the way the polish has a factor budget. |
| **Interior point: stop before the barrier breaks the factorization** | [#209](https://github.com/thegoodengineers/SANKHYA/issues/209) | #241 reads the regularization spike as convergence and stops one factorization earlier; the answer is `feasible`, not `optimal`, because its dual side misses the tolerance by 12%, which crossover (#219) is the honest way to close. |

### Finish

| what | where it stands |
|---|---|
| **The final read** ([#213](https://github.com/thegoodengineers/SANKHYA/issues/213)) | A person reading the submission cold with the problem statement open, before 20 September. Not a code task. Two automated audits found and fixed stale and contradictory claims; a third pair of eyes is the point. |
| **The demo recording** ([#73](https://github.com/thegoodengineers/SANKHYA/issues/73)) | The one open box of the packaging issue, for the PS metadata's YouTube field. `demo/run_sih_demo.sh --quick` is what to record. |

## Licence

Apache-2.0.

# PS26119 coverage tracker

One place to see what PS26119 asks for against what exists on `main`, cross-checked against
`git log` and the source directly rather than asserted. This file mirrors and supersedes the
tracker previously kept only in issue #54's body — a repo file is diffable and reviewable the
way an issue body is not, and every status here was checked against a specific commit.

**Refreshed 2026-09-10 at `0048da8`; every number cites the CSV it comes from.**

## Problem classes

| requirement | status | evidence / issue |
|---|---|---|
| Linear Programming | **done** | revised primal simplex; Netlib full set 78/89 (`bench/results/netlib-full-53cbe16.csv`), medium 48/50, small 9/9 (`bench/results/netlib-{medium,small}-adcee1b.csv`; #34). Cross-checked against HiGHS (#154): on every instance where the solver produces a final answer, that answer agrees with HiGHS to 3.1e-10 or better; 8 of the 18 failures are cases where the published Netlib table is the outlier, not our answer — the full set is the headline per #142, the narrower tiers being row-capped and so the easier half |
| Mixed-Integer LP | **done** | branch & bound; corroborated by exhaustive oracle in the demo |
| Quadratic Programming | **done** | #55 — Condat-Vu primal-dual engine, `src/qp/`. Convexity decided by LDL^T on `sense * Q` before any arithmetic; non-convex is refused with a certificate, never solved to a local point. Readable from a QPS `QUADOBJ` file (#112) and independently verified. |
| Mixed-Integer QP | **done** | #139 — branch and bound over convex QP node relaxations, joining the two existing engines. `src/core/solve.cpp` dispatches a real `ProblemClass::kMiqp` case; a non-convex Hessian is still refused before any arithmetic, exactly as plain QP does. `tests/unit/test_miqp.cpp` covers it. (Superseded: this used to return `not_solved`; #142 corrected the demo and docs to stop saying so.) |
| Modular for NLP / MINLP | partial | `solve()` seam exists and dispatches four classes cleanly (LP, MILP, QP, MIQP); NLP/MINLP themselves are not attempted |

## Algorithms the PS names

| requirement | status | evidence / issue |
|---|---|---|
| Revised simplex | **done** | `src/simplex/primal_simplex.cpp`, bounded-variable, composite phase 1, no big-M |
| First-order methods (PDHG) | **done** | `src/pdhg/`, restarted PDHG on the CPU, opt-in as `--option algorithm=pdhg`. On the nine committed instances **8 of 9** reach `optimal` at both 1e-4 and 1e-8 with restarts on, 7 of 9 with restarts off (`bench/results/pdhg-fix-367e0a2.csv`, `docs/BENCHMARKS.md` section 1e). That was 3 of 9 before #179 fixed two things the CUDA branch had quietly fixed and described as preservation: the step size collapsed to its floor on iteration zero, and the loop stopped on a point the report then had to downgrade. A loose tolerance request is ignored by default; `pdhg_stop_at_request=true` waives the dual, gap and complementarity halves of the standard (absolute primal feasibility is kept, so `feasible` still means feasible) and reports the point as `feasible` unless it meets the full standard anyway: 0.85x the iterations at 1e-4 on these instances, and `share2b` becomes a usable point instead of an iteration limit (`bench/results/pdhg-stop-at-request-f18d4b0.csv`; #180). |
| Interior-point methods | **done, opt-in** | #56 via #169: Mehrotra predictor-corrector on the bounded form, normal equations through a from-scratch sparse LDLᵀ (`src/ipm/ipm.cpp`, `src/la/ldl.cpp`), `--option algorithm=ipm`. Produces no basis, so it is not the node engine and not the default; on the full Netlib set it verified 47/89 (`bench/results/netlib-full-59ac6e3-ipm.csv`, 52 optimal, two rejected by the verifier in original units after postsolve: `pilot4`, `scagr25`), against 79/89 for the dual simplex at the same commit. |
| Branch-and-bound | **done** | `src/mip/branch_and_bound.cpp`, domain-change stacks, no per-node copy |
| Branch-and-cut / cutting planes | **done, off by default** | #159 (closes #23): root Gomory mixed-integer and lifted knapsack cover cuts, `src/mip/cuts.cpp`, validity gated in exact arithmetic against the rational oracle (`tests/unit/test_cuts.cpp`) and by the 600-instance MILP fuzz run with cuts on. Off by default by measurement: the A/B at 60 s on the 30 MIPLIB instances took the node count to 0.887x over the 28 instances that end the same way and cost two proofs (`bench/results/miplib-cuts-{off,on}.csv`; `docs/BENCHMARKS.md` section 2). No MIR cuts, none below the root. |
| Presolve | **done** | #43 landed six reductions (empty row, redundant row, fixed column, empty column, singleton row, forcing row); #92, merged as `e25da08`, added the two it deferred, free-column-singleton and doubleton-equation elimination, with their dual reconstruction corrected in `3b69f00`. On by default, and postsolve re-measures the recovered point against the original model, so a wrong dual from a bad postsolve surfaces as a feasibility violation rather than as a quiet wrong answer - which is exactly the failure class this project is built to avoid. |
| Heuristics | **done** | #25 — diving to an incumbent, then best-bound |
| Advanced node selection | **done** | #69 via #166: reliability branching - pseudocosts, strong branching on the ten best candidates until each has eight observations, product score - on top of the dive-then-best-bound node order; every node LP is a warm start of the bounded dual simplex (#65 via #165). |

## Implementation requirements

| requirement | status | evidence / issue |
|---|---|---|
| Sparse matrix techniques | **done** | CSC/CSR, sparse Markowitz LU with threshold stability |
| Efficient numerical linear algebra | **done** | #49 scaling (Ruiz + Pock-Chambolle, default on), #50 basis update (product-form, with an FTRAN-residual accuracy check that forces refactorization). #144 fixed `eliminate()` reporting the wrong singular column when an earlier column is merely unpivotable within the search budget, #147 repairs the genuine rank defects. #68 via #169: hyper-sparse FTRAN back-substitution through U stored by column, with the gather form kept as the tested reference (`tests/unit/test_sparse_lu.cpp`). #72 via #169: iterative refinement of the final basis, primal and dual, with a compensated residual (`refinement_steps`, `residual_before_refinement`, `residual_after_refinement` on the `Solution`). #169 also adds a sparse LDLᵀ (`src/la/ldl.cpp`: minimum degree, etree, up-looking numeric, regularized pivot floor) for the interior-point method. |
| Pricing | **done** | Devex (#66) is the default. It landed in #126 as opt-in because it turned `grow22` and `scsd8` into singular bases; that failure class was removed by #144 and #147, and re-measured on the Netlib medium tier devex solves the same 49 instances in a third fewer iterations (34580 vs 51968) and a third less time, with no status change on any instance. Dantzig remains selectable via `--option pricing=dantzig` so the comparison can be regenerated. The Harris two-pass ratio test (#67) is selectable via `--option ratio_test=harris` and stays opt-in: re-measured alongside, it changes no status and costs time under Dantzig (274.8 s vs 191.0 s) while doing nothing for devex. |
| Multi-core parallelization | **done, honestly small** | #57 via #169: OpenMP over the column loops that dominate an iteration (pricing, pivot row), deterministic by construction - results are bit-identical at `threads=1` and `threads=8` on the seven of the eight largest Netlib instances that finish (`bench/results/netlib-0cd08cf-threads1.csv` / `-threads8.csv`); the eighth, `dfl001`, hits the time limit in both and got FEWER iterations with eight threads (31,256 against 32,739) - measured with NO speedup at Netlib scale, where an iteration is too short to amortize a fork. Off unless `--option threads=N`; the build flag is `SANKHYA_WITH_OPENMP`. |
| GPU acceleration | **not started** | #16-#19. The engine it needs — restarted PDHG — exists, is verified, and solves the 5000x5000 instance in the demo on CPU. `--gpu` warns and falls back. |
| **Not built on any existing solver** | **done** | `docs/PROVENANCE.md`, CI-enforced, live link list in demo section 1 |

## Industrial scope

| domain | status |
|---|---|
| crude blending | **done** — `demo/crude_blend.mps` (LP) and `demo/crude_blend_qp.mps` (QP, price impact) |
| refinery scheduling | **done** — `demo/blend_milp.mps` |
| power system dispatch | **done** — `data/casestudies/power_dispatch.mps`, unit commitment, settled by exhaustive oracle |
| transportation / supply chain | **done** — `data/casestudies/supply_chain.mps` |
| production planning | **done** — `data/casestudies/lot_sizing.mps` |
| logistics | covered by transportation |
| process optimization | partial — the QP blend is the closest thing here |

## The benchmark bar

| requirement | status | note |
|---|---|---|
| Highly degenerate models | **done** | exact rank 6 of 7, computed over the rationals by the independent checker |
| Ill-conditioned matrices | **done** | 10^22 entry spread, condition number 2.14e+30, objective provably unmoved |
| Weak LP relaxations | **done** | 33.77% integrality gap closed |
| Difficult MILP formulations | **partial** | MIPLIB runs (below), but 6 of 30 proved is the honest reading |
| **Thousands to millions of variables** | **partial, and now measured** | #34 via `bench/runners/scale.py`: four sizes from 1,000 to 100,000 rows and columns, three engines, every answer checked against an optimum exact by construction (`bench/results/scale-eac6f75.csv`, `docs/BENCHMARKS.md` section 1f). **6 of 12** solves reached it. The first-order engine reached it at **100,000 x 100,000** to a relative 1.0e-07 in 121 s, without being able to certify it inside the limit; the dual simplex and the interior-point method reached it only at 1,000. Still partial, and honestly so: 100,000 is the low end of "millions", the instances are one generated shape, and nothing here is evidence about an industrial model of that size. |

## Expected solution

| requirement | status | note |
|---|---|---|
| Basic API **or** CLI | **done** | CLI satisfies the PS wording; C API (#58) and Python bindings (#59) both merged (#128, #129) on top of it |
| Netlib benchmark | **done** | measured on `main` at `53cbe16`: full set **78/89**, medium **48/50**, small **9/9**, every pass also verified by the independent checker (`bench/results/netlib-{full,medium,small}-adcee1b.csv`). The 11 full-set non-passes by reason: 7 where Netlib's published value is the outlier and our answer agrees with HiGHS (`80bau3b`, `e226` by its objective constant, `ganges`, `greenbea`, `greenbeb`, `nesm`, `scrs8`); 2 time limits at 120 s (`dfl001`, and `pilot87`, which reaches its optimum in 26,226 iterations whenever the machine gives it the time); `pilot`, whose answer agrees with HiGHS but which our own dual-feasibility check downgrades to feasible; `maros-r7`, where the solver declines to answer rather than claim. On this laptop the clock decides exactly one instance: `pilot87` reaches its optimum in 26,226 iterations every time under `--option iteration_limit=27000`, and whether that fits inside the 120 s limit varies with the machine - it did at `adcee1b` (55.9 s, 79/89) and did not in the run above. 86 of the 89 rows are identical in status, iteration count and objective between the two; the others are `dfl001` and `fit2p`, both clock-decided (#172). Answers and verdicts never move, only the clock. |
| MIPLIB benchmark | **done, weakly** | #24 — 30 instances at 60 s on `53cbe16`: 13 reach the published optimum, 6 prove it (that run predates #188, which made a gap-target stop report `optimal`; `f2gap40400`, `flugpl` and `p0201` stopped that way and a rerun would count them as proved - a renamed status, not a better search) (6 optimal, 21 feasible, 3 node limit; `bench/results/miplib-53cbe16.csv`, on AC throughout). Reliability branching (#69) moved the A/B at `63ec8de` by one instance each way (14 matched / 7 proved against 13 / 6 for most-fractional branching, `bench/results/miplib-{reliability,most-fractional}-63ec8de.csv`); root cuts (#159) landed and are off by default because at this limit they cost proofs (the row above; `docs/BENCHMARKS.md` section 2), so the proof rate now waits on a longer limit or cheaper node LPs. |
| Mittelmann benchmark | **done, and it fails** | #60 via #169: fetch with recorded provenance (`bench/runners/fetch_mittelmann.py`, Netlib-packed archives decoded with `emps`) and a runner that names every instance's outcome. On the eight smallest archives at 300 s on `main` (`bench/results/mittelmann-592aea3.csv`): **0 of 8** reach `optimal`; seven rows are named `time_limit`s and `qap15` ends in a singular basis at iteration 13,954 (#174) (`docs/BENCHMARKS.md` section 1d). HiGHS finishes two of the eight in the same limit on this machine state (four on the cooler run, `mittelmann-ca64dd5.csv`). This is where the solver stops today, stated rather than omitted. |
| Compared against an established solver | **done** | HiGHS as a separate process, objectives agree **50/50** on the medium tier (`bench/results/compare-highs-medium-adcee1b.csv` and `compare-highs-medium-53cbe16.csv`; the median of the per-instance solve-time ratios is NOT quotable: the two committed runs, three days apart on this machine, gave 1.38x and 2.11x, and on the second our total solve time was 0.68x the first's while the median read worse. 20 of 50 timing envelopes overlap outright on the second run, 34 of 50 on the first) and every full-set non-pass cross-checked by name (`bench/results/cross-check-highs-adcee1b.csv`). On speed: single-digit-millisecond instances put both solvers' timing envelopes on top of each other: the median ratio moved between 0.72x and 1.60x on an unchanged binary at every repeat count tried (measured in #120, recorded in `bench/runners/compare.py`'s own note, not re-run here), so "indistinguishable" is the honest reading, not a specific multiplier. The reproducible comparison is iterations: 2.16x behind on the small set (#66). |
| Robustness demonstration | **done** | `demo/run_sih_demo.sh` section 4, all three hazards the PS names |
| Transparent, extensible, sovereign foundation | **done** | provenance discipline, citations in code, exact oracles, an independent verifier sharing no code with the solver, `docs/ARCHITECTURE.md` for the module boundaries, `docs/sbom.spdx.json` (SPDX 2.3, generated from the pinned tags in `CMakeLists.txt` by `tools/make_sbom.py` and checked in CI), and the tested toolchains recorded in `docs/PROVENANCE.md` section 2b |

## Open PRs relevant to this table

None. The four this section used to list (#137, #140, #141, #144) are merged, and the rows
above count them.

## Rough tally

**31 checkable requirements across the five status tables above (the industrial-scope list is a domain list, not counted): 27 done, 3 partial, 1 not started** — since the last refresh, cutting planes moved from not started to done, off by default (#159), and presolve from partial to done (#92 merged as `e25da08`). The one row still not started is GPU acceleration (#16-#19).

What the count hides, same as before:

- The **core is complete and independently verified**: LP, MILP, convex QP and now MIQP all
  work, all are checked by a verifier sharing no code with the solver, and the MILP/MIQP
  proof is corroborated by exhaustion rather than by itself.
- **GPU acceleration is in the PS title and does not exist** (#16-#19). The CPU first-order
  engine it needs is written and demonstrated at 5000x5000; the remaining work is the CUDA
  backend, not the algorithm, and nothing here measures or claims a speed-up.
- The **scale claim is one instance**, not a benchmark (#34).
- Where this project was behind, it can say by how much and why: Dantzig pricing cost 2.16x
  HiGHS's iterations (#66). Devex was built, held back while the measurement said it cost a
  correct answer on the medium tier, and made the default only once the basis-conditioning
  fixes (#144, #147) removed that failure and the measurement was repeated. That is a
  stronger position than a wall-clock number would be.

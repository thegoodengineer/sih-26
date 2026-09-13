# PS26119 coverage tracker

One place to see what PS26119 asks for against what exists on `main`, cross-checked against
`git log` and the source directly rather than asserted. This file mirrors and supersedes the
tracker previously kept only in issue #54's body — a repo file is diffable and reviewable the
way an issue body is not, and every status here was checked against a specific commit.

**Refreshed 2026-09-10 at `351a558`; every number cites the CSV it comes from, and the CSV it cites is the one the number came from.**

## Problem classes

| requirement | status | evidence / issue |
|---|---|---|
| Linear Programming | **done** | revised primal simplex; Netlib full set 78/89 (`bench/results/netlib-full-53cbe16.csv`), medium 48/50, small 9/9 (`bench/results/netlib-{medium,small}-adcee1b.csv`; #34). Cross-checked against HiGHS (#154): on every instance where the solver produces a final answer, that answer agrees with HiGHS to 3.0e-07 or better - that worst case is `pilot`, and the next is `e226` at 3.1e-10, which is the figure this row used to quote as though it were the worst; 8 of the 12 instances cross-checked are cases where the published Netlib table is the outlier, not our answer — the full set is the headline per #142, the narrower tiers being row-capped and so the easier half |
| Mixed-Integer LP | **done** | branch & bound; corroborated by exhaustive oracle in the demo |
| Quadratic Programming | **done** | #55 — Condat-Vu primal-dual engine, `src/qp/`. Convexity decided by LDL^T on `sense * Q` before any arithmetic; non-convex is refused with a certificate, never solved to a local point. Readable from a QPS `QUADOBJ` file (#112) and independently verified. |
| Mixed-Integer QP | **done** | #139 — branch and bound over convex QP node relaxations, joining the two existing engines. `src/core/solve.cpp` dispatches a real `ProblemClass::kMiqp` case; a non-convex Hessian is still refused before any arithmetic, exactly as plain QP does. `tests/unit/test_miqp.cpp` covers it. (Superseded: this used to return `not_solved`; #142 corrected the demo and docs to stop saying so.) |
| Modular for NLP / MINLP | partial | `solve()` seam exists and dispatches four classes cleanly (LP, MILP, QP, MIQP); NLP/MINLP themselves are not attempted |

## Algorithms the PS names

| requirement | status | evidence / issue |
|---|---|---|
| Revised simplex | **done** | `src/simplex/primal_simplex.cpp`, bounded-variable, composite phase 1, no big-M |
| First-order methods (PDHG) | **done** | `src/pdhg/`, restarted PDHG on the CPU, opt-in as `--option algorithm=pdhg`. On the nine committed instances **9 of 9** reach `optimal` at both 1e-4 and 1e-8 with restarts on, and 9 of 9 with restarts off (`bench/results/pdhg-79ec7f7.csv`, `docs/BENCHMARKS.md` section 1e): since #229 an answer the first-order method leaves at its iteration limit is finished by the interior point from that point, which is what closes `share2b` (nine polish iterations) and `israel` without restarts (five). It was 8 of 9 and 7 of 9 at `367e0a2` (`bench/results/pdhg-fix-367e0a2.csv`), and 3 of 9 before #179 fixed two things the CUDA branch had quietly fixed and described as preservation: the step size collapsed to its floor on iteration zero, and the loop stopped on a point the report then had to downgrade. A loose tolerance request is ignored by default; `pdhg_stop_at_request=true` waives the dual, gap and complementarity halves of the standard (absolute primal feasibility is kept, so `feasible` still means feasible) and reports the point as `feasible` unless it meets the full standard anyway: 0.85x the iterations at 1e-4 on these instances, and `share2b` becomes a usable point instead of an iteration limit (`bench/results/pdhg-stop-at-request-f18d4b0.csv`; #180). |
| Interior-point methods | **done, opt-in** | #56 via #169: Mehrotra predictor-corrector on the bounded form, normal equations through a from-scratch sparse LDLᵀ (`src/ipm/ipm.cpp`, `src/la/ldl.cpp`), `--option algorithm=ipm`. Produces no basis, so it is not the node engine and not the default; on the full Netlib set it verified 47/89 (`bench/results/netlib-full-59ac6e3-ipm.csv`, 52 optimal, two rejected by the verifier in original units after postsolve: `pilot4`, `scagr25`), against 79/89 for the dual simplex at the same commit. |
| Branch-and-bound | **done** | `src/mip/branch_and_bound.cpp`, domain-change stacks, no per-node copy |
| Branch-and-cut / cutting planes | **done, off by default** | #159 (closes #23): root Gomory mixed-integer and lifted knapsack cover cuts, `src/mip/cuts.cpp`, validity gated in exact arithmetic against the rational oracle (`tests/unit/test_cuts.cpp`) and by the 600-instance MILP fuzz run with cuts on. Off by default by measurement: the A/B at 60 s on the 30 MIPLIB instances took the node count to 0.887x over the instances that end the same way and cost one proof (`enlight8`; a second apparent loss was only the pre-#188 status convention) (`bench/results/miplib-cuts-{off,on}.csv`; `docs/BENCHMARKS.md` section 2). No MIR cuts, none below the root. |
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
| Difficult MILP formulations | **partial** | MIPLIB runs (below), but 9 of 30 proved is the honest reading |
| **Thousands to millions of variables** | **partial, measured to a million** | #198 via `bench/runners/scale.py`, against optima exact by construction (`docs/BENCHMARKS.md` sections 1f and 1f.1). Two experiments. **Under a 120 s clock** (`scale-f7ca7e9.csv`, four sizes, three engines): 7 of 12 solves reach the optimum; the first-order engine reaches it at **100,000 x 100,000** to a relative 9.1e-08 without certifying it inside the limit, the interior point at 5,000 since the AMD ordering (#193; it was a numerical error there before), and the dual simplex only at 1,000. **Under a fixed budget of 1,000 iterations** (`scale-iterations-f7ca7e9.csv`), which removes the machine and is the only accuracy claim here another machine reproduces exactly: from 1,000 up to **1,000,000 x 1,000,000** the unpolished error stays between 6.9e-06 and 6.1e-04, so the iteration count a first-order method needs does not grow with the model - only the cost per iteration does; the interior-point polish (#229) takes the 1,000-row point from 6.1e-04 to 1.9e-10 in 7 iterations and declines within its 30 s budget at every larger size of this family, whose factor is dense. A second generated shape, a staircase with the structure of a multi-period planning model, moves the interior point to 20,000 rows and is section 1f.2 (`scale-staircase-f7ca7e9.csv`, 8 of 12); the random family was an expander graph, the worst case for any method that factorizes. And an industrial-structured one (#211, section 1f.3, `scale-refinery-f7ca7e9.csv`): a T-period refinery planning LP with the optimum exact by construction, solved exactly at the monthly year by all three engines and at the daily year (32,485 rows) by the interior point and by polished PDHG, reached at the hourly year (779,640 rows) by PDHG to 1.1e-06 and proved by nothing - 5 of 9. What is still not evidence: a real refinery's own data, which no generator supplies. |

## Expected solution

| requirement | status | note |
|---|---|---|
| Basic API **or** CLI | **done** | CLI satisfies the PS wording; C API (#58) and Python bindings (#59) both merged (#128, #129) on top of it |
| Netlib benchmark | **done** | measured on `main` at `53cbe16`: full set **78/89**, medium **48/50**, small **9/9**, every pass also verified by the independent checker (`bench/results/netlib-{full,medium,small}-53cbe16.csv`, which is what `latest_result.py` selects for all three tiers). The 11 full-set non-passes by reason: 7 where Netlib's published value is the outlier and our answer agrees with HiGHS (`80bau3b`, `e226` by its objective constant, `ganges`, `greenbea`, `greenbeb`, `nesm`, `scrs8`); 2 time limits at 120 s (`dfl001`, and `pilot87`, which reaches its optimum in 26,226 iterations whenever the machine gives it the time); `pilot`, whose answer agrees with HiGHS but which our own dual-feasibility check downgrades to feasible; `maros-r7`, where the solver declines to answer rather than claim. On this laptop the clock decides exactly one instance: `pilot87` reaches its optimum in 26,226 iterations every time under `--option iteration_limit=27000`, and whether that fits inside the 120 s limit varies with the machine - it did at `adcee1b` (55.9 s, 79/89) and did not in the run above. 86 of the 89 rows are identical in status, iteration count and objective between the two; the others are `dfl001` and `fit2p`, both clock-decided (#172). Answers and verdicts never move, only the clock. |
| MIPLIB benchmark | **done, weakly** | #24 — 30 instances at 60 s on `main` at `f7ca7e9`: 13 reach the published optimum, 9 prove it (9 optimal, 19 feasible, 2 node limit; `bench/results/miplib-f7ca7e9.csv`, on AC, alone on the machine). It was 6 proved at `53cbe16`; the three that moved - `f2gap40400`, `flugpl`, `p0201` - stop on their gap target, which #188 made report `optimal`: a renamed status, not a better search. Reliability branching (#69) moved the A/B at `63ec8de` by one instance each way (14 matched / 7 proved against 13 / 6 for most-fractional branching, `bench/results/miplib-{reliability,most-fractional}-63ec8de.csv`); root cuts (#159) landed and are off by default: measured on the same commit they prove the same nine, take the nodes to 0.918x and cost one published match (`noswot`; the row above, `docs/BENCHMARKS.md` section 2), so the proof rate now waits on MIR cuts below the root (#221), a longer limit, or cheaper node LPs. |
| Mittelmann benchmark | **done, and it fails** | #60 via #169: fetch with recorded provenance (`bench/runners/fetch_mittelmann.py`, Netlib-packed archives decoded with `emps`) and a runner that names every instance's outcome. On the eight smallest archives at 300 s on `main` at `f7ca7e9` (`bench/results/mittelmann-f7ca7e9.csv`): **0 of 8** reach `optimal`; all eight rows are named `time_limit`s - `qap15` no longer goes singular (#174, fixed in #178), and `bdry2`'s overrun of the limit is 380 s where it was 648 s, the remainder being the sparse LU the deadline does not yet reach (#208) (`docs/BENCHMARKS.md` section 1d). HiGHS finishes two of the eight in the same limit on this machine state (four on the cooler run, `mittelmann-ca64dd5.csv`). This is where the solver stops today, stated rather than omitted. |
| Compared against an established solver | **done** | HiGHS as a separate process, objectives agree **50/50** on the medium tier, and every full-set non-pass cross-checked by name (`bench/results/cross-check-highs-adcee1b.csv`). On speed, the ratio is quotable since `f7ca7e9`: three runs of the same binary within an hour (`bench/results/compare-highs-medium-f7ca7e9{,-second,-third}.csv`, alone on the machine, on mains) put the median per-instance ratio at **2.01x, 2.04x and 2.12x** slower, total time 2.64x to 2.62x; the first and third are an hour apart and agree within 6%, the bar #212 set. The earlier committed pair three days apart read 1.38x and 2.11x, which is why no number was quoted before, and the caveats stand: 9 to 16 of 50 timing envelopes overlap outright, the instances solve in single-digit milliseconds, and the multiplier belongs to this machine state as much as to the solver. The reproducible comparison remains iterations: 2.16x behind on the small set (#66). |
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
- The **scale evidence is generated, and the largest industrial-shaped model is solved at 32,485 rows, not at a million** (#198). It reaches a million rows and columns on two generated shapes, one random and one structured like a planning model, and since #211 a refinery planning model rolled out by the year: the daily year is solved exactly, the hourly year (779,640 rows) is reached by the first-order engine to 1.1e-06 and proved by nothing. Sections 1f to 1f.3 of `docs/BENCHMARKS.md` say what each shows. What is still missing is a real refinery's own data.
- Where this project was behind, it can say by how much and why: Dantzig pricing cost 2.16x
  HiGHS's iterations (#66). Devex was built, held back while the measurement said it cost a
  correct answer on the medium tier, and made the default only once the basis-conditioning
  fixes (#144, #147) removed that failure and the measurement was repeated. That is a
  stronger position than a wall-clock number would be.

# PS26119 coverage tracker

One place to see what PS26119 asks for against what exists on `main`, cross-checked against
`git log` and the source directly rather than asserted. This file mirrors and supersedes the
tracker previously kept only in issue #54's body — a repo file is diffable and reviewable the
way an issue body is not, and every status here was checked against a specific commit.

**Refreshed 2026-08-31 at `ca6fde9`.**

## Problem classes

| requirement | status | evidence / issue |
|---|---|---|
| Linear Programming | **done** | revised primal simplex; Netlib full set 66/89, medium 43/50, small 9/9 (#34) — the full set is the headline per #142, the narrower tiers being row-capped and so the easier half |
| Mixed-Integer LP | **done** | branch & bound; corroborated by exhaustive oracle in the demo |
| Quadratic Programming | **done** | #55 — Condat-Vu primal-dual engine, `src/qp/`. Convexity decided by LDL^T on `sense * Q` before any arithmetic; non-convex is refused with a certificate, never solved to a local point. Readable from a QPS `QUADOBJ` file (#112) and independently verified. |
| Mixed-Integer QP | **done** | #139 — branch and bound over convex QP node relaxations, joining the two existing engines. `src/core/solve.cpp` dispatches a real `ProblemClass::kMiqp` case; a non-convex Hessian is still refused before any arithmetic, exactly as plain QP does. `tests/unit/test_miqp.cpp` covers it. (Superseded: this used to return `not_solved`; #142 corrected the demo and docs to stop saying so.) |
| Modular for NLP / MINLP | partial | `solve()` seam exists and dispatches four classes cleanly (LP, MILP, QP, MIQP); NLP/MINLP themselves are not attempted |

## Algorithms the PS names

| requirement | status | evidence / issue |
|---|---|---|
| Revised simplex | **done** | `src/simplex/primal_simplex.cpp`, bounded-variable, composite phase 1, no big-M |
| Interior-point methods | **not started** | #56 |
| Branch-and-bound | **done** | `src/mip/branch_and_bound.cpp`, domain-change stacks, no per-node copy |
| Branch-and-cut / cutting planes | **not started** | #23 — the direct cause of the weak MIPLIB proof rate below |
| Presolve | **partial** | #43 landed six reductions (empty row, redundant row, fixed column, empty column, singleton row, forcing row), on by default, postsolve re-measured against the original model. Free-column-singleton and doubleton-equation elimination (#92) — the two reductions #43 explicitly deferred, and where most of the remaining reduction on `sc105`/`afiro`-shaped models lives — are still open; correctly reporting this as partial rather than done matters because a wrong dual from a bad postsolve is exactly the failure class this project is built to avoid. |
| Heuristics | **done** | #25 — diving to an incumbent, then best-bound |
| Advanced node selection | partial | depth-first while diving, best-bound after; no pseudocost or reliability branching (#69) |

## Implementation requirements

| requirement | status | evidence / issue |
|---|---|---|
| Sparse matrix techniques | **done** | CSC/CSR, sparse Markowitz LU with threshold stability |
| Efficient numerical linear algebra | **done** | #49 scaling (Ruiz + Pock-Chambolle, default on), #50 basis update (product-form, with an FTRAN-residual accuracy check that forces refactorization). #144 (open, CI green) fixes `eliminate()` reporting the wrong singular column when an earlier column is merely unpivotable within the search budget — a prerequisite for basis repair, not the repair itself. |
| Pricing | **partial** | Dantzig is the default; Devex (#66) landed in #126 and is selectable via `--option pricing=devex`, but stays opt-in — measured on the Netlib medium tier it turns `grow22` from `optimal` into a singular basis for no reduction in the singular-basis count elsewhere, so shipping it as the default would trade an iteration-count headline for a wrong answer. The Harris two-pass ratio test (#67) landed in #137 and is selectable via `--option ratio_test=harris`; it was built specifically to test whether it would fix that interaction, and measured, it does not — it trades `grow22` for no reduction in singular-basis failures, so it too stays opt-in. |
| Multi-core parallelization | **not started** | #57 |
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
| Difficult MILP formulations | **partial** | MIPLIB runs (below), but 5 of 30 proved is the honest reading |
| **Thousands to millions of variables** | **partial — still the biggest single gap** | one generated 5000x5000 LP against an optimum exact by construction. A demonstration, not a benchmark: one instance, generated rather than industrial, nowhere near the "millions" end. |

## Expected solution

| requirement | status | note |
|---|---|---|
| Basic API **or** CLI | **done** | CLI satisfies the PS wording; C API (#58) and Python bindings (#59) both merged (#128, #129) on top of it |
| Netlib benchmark | **done** | small 9/9 live in the demo, medium 43/50 (#34, last measured at `a90db47` / re-run pending in the PR this tracker update accompanies). Six remaining failures: singular bases (`d6cube`, `grow15`, `pilot4`), accuracy short of the 1e-6 bar (`scrs8`, `grow7`), and dual feasibility (`etamacro`). `e226` differs from the published table only by an objective-row constant convention and verifies as optimal (#134). |
| MIPLIB benchmark | **done, weakly** | #24 — 30 instances: 5 optimal, 21 feasible, 4 node limit. The proof rate is what #23 and #69 exist to fix. |
| Mittelmann benchmark | **not started** | #60 |
| Compared against an established solver | **done** | HiGHS as a separate process, objectives agree 9/9. On speed: single-digit-millisecond instances put both solvers' timing envelopes on top of each other: the median ratio moved between 0.72x and 1.60x on an unchanged binary at every repeat count tried (#120), so "indistinguishable" is the honest reading, not a specific multiplier. The reproducible comparison is iterations: 2.16x behind on the small set (#66). |
| Robustness demonstration | **done** | `demo/run_sih_demo.sh` section 4, all three hazards the PS names |
| Transparent, extensible, sovereign foundation | **done** | provenance discipline, citations in code, exact oracles, an independent verifier sharing no code with the solver |

## Open PRs relevant to this table (not yet merged; nothing above counts them as done)

| PR | closes | CI |
|---|---|---|
| #137 `feat/harris-ratio-test` | #67 | green |
| #140 `fix/highs-log-tempdir` | #47 | green |
| #141 `refactor/split-mps-reader` | #9 | green |
| #144 `fix/eliminate-singular-location` | #143 | green |

## Rough tally

**30 checkable requirements: 22 done, 5 partial, 3 not started** — MIQP moved from partial to
done since the last refresh (#139, #142); presolve moved from done to partial, correcting an
overstatement (doubleton and free-column-singleton, #92, are not merged).

What the count hides, same as before:

- The **core is complete and independently verified**: LP, MILP, convex QP and now MIQP all
  work, all are checked by a verifier sharing no code with the solver, and the MILP/MIQP
  proof is corroborated by exhaustion rather than by itself.
- **GPU acceleration is in the PS title and does not exist** (#16-#19). The CPU first-order
  engine it needs is written and demonstrated at 5000x5000; the remaining work is the CUDA
  backend, not the algorithm, and nothing here measures or claims a speed-up.
- The **scale claim is one instance**, not a benchmark (#34).
- Where this project is behind, it can say by how much and why: Dantzig pricing costs 2.16x
  HiGHS's iterations (#66), and the two most direct candidate fixes (Devex, Harris) are both
  built, measured, and both currently decline to become the default because the measurement
  says they cost a correct answer somewhere on the medium tier. That is a stronger position
  than a wall-clock number would be.

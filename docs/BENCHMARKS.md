# SANKHYA — benchmarks

<!-- GENERATED FILE. Do not edit by hand. -->
<!-- Regenerate with: python bench/runners/make_benchmarks_doc.py -->

This file is generated from the CSVs in `bench/results/`, so it cannot drift from the
evidence. Every number below came out of a run that recorded the instance sha256, the git
commit and the machine tag alongside it.

Times are wall-clock, measured around the whole process, so they include reading the model
and writing the outputs. That makes them slightly pessimistic and honest; it is not the
figure to quote for algorithmic speed, and no attempt is made to dress it up.

Reporting follows Mittelmann's conventions: shifted geometric means with a
1-second shift, an explicit time limit, and failures counted and named
rather than dropped.

---

## 1. Netlib LP — accuracy against published optima

The reference optimum for each instance is parsed by `bench/runners/fetch_data.py` from
Netlib's own `readme`. None of these values was typed from memory.

### 1a. The small set — what the demo runs

Eight instances, committed to the repository so a fresh clone can reproduce this with no
network. **This is the set `demo/run_demo.sh` lets a judge pick from, and it is the easy end
of Netlib.** Its pass rate is not the headline; section 1b is.

Source CSV: `bench/results/netlib-small-5869f3c.csv`  
Commit `5869f3c` · machine `Windows-AMD64` · generated 2026-08-26T19:34:55+00:00

**8 of 8 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **8 of the 89 instances** Netlib publishes an optimal value for (set `small`, selected by `fetch_data.py --set small`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

Every instance in this set passed.

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 158 | 0.047 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 16 | 0.040 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 337 | 0.049 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 107 | 0.051 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 46 | 0.037 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 4.1e-16 | 50 | 0.043 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 114 | 0.033 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 68 | 0.041 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.043s**
- slowest solved instance: 0.051s
- worst relative error against a published optimum: **1.06e-11**
- no failures on this set

### 1b. The medium tier — the honest headline

Source CSV: `bench/results/netlib-medium-5869f3c.csv`  
Commit `5869f3c` · machine `Windows-AMD64` · generated 2026-08-26T19:36:34+00:00

**41 of 50 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **50 of the 89 instances** Netlib publishes an optimal value for (set `medium`, selected by `fetch_data.py --set medium`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

**9 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| basis went singular (#49) | 3 | d6cube, grow15, pilot4 |
| degenerate stall (#51) | 2 | tuff, wood1p |
| disagrees with the published optimum (#75) | 2 | e226, scrs8 |
| duals miss feasibility (#52) | 1 | etamacro |
| point misses feasibility (#72) | 1 | grow7 |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 158 | 0.053 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 16 | 0.032 | yes |
| `agg` | 488 | 163 | optimal | -3.5991767287e+07 | -3.5991767287e+07 | 1.2e-11 | 141 | 0.069 | yes |
| `bandm` | 305 | 472 | optimal | -1.5862801845e+02 | -1.5862801845e+02 | 7.6e-13 | 574 | 0.080 | yes |
| `beaconfd` | 173 | 262 | optimal | 3.3592485807e+04 | 3.3592485807e+04 | 6.0e-12 | 96 | 0.052 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 337 | 0.072 | yes |
| `boeing1` | 351 | 384 | optimal | -3.3521356751e+02 | -3.3521356751e+02 | 8.6e-12 | 760 | 0.067 | yes |
| `boeing2` | 166 | 143 | optimal | -3.1501872802e+02 | -3.1501872802e+02 | 1.5e-11 | 208 | 0.051 | yes |
| `bore3d` | 233 | 315 | optimal | 1.3730803942e+03 | 1.3730803942e+03 | 6.2e-12 | 167 | 0.077 | yes |
| `brandy` | 220 | 249 | optimal | 1.5185098965e+03 | 1.5185098965e+03 | 7.8e-12 | 323 | 0.068 | yes |
| `capri` | 271 | 353 | optimal | 2.6900129138e+03 | 2.6900129138e+03 | 1.2e-11 | 458 | 0.069 | yes |
| `d6cube` | 415 | 6184 | numerical_error | 1.0000000000e+00 | 3.1549166667e+02 | 1.0e+00 | 5230 | 2.336 | - |
| `degen2` | 444 | 534 | optimal | -1.4351780000e+03 | -1.4351780000e+03 | 0.0e+00 | 1495 | 0.726 | yes |
| `e226` | 223 | 282 | optimal | -1.1638929066e+01 | -1.8751929066e+01 | 3.8e-01 | 592 | 0.068 | yes |
| `etamacro` | 400 | 688 | feasible | -7.5571523316e+02 | -7.5571521774e+02 | 2.0e-08 | 661 | 0.083 | yes |
| `finnis` | 497 | 614 | optimal | 1.7279106560e+05 | 1.7279096547e+05 | 5.8e-07 | 643 | 0.086 | yes |
| `fit1d` | 24 | 1026 | optimal | -9.1463780924e+03 | -9.1463780924e+03 | 2.3e-12 | 1622 | 0.270 | yes |
| `fit2d` | 25 | 10500 | optimal | -6.8464293294e+04 | -6.8464293294e+04 | 2.4e-12 | 30210 | 15.064 | yes |
| `forplan` | 161 | 421 | optimal | -6.6421896127e+02 | -6.6421873953e+02 | 3.3e-07 | 315 | 0.134 | yes |
| `grow15` | 300 | 645 | numerical_error | 0.0000000000e+00 | -1.0687094129e+08 | 1.0e+00 | 577 | 0.312 | - |
| `grow22` | 440 | 946 | optimal | -1.6083433648e+08 | -1.6083433648e+08 | 1.6e-11 | 1246 | 0.259 | yes |
| `grow7` | 140 | 301 | numerical_error | -4.7787811815e+07 | -4.7787811815e+07 | 6.0e-12 | 297 | 0.087 | - |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 251 | 0.078 | yes |
| `kb2` | 43 | 41 | optimal | -1.7499001299e+03 | -1.7499001299e+03 | 3.5e-12 | 71 | 0.081 | yes |
| `lotfi` | 153 | 308 | optimal | -2.5264706062e+01 | -2.5264706062e+01 | 4.7e-12 | 261 | 0.055 | yes |
| `pilot4` | 410 | 1000 | numerical_error | 0.0000000000e+00 | -2.5811392641e+03 | 1.0e+00 | 998 | 0.183 | - |
| `recipe` | 91 | 180 | optimal | -2.6661600000e+02 | -2.6661600000e+02 | 1.1e-15 | 49 | 0.056 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 107 | 0.057 | yes |
| `sc205` | 205 | 203 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 255 | 0.094 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 46 | 0.055 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 4.1e-16 | 50 | 0.051 | yes |
| `scagr25` | 471 | 500 | optimal | -1.4753433061e+07 | -1.4753433061e+07 | 1.6e-11 | 591 | 0.195 | yes |
| `scagr7` | 129 | 140 | optimal | -2.3313898243e+06 | -2.3313892548e+06 | 2.4e-07 | 159 | 0.054 | yes |
| `scfxm1` | 330 | 457 | optimal | 1.8416759028e+04 | 1.8416759028e+04 | 1.9e-11 | 648 | 0.090 | yes |
| `scorpion` | 388 | 358 | optimal | 1.8781248227e+03 | 1.8781248227e+03 | 2.0e-11 | 323 | 0.074 | yes |
| `scrs8` | 490 | 1169 | optimal | 9.0429695380e+02 | 9.0429998619e+02 | 3.4e-06 | 645 | 0.109 | yes |
| `scsd1` | 77 | 760 | optimal | 8.6666666743e+00 | 8.6666666743e+00 | 3.8e-12 | 534 | 0.079 | yes |
| `scsd6` | 147 | 1350 | optimal | 5.0500000078e+01 | 5.0500000078e+01 | 5.2e-12 | 627 | 0.168 | yes |
| `scsd8` | 397 | 2750 | optimal | 9.0499999993e+02 | 9.0499999993e+02 | 5.0e-12 | 926 | 0.194 | yes |
| `sctap1` | 300 | 480 | optimal | 1.4122500000e+03 | 1.4122500000e+03 | 0.0e+00 | 303 | 0.082 | yes |
| `share1b` | 117 | 225 | optimal | -7.6589318579e+04 | -7.6589318579e+04 | 2.4e-12 | 207 | 0.069 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 114 | 0.058 | yes |
| `ship04l` | 402 | 2118 | optimal | 1.7933245380e+06 | 1.7933245380e+06 | 1.7e-11 | 511 | 0.192 | yes |
| `ship04s` | 402 | 1458 | optimal | 1.7987147004e+06 | 1.7987147004e+06 | 2.5e-11 | 385 | 0.083 | yes |
| `stair` | 356 | 467 | optimal | -2.5126695119e+02 | -2.5126695119e+02 | 1.2e-11 | 978 | 1.437 | yes |
| `standata` | 359 | 1075 | optimal | 1.2576995000e+03 | 1.2576995000e+03 | 0.0e+00 | 53 | 0.074 | yes |
| `standmps` | 467 | 1075 | optimal | 1.4060175000e+03 | 1.4060175000e+03 | 0.0e+00 | 193 | 0.068 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 68 | 0.051 | yes |
| `tuff` | 333 | 587 | numerical_error | 0.0000000000e+00 | 2.9214776509e-01 | 2.9e-01 | 1001 | 0.147 | - |
| `wood1p` | 244 | 2594 | numerical_error | 0.0000000000e+00 | 1.4429024116e+00 | 1.0e+00 | 1000 | 0.626 | - |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.201s**
- slowest solved instance: 15.064s
- worst relative error against a published optimum: **5.79e-07**
- **failed: `d6cube`, `e226`, `etamacro`, `grow15`, `grow7`, `pilot4`, `scrs8`, `tuff`, `wood1p`** — kept in the table on purpose

---

## 2. Correctness beyond the objective value

An objective that matches a published number is necessary, not sufficient — it says nothing
about whether the reported solution is internally consistent. Two independent checks cover
that, and both run in CI:

- **`tools/verify_solution.py`** re-parses the model with its own MPS reader, recomputes the
  row activities, the objective, the reduced costs and the dual objective, and checks primal
  feasibility, dual feasibility, complementary slackness and strong duality. It shares no
  code with the solver, so a reader bug shows up as a disagreement rather than as agreement.
  The `verified` column above is its verdict.

- **The exact rational oracle** (`tests/oracles/`) solves generated instances in exact
  arithmetic with no rounding error anywhere, and the floating-point simplex is compared
  against it. See `docs/PROVENANCE.md` for the citations.

---

## 3. Comparison against an established solver

Source CSV: `bench/results/compare-highs-5869f3c.csv`  
Commit `5869f3c` · machine `Windows-AMD64`

**8 of 8** instances where the two solvers agree on the objective.

Times are **solver-internal on both sides** - HiGHS's own `getRunTime()` against our `effort.solve_seconds` - so process start-up is excluded for both. At this instance size start-up would otherwise dominate and the comparison would measure the wrong thing entirely.

| instance | SANKHYA obj | HiGHS obj | agree | SANKHYA (s) | HiGHS (s) | ratio |
|---|---:|---:|:--:|---:|---:|---:|
| `adlittle` | 2.25494963e+05 | 2.25494963e+05 | yes | 0.006 | 0.022 | 0.28x |
| `afiro` | -4.64753143e+02 | -4.64753143e+02 | yes | 0.001 | 0.002 | 0.43x |
| `blend` | -3.08121498e+01 | -3.08121498e+01 | yes | 0.017 | 0.005 | 3.62x |
| `sc105` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.010 | 0.003 | 3.17x |
| `sc50a` | -6.45750771e+01 | -6.45750771e+01 | yes | 0.002 | 0.001 | 1.64x |
| `sc50b` | -7.00000000e+01 | -7.00000000e+01 | yes | 0.002 | 0.002 | 1.10x |
| `share2b` | -4.15732241e+02 | -4.15732241e+02 | yes | 0.005 | 0.009 | 0.54x |
| `stocfor1` | -4.11319762e+04 | -4.11319762e+04 | yes | 0.003 | 0.005 | 0.50x |

**Summary**

- SANKHYA shifted geometric mean: **0.006s**
- HiGHS shifted geometric mean: **0.006s**
- SANKHYA is **0.9x** the HiGHS time by that measure

The two are **within noise of each other** here, at 0.93x. A narrow claim: eight small instances settle nothing about large models. HiGHS is a decade of specialist work with presolve, a dual simplex and a mature pricing scheme, and this solver still has neither of the first two. The part that has to be right first is that **the answers agree** - the problem statement asks us to compare, not to win.

---

## 4. What these numbers do not say

- **Nothing here supports a claim about large models.** The medium tier is capped at
  instances Netlib publishes with a few hundred rows. PS26119 asks about "thousands to
  millions of variables"; that is not demonstrated anywhere on this page, and no pass rate
  above substitutes for it. Tracked as part of #54.
- Wall-clock times at this size are dominated by process start-up and file reading, so
  ratios between solvers are not meaningful until the instances get big enough to matter.
  The comparison in section 3 uses solver-internal time on both sides for that reason.
- The failures in section 1b are real and are not going to be quietly dropped from a later
  edition of this file. Each one carries the issue tracking it.
- The largest remaining gap is not on this page at all: there is no QP engine, no
  interior-point method and no GPU backend, all three named in PS26119. `docs/PROVENANCE.md`
  and issue #54 carry the full accounting.

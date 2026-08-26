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

Source CSV: `bench/results/netlib-small-79809da.csv`  
Commit `79809da` · machine `Windows-AMD64` · generated 2026-08-26T18:33:22+00:00

**8 of 8 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **8 of the 89 instances** Netlib publishes an optimal value for (set `small`, selected by `fetch_data.py --set small`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

Every instance in this set passed.

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 160 | 0.038 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 16 | 0.025 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 423 | 0.033 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 107 | 0.045 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 46 | 0.033 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 4.1e-16 | 50 | 0.030 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 117 | 0.036 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 79 | 0.039 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.035s**
- slowest solved instance: 0.045s
- worst relative error against a published optimum: **1.06e-11**
- no failures on this set

### 1b. The medium tier — the honest headline

Source CSV: `bench/results/netlib-medium-79809da.csv`  
Commit `79809da` · machine `Windows-AMD64` · generated 2026-08-26T18:29:45+00:00

**39 of 50 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **50 of the 89 instances** Netlib publishes an optimal value for (set `medium`, selected by `fetch_data.py --set medium`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

**11 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| basis went singular (#49) | 3 | d6cube, grow15, pilot4 |
| degenerate stall (#51) | 3 | bore3d, tuff, wood1p |
| disagrees with the published optimum (#75) | 2 | e226, scrs8 |
| duals miss feasibility (#52) | 1 | etamacro |
| point misses feasibility (#72) | 1 | grow7 |
| verifier cannot parse the model (#48) | 1 | forplan |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 160 | 0.056 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 16 | 0.042 | yes |
| `agg` | 488 | 163 | optimal | -3.5991767287e+07 | -3.5991767287e+07 | 1.2e-11 | 137 | 0.043 | yes |
| `bandm` | 305 | 472 | optimal | -1.5862801845e+02 | -1.5862801845e+02 | 7.7e-13 | 584 | 0.053 | yes |
| `beaconfd` | 173 | 262 | optimal | 3.3592485807e+04 | 3.3592485807e+04 | 6.0e-12 | 106 | 0.049 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 423 | 0.035 | yes |
| `boeing1` | 351 | 384 | optimal | -3.3521356751e+02 | -3.3521356751e+02 | 8.6e-12 | 770 | 0.042 | yes |
| `boeing2` | 166 | 143 | optimal | -3.1501872802e+02 | -3.1501872802e+02 | 1.5e-11 | 212 | 0.040 | yes |
| `bore3d` | 233 | 315 | numerical_error | 0.0000000000e+00 | 1.3730803942e+03 | 1.0e+00 | 1014 | 0.068 | - |
| `brandy` | 220 | 249 | optimal | 1.5185098965e+03 | 1.5185098965e+03 | 7.8e-12 | 919 | 0.055 | yes |
| `capri` | 271 | 353 | optimal | 2.6900129138e+03 | 2.6900129138e+03 | 1.2e-11 | 433 | 0.045 | yes |
| `d6cube` | 415 | 6184 | numerical_error | 0.0000000000e+00 | 3.1549166667e+02 | 1.0e+00 | 4836 | 5.577 | - |
| `degen2` | 444 | 534 | optimal | -1.4351780000e+03 | -1.4351780000e+03 | 0.0e+00 | 1495 | 0.553 | yes |
| `e226` | 223 | 282 | optimal | -1.1638929066e+01 | -1.8751929066e+01 | 3.8e-01 | 542 | 0.041 | yes |
| `etamacro` | 400 | 688 | feasible | -7.5571523316e+02 | -7.5571521774e+02 | 2.0e-08 | 810 | 0.051 | yes |
| `finnis` | 497 | 614 | optimal | 1.7279106560e+05 | 1.7279096547e+05 | 5.8e-07 | 662 | 0.100 | yes |
| `fit1d` | 24 | 1026 | optimal | -9.1463780924e+03 | -9.1463780924e+03 | 2.3e-12 | 1622 | 0.104 | yes |
| `fit2d` | 25 | 10500 | optimal | -6.8464293294e+04 | -6.8464293294e+04 | 2.4e-12 | 30210 | 23.790 | yes |
| `forplan` | 161 | 421 | optimal | -6.6421896127e+02 | -6.6421873953e+02 | 3.3e-07 | 393 | 0.190 | **NO** |
| `grow15` | 300 | 645 | numerical_error | 0.0000000000e+00 | -1.0687094129e+08 | 1.0e+00 | 577 | 0.239 | - |
| `grow22` | 440 | 946 | optimal | -1.6083433648e+08 | -1.6083433648e+08 | 1.6e-11 | 1246 | 0.345 | yes |
| `grow7` | 140 | 301 | numerical_error | -4.7787811815e+07 | -4.7787811815e+07 | 6.0e-12 | 297 | 0.073 | - |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 252 | 0.091 | yes |
| `kb2` | 43 | 41 | optimal | -1.7499001299e+03 | -1.7499001299e+03 | 3.5e-12 | 71 | 0.065 | yes |
| `lotfi` | 153 | 308 | optimal | -2.5264706062e+01 | -2.5264706062e+01 | 4.8e-12 | 304 | 0.059 | yes |
| `pilot4` | 410 | 1000 | numerical_error | 0.0000000000e+00 | -2.5811392641e+03 | 1.0e+00 | 2503 | 0.293 | - |
| `recipe` | 91 | 180 | optimal | -2.6661600000e+02 | -2.6661600000e+02 | 1.1e-15 | 49 | 0.051 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 107 | 0.048 | yes |
| `sc205` | 205 | 203 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 255 | 0.071 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 46 | 0.053 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 4.1e-16 | 50 | 0.053 | yes |
| `scagr25` | 471 | 500 | optimal | -1.4753433061e+07 | -1.4753433061e+07 | 1.6e-11 | 512 | 0.161 | yes |
| `scagr7` | 129 | 140 | optimal | -2.3313898243e+06 | -2.3313892548e+06 | 2.4e-07 | 185 | 0.060 | yes |
| `scfxm1` | 330 | 457 | optimal | 1.8416759028e+04 | 1.8416759028e+04 | 1.9e-11 | 557 | 0.073 | yes |
| `scorpion` | 388 | 358 | optimal | 1.8781248227e+03 | 1.8781248227e+03 | 2.0e-11 | 400 | 0.054 | yes |
| `scrs8` | 490 | 1169 | optimal | 9.0429695380e+02 | 9.0429998619e+02 | 3.4e-06 | 697 | 0.064 | yes |
| `scsd1` | 77 | 760 | optimal | 8.6666666743e+00 | 8.6666666743e+00 | 3.8e-12 | 534 | 0.030 | yes |
| `scsd6` | 147 | 1350 | optimal | 5.0500000078e+01 | 5.0500000078e+01 | 5.2e-12 | 627 | 0.054 | yes |
| `scsd8` | 397 | 2750 | optimal | 9.0499999993e+02 | 9.0499999993e+02 | 5.0e-12 | 926 | 0.085 | yes |
| `sctap1` | 300 | 480 | optimal | 1.4122500000e+03 | 1.4122500000e+03 | 1.6e-16 | 304 | 0.025 | yes |
| `share1b` | 117 | 225 | optimal | -7.6589318579e+04 | -7.6589318579e+04 | 2.4e-12 | 226 | 0.027 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 117 | 0.020 | yes |
| `ship04l` | 402 | 2118 | optimal | 1.7933245380e+06 | 1.7933245380e+06 | 1.7e-11 | 508 | 0.043 | yes |
| `ship04s` | 402 | 1458 | optimal | 1.7987147004e+06 | 1.7987147004e+06 | 2.5e-11 | 450 | 0.043 | yes |
| `stair` | 356 | 467 | optimal | -2.5126695119e+02 | -2.5126695119e+02 | 1.2e-11 | 926 | 0.125 | yes |
| `standata` | 359 | 1075 | optimal | 1.2576995000e+03 | 1.2576995000e+03 | 1.8e-16 | 64 | 0.030 | yes |
| `standmps` | 467 | 1075 | optimal | 1.4060175000e+03 | 1.4060175000e+03 | 3.2e-16 | 295 | 0.036 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 79 | 0.023 | yes |
| `tuff` | 333 | 587 | numerical_error | 0.0000000000e+00 | 2.9214776509e-01 | 2.9e-01 | 1248 | 0.085 | - |
| `wood1p` | 244 | 2594 | numerical_error | 0.0000000000e+00 | 1.4429024116e+00 | 1.0e+00 | 1000 | 0.228 | - |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.164s**
- slowest solved instance: 23.790s
- worst relative error against a published optimum: **5.79e-07**
- **failed: `bore3d`, `d6cube`, `e226`, `etamacro`, `forplan`, `grow15`, `grow7`, `pilot4`, `scrs8`, `tuff`, `wood1p`** — kept in the table on purpose

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

Source CSV: `bench/results/compare-highs-79809da.csv`  
Commit `79809da` · machine `Windows-AMD64`

**8 of 8** instances where the two solvers agree on the objective.

Times are **solver-internal on both sides** - HiGHS's own `getRunTime()` against our `effort.solve_seconds` - so process start-up is excluded for both. At this instance size start-up would otherwise dominate and the comparison would measure the wrong thing entirely.

| instance | SANKHYA obj | HiGHS obj | agree | SANKHYA (s) | HiGHS (s) | ratio |
|---|---:|---:|:--:|---:|---:|---:|
| `adlittle` | 2.25494963e+05 | 2.25494963e+05 | yes | 0.003 | 0.004 | 0.70x |
| `afiro` | -4.64753143e+02 | -4.64753143e+02 | yes | 0.000 | 0.001 | 0.13x |
| `blend` | -3.08121498e+01 | -3.08121498e+01 | yes | 0.007 | 0.003 | 2.48x |
| `sc105` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.004 | 0.003 | 1.14x |
| `sc50a` | -6.45750771e+01 | -6.45750771e+01 | yes | 0.001 | 0.002 | 0.41x |
| `sc50b` | -7.00000000e+01 | -7.00000000e+01 | yes | 0.002 | 0.003 | 0.65x |
| `share2b` | -4.15732241e+02 | -4.15732241e+02 | yes | 0.003 | 0.005 | 0.62x |
| `stocfor1` | -4.11319762e+04 | -4.11319762e+04 | yes | 0.004 | 0.003 | 1.31x |

**Summary**

- SANKHYA shifted geometric mean: **0.003s**
- HiGHS shifted geometric mean: **0.003s**
- SANKHYA is **1.0x** the HiGHS time by that measure

The two are **within noise of each other** here, at 0.97x. A narrow claim: eight small instances settle nothing about large models. HiGHS is a decade of specialist work with presolve, a dual simplex and a mature pricing scheme, and this solver still has neither of the first two. The part that has to be right first is that **the answers agree** - the problem statement asks us to compare, not to win.

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

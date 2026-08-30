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

Source CSV: `bench/results/netlib-small-410789c.csv`  
Commit `410789c` · machine `Windows-AMD64` · generated 2026-08-30T11:55:13+00:00

**9 of 9 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **9 of the 89 instances** Netlib publishes an optimal value for (set `small`, selected by `fetch_data.py --set small`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

Every instance in this set passed.

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 158 | 0.027 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 16 | 0.020 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 337 | 0.023 | yes |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 251 | 0.029 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 107 | 0.031 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 46 | 0.031 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 4.1e-16 | 50 | 0.030 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 114 | 0.044 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 68 | 0.045 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.031s**
- slowest solved instance: 0.045s
- worst relative error against a published optimum: **1.06e-11**
- no failures on this set

### 1b. The medium tier — the honest headline

Source CSV: `bench/results/netlib-medium-perturb-31fbe4d.csv`  
Commit `31fbe4d` · machine `Windows-AMD64` · generated 2026-08-30T04:58:02+00:00

**43 of 50 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **50 of the 89 instances** Netlib publishes an optimal value for (set `medium`, selected by `fetch_data.py --set medium`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

**7 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| basis went singular (#49) | 3 | d6cube, grow15, pilot4 |
| disagrees with the published optimum (#75) | 2 | e226, scrs8 |
| duals miss feasibility (#52) | 1 | etamacro |
| point misses feasibility (#72) | 1 | grow7 |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 158 | 0.021 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 16 | 0.025 | yes |
| `agg` | 488 | 163 | optimal | -3.5991767287e+07 | -3.5991767287e+07 | 1.2e-11 | 141 | 0.049 | yes |
| `bandm` | 305 | 472 | optimal | -1.5862801845e+02 | -1.5862801845e+02 | 7.6e-13 | 574 | 0.056 | yes |
| `beaconfd` | 173 | 262 | optimal | 3.3592485807e+04 | 3.3592485807e+04 | 6.0e-12 | 96 | 0.034 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 337 | 0.033 | yes |
| `boeing1` | 351 | 384 | optimal | -3.3521356751e+02 | -3.3521356751e+02 | 8.6e-12 | 760 | 0.050 | yes |
| `boeing2` | 166 | 143 | optimal | -3.1501872802e+02 | -3.1501872802e+02 | 1.5e-11 | 208 | 0.033 | yes |
| `bore3d` | 233 | 315 | optimal | 1.3730803942e+03 | 1.3730803942e+03 | 6.2e-12 | 167 | 0.034 | yes |
| `brandy` | 220 | 249 | optimal | 1.5185098965e+03 | 1.5185098965e+03 | 7.8e-12 | 323 | 0.052 | yes |
| `capri` | 271 | 353 | optimal | 2.6900129138e+03 | 2.6900129138e+03 | 1.2e-11 | 458 | 0.062 | yes |
| `d6cube` | 415 | 6184 | numerical_error | 1.0000000000e+00 | 3.1549166667e+02 | 1.0e+00 | 5230 | 2.024 | - |
| `degen2` | 444 | 534 | optimal | -1.4351780000e+03 | -1.4351780000e+03 | 0.0e+00 | 1495 | 347.430 | yes |
| `e226` | 223 | 282 | optimal | -1.1638929066e+01 | -1.8751929066e+01 | 3.8e-01 | 592 | 0.211 | yes |
| `etamacro` | 400 | 688 | feasible | -7.5571523316e+02 | -7.5571521774e+02 | 2.0e-08 | 661 | 0.108 | yes |
| `finnis` | 497 | 614 | optimal | 1.7279106560e+05 | 1.7279096547e+05 | 5.8e-07 | 643 | 0.116 | yes |
| `fit1d` | 24 | 1026 | optimal | -9.1463780924e+03 | -9.1463780924e+03 | 2.3e-12 | 1622 | 0.164 | yes |
| `fit2d` | 25 | 10500 | optimal | -6.8464293294e+04 | -6.8464293294e+04 | 2.4e-12 | 30210 | 10.581 | yes |
| `forplan` | 161 | 421 | optimal | -6.6421896127e+02 | -6.6421873953e+02 | 3.3e-07 | 1041 | 0.059 | yes |
| `grow15` | 300 | 645 | numerical_error | 0.0000000000e+00 | -1.0687094129e+08 | 1.0e+00 | 577 | 0.151 | - |
| `grow22` | 440 | 946 | optimal | -1.6083433648e+08 | -1.6083433648e+08 | 1.6e-11 | 1246 | 0.190 | yes |
| `grow7` | 140 | 301 | numerical_error | -4.7787811815e+07 | -4.7787811815e+07 | 6.0e-12 | 297 | 0.037 | - |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 251 | 0.036 | yes |
| `kb2` | 43 | 41 | optimal | -1.7499001299e+03 | -1.7499001299e+03 | 3.5e-12 | 71 | 0.024 | yes |
| `lotfi` | 153 | 308 | optimal | -2.5264706062e+01 | -2.5264706062e+01 | 4.7e-12 | 261 | 0.034 | yes |
| `pilot4` | 410 | 1000 | numerical_error | 0.0000000000e+00 | -2.5811392641e+03 | 1.0e+00 | 998 | 0.183 | - |
| `recipe` | 91 | 180 | optimal | -2.6661600000e+02 | -2.6661600000e+02 | 1.1e-15 | 49 | 0.025 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 107 | 0.033 | yes |
| `sc205` | 205 | 203 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 255 | 0.048 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 46 | 0.021 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 4.1e-16 | 50 | 0.022 | yes |
| `scagr25` | 471 | 500 | optimal | -1.4753433061e+07 | -1.4753433061e+07 | 1.6e-11 | 591 | 0.070 | yes |
| `scagr7` | 129 | 140 | optimal | -2.3313898243e+06 | -2.3313892548e+06 | 2.4e-07 | 159 | 0.032 | yes |
| `scfxm1` | 330 | 457 | optimal | 1.8416759028e+04 | 1.8416759028e+04 | 1.9e-11 | 648 | 0.084 | yes |
| `scorpion` | 388 | 358 | optimal | 1.8781248227e+03 | 1.8781248227e+03 | 2.0e-11 | 323 | 0.036 | yes |
| `scrs8` | 490 | 1169 | optimal | 9.0429695380e+02 | 9.0429998619e+02 | 3.4e-06 | 645 | 0.061 | yes |
| `scsd1` | 77 | 760 | optimal | 8.6666666743e+00 | 8.6666666743e+00 | 3.8e-12 | 534 | 0.038 | yes |
| `scsd6` | 147 | 1350 | optimal | 5.0500000078e+01 | 5.0500000078e+01 | 5.2e-12 | 627 | 0.068 | yes |
| `scsd8` | 397 | 2750 | optimal | 9.0499999993e+02 | 9.0499999993e+02 | 5.0e-12 | 926 | 0.134 | yes |
| `sctap1` | 300 | 480 | optimal | 1.4122500000e+03 | 1.4122500000e+03 | 0.0e+00 | 303 | 0.038 | yes |
| `share1b` | 117 | 225 | optimal | -7.6589318579e+04 | -7.6589318579e+04 | 2.4e-12 | 207 | 0.045 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 114 | 0.031 | yes |
| `ship04l` | 402 | 2118 | optimal | 1.7933245380e+06 | 1.7933245380e+06 | 1.7e-11 | 511 | 0.081 | yes |
| `ship04s` | 402 | 1458 | optimal | 1.7987147004e+06 | 1.7987147004e+06 | 2.5e-11 | 385 | 0.042 | yes |
| `stair` | 356 | 467 | optimal | -2.5126695119e+02 | -2.5126695119e+02 | 1.2e-11 | 978 | 1.064 | yes |
| `standata` | 359 | 1075 | optimal | 1.2576995000e+03 | 1.2576995000e+03 | 0.0e+00 | 53 | 0.044 | yes |
| `standmps` | 467 | 1075 | optimal | 1.4060175000e+03 | 1.4060175000e+03 | 0.0e+00 | 193 | 0.059 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 68 | 0.044 | yes |
| `tuff` | 333 | 587 | optimal | 2.9214776509e-01 | 2.9214776509e-01 | 3.6e-12 | 969 | 0.117 | yes |
| `wood1p` | 244 | 2594 | optimal | 1.4429024116e+00 | 1.4429024116e+00 | 1.8e-11 | 836 | 0.255 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.302s**
- slowest solved instance: 347.430s
- worst relative error against a published optimum: **5.79e-07**
- **failed: `d6cube`, `e226`, `etamacro`, `grow15`, `grow7`, `pilot4`, `scrs8`** — kept in the table on purpose

---

## 2. MIPLIB — the mixed-integer side

The LP tiers above say nothing about the branch and bound. This is the MILP evidence, and it
is a harder library: MIPLIB instances are chosen to be difficult for mature solvers.

Source CSV: `bench/results/miplib-8dba6a7.csv`  
Commit `8dba6a7` · machine `Windows-AMD64`

**10 of 30** instances reached the published optimum. **5 of 30** also PROVED it - closed the bound rather than stopping at a gap target or a limit.

Those are different claims and are kept apart deliberately. Branch and bound here has no cutting planes - a rounding heuristic and a root dive, but nothing that tightens the relaxation - so it finds good incumbents far more often than it finishes the proof. Collapsing the two columns would hide exactly the thing #23 is meant to improve.

**The time limit decides some of these, not the solver.** `enlight8` proves optimality in about 55 seconds on an idle machine and misses a 60-second budget when the rest of the set is running alongside it - so its row moves with background load rather than with anything about the search. Instances close to the limit should be read as "needs more time than we gave it", not as a capability. The remedy is a longer limit, and the reason this table does not already use one is that the full set takes about half an hour per run as it stands.

Instances are the smallest MIPLIB 2017 instances tagged easy that carry a **proven** optimum (`=opt=` in MIPLIB's own solution file). A `=best=` value is the best anyone has found, not a proof, and scoring against one would let a wrong answer look like a record.

| instance | rows | cols | int | status | our objective | published | rel. gap | nodes | time (s) | matched | proved | verified |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| `b-ball` | 30 | 100 | 88 | feasible | -1.5 | -1.5 | 2.12e-01 | 96310 | 60.5 | yes | **NO** | yes |
| `ej` | 1 | 3 | 3 | feasible | 41014 | 25508 | 1.00e+00 | 139820 | 60.1 | **NO** | **NO** | yes |
| `enlight8` | 64 | 128 | 128 | node_limit | inf | 27 | - | 52287 | 60.0 | **NO** | **NO** | **NO** |
| `enlight_hard` | 100 | 200 | 200 | node_limit | inf | 37 | - | 15363 | 60.1 | **NO** | **NO** | **NO** |
| `f2gap40400` | 40 | 400 | 400 | optimal | 20772 | 20772 | 0.00e+00 | 509 | 1.9 | yes | yes | yes |
| `flugpl` | 18 | 18 | 11 | feasible | 1201500 | 1201500 | 8.74e-05 | 1402 | 0.1 | yes | **NO** | yes |
| `gen-ip016` | 24 | 28 | 28 | feasible | -9424.369109 | -9476.155197 | 8.22e-03 | 33432 | 60.1 | **NO** | **NO** | yes |
| `gen-ip054` | 27 | 30 | 30 | feasible | 6872.821391 | 6840.965642 | 1.37e-02 | 30415 | 60.1 | **NO** | **NO** | yes |
| `gr4x6` | 34 | 48 | 24 | optimal | 202.35 | 202.35 | 0.00e+00 | 142 | 0.2 | yes | yes | yes |
| `gt2` | 29 | 188 | 188 | feasible | 30518 | 21166 | 5.43e-01 | 57469 | 60.1 | **NO** | **NO** | yes |
| `k16x240b` | 256 | 480 | 240 | feasible | 12512 | 11393 | 5.18e-01 | 8305 | 60.1 | **NO** | **NO** | yes |
| `markshare1` | 6 | 62 | 50 | feasible | 28 | 1 | 1.00e+00 | 71834 | 60.1 | **NO** | **NO** | yes |
| `markshare_4_0` | 4 | 34 | 30 | feasible | 5 | 1 | 1.00e+00 | 103588 | 60.1 | **NO** | **NO** | yes |
| `markshare_5_0` | 5 | 45 | 40 | feasible | 16 | 1 | 1.00e+00 | 90227 | 60.0 | **NO** | **NO** | yes |
| `neos-1425699` | 89 | 105 | 85 | node_limit | inf | 3179698977 | - | 24694 | 60.0 | **NO** | **NO** | **NO** |
| `neos-3072252-nete` | 432 | 576 | 144 | feasible | 13937034 | 11807698 | 2.38e-01 | 1074 | 60.2 | **NO** | **NO** | yes |
| `neos-3611689-kaihu` | 323 | 421 | 88 | feasible | 124 | 119 | 1.68e-01 | 1361 | 60.1 | **NO** | **NO** | yes |
| `neos-5140963-mincio` | 184 | 196 | 183 | feasible | 15079 | 14393 | 2.84e-01 | 14629 | 60.1 | **NO** | **NO** | yes |
| `neos-5192052-neckar` | 57 | 180 | 24 | optimal | -11670000 | -11670000 | 0.00e+00 | 9 | 0.1 | yes | yes | yes |
| `neos5` | 63 | 63 | 53 | feasible | 16 | 15 | 1.33e-01 | 27956 | 60.0 | **NO** | **NO** | yes |
| `noswot` | 182 | 128 | 100 | feasible | -39 | -41.00000885 | 1.03e-01 | 74098 | 60.1 | **NO** | **NO** | yes |
| `opt1217` | 64 | 769 | 768 | feasible | -16 | -16 | 2.51e-01 | 24208 | 60.0 | yes | **NO** | yes |
| `p0201` | 133 | 201 | 201 | feasible | 7615 | 7615 | 8.75e-05 | 2125 | 6.4 | yes | **NO** | yes |
| `pk1` | 45 | 86 | 55 | feasible | 19 | 11 | 8.29e-01 | 67782 | 60.0 | **NO** | **NO** | yes |
| `ran12x21` | 285 | 504 | 252 | feasible | 3795 | 3664 | 1.22e-01 | 7271 | 60.0 | **NO** | **NO** | yes |
| `ran13x13` | 195 | 338 | 169 | feasible | 3385 | 3252 | 1.31e-01 | 14442 | 60.0 | **NO** | **NO** | yes |
| `rlp1` | 68 | 461 | 450 | feasible | 15 | 15 | 1.17e-01 | 6621 | 60.0 | yes | **NO** | yes |
| `supportcase14` | 234 | 304 | 304 | optimal | 288 | 288 | 0.00e+00 | 141 | 0.6 | yes | yes | yes |
| `supportcase16` | 130 | 319 | 319 | optimal | 288 | 288 | 0.00e+00 | 257 | 0.6 | yes | yes | yes |
| `timtab1` | 171 | 397 | 171 | node_limit | inf | 764772 | - | 4810 | 60.0 | **NO** | **NO** | **NO** |

**Not proved optimal**, named rather than dropped: `b-ball`, `ej`, `enlight8`, `enlight_hard`, `flugpl`, `gen-ip016`, `gen-ip054`, `gt2`, `k16x240b`, `markshare1`, `markshare_4_0`, `markshare_5_0`, `neos-1425699`, `neos-3072252-nete`, `neos-3611689-kaihu`, `neos-5140963-mincio`, `neos5`, `noswot`, `opt1217`, `p0201`, `pk1`, `ran12x21`, `ran13x13`, `rlp1`, `timtab1`.

---

## 3. Correctness beyond the objective value

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

## 4. Comparison against an established solver

HiGHS is the reference. It runs as a SEPARATE PROCESS over the same MPS files; no HiGHS code
is linked into, or read by, SANKHYA - see `docs/PROVENANCE.md`. Both sides are timed on
solver-internal time only.

The comparison below is run on **the same tier as section 1b**, not on the eight-instance
demo set. Comparing only where we pass would be the easy version of this table and would say
nothing: the instances we fail are exactly the ones a reader should want to see against a
mature solver.

Source CSV: `bench/results/compare-highs-medium-a90db47.csv`  
Commit `a90db47` · machine `Windows-AMD64`

**45 of 50** instances where the two solvers agree on the objective.

Times are **solver-internal on both sides** - HiGHS's own `getRunTime()` against our `effort.solve_seconds` - so process start-up is excluded for both. At this instance size start-up would otherwise dominate and the comparison would measure the wrong thing entirely.

| instance | SANKHYA obj | HiGHS obj | agree | SANKHYA (s) | HiGHS (s) | ratio |
|---|---:|---:|:--:|---:|---:|---:|
| `adlittle` | 2.25494963e+05 | 2.25494963e+05 | yes | 0.004 | 0.013 | 0.30x |
| `afiro` | -4.64753143e+02 | -4.64753143e+02 | yes | 0.001 | 0.003 | 0.29x |
| `agg` | -3.59917673e+07 | -3.59917673e+07 | yes | 0.034 | 0.014 | 2.53x |
| `bandm` | -1.58628018e+02 | -1.58628018e+02 | yes | 0.049 | 0.018 | 2.74x |
| `beaconfd` | 3.35924858e+04 | 3.35924858e+04 | yes | 0.005 | 0.006 | 0.75x |
| `blend` | -3.08121498e+01 | -3.08121498e+01 | yes | 0.009 | 0.005 | 2.06x |
| `boeing1` | -3.35213568e+02 | -3.35213568e+02 | yes | 0.029 | 0.028 | 1.02x |
| `boeing2` | -3.15018728e+02 | -3.15018728e+02 | yes | 0.005 | 0.007 | 0.74x |
| `bore3d` | 1.37308039e+03 | 1.37308039e+03 | yes | 0.008 | 0.005 | 1.44x |
| `brandy` | 1.51850990e+03 | 1.51850990e+03 | yes | 0.015 | 0.013 | 1.17x |
| `capri` | 2.69001291e+03 | 2.69001291e+03 | yes | 0.020 | 0.008 | 2.64x |
| `d6cube` | 1.00000000e+00 | 3.15491667e+02 | **NO** | 1.831 | 0.463 | 3.95x |
| `degen2` | -1.43517800e+03 | -1.43517800e+03 | yes | 0.640 | 0.053 | 12.16x |
| `e226` | -1.16389291e+01 | -1.16389291e+01 | yes | 0.028 | 0.029 | 0.99x |
| `etamacro` | -7.55715233e+02 | -7.55715233e+02 | yes | 0.037 | 0.039 | 0.96x |
| `finnis` | 1.72791066e+05 | 1.72791066e+05 | yes | 0.039 | 0.021 | 1.86x |
| `fit1d` | -9.14637809e+03 | -9.14637809e+03 | yes | 0.079 | 0.044 | 1.82x |
| `fit2d` | -6.84642933e+04 | -6.84642933e+04 | yes | 27.962 | 0.715 | 39.09x |
| `forplan` | -6.64218961e+02 | -6.64218961e+02 | yes | 0.174 | 0.052 | 3.31x |
| `grow15` | 0.00000000e+00 | -1.06870941e+08 | **NO** | 0.221 | 0.176 | 1.25x |
| `grow22` | -1.60834336e+08 | -1.60834336e+08 | yes | 0.396 | 0.395 | 1.00x |
| `grow7` | -4.77878118e+07 | -4.77878118e+07 | yes | 0.034 | 0.046 | 0.74x |
| `israel` | -8.96644822e+05 | -8.96644822e+05 | yes | 0.045 | 0.016 | 2.86x |
| `kb2` | -1.74990013e+03 | -1.74990013e+03 | yes | 0.003 | 0.003 | 0.87x |
| `lotfi` | -2.52647061e+01 | -2.52647061e+01 | yes | 0.018 | 0.012 | 1.51x |
| `pilot4` | 0.00000000e+00 | -2.58113926e+03 | **NO** | 0.271 | 0.154 | 1.76x |
| `recipe` | -2.66616000e+02 | -2.66616000e+02 | yes | 0.003 | 0.006 | 0.46x |
| `sc105` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.011 | 0.006 | 1.77x |
| `sc205` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.027 | 0.009 | 2.98x |
| `sc50a` | -6.45750771e+01 | -6.45750771e+01 | yes | 0.002 | 0.003 | 0.82x |
| `sc50b` | -7.00000000e+01 | -7.00000000e+01 | yes | 0.003 | 0.004 | 0.70x |
| `scagr25` | -1.47534331e+07 | -1.47534331e+07 | yes | 0.136 | 0.028 | 4.86x |
| `scagr7` | -2.33138982e+06 | -2.33138982e+06 | yes | 0.005 | 0.015 | 0.35x |
| `scfxm1` | 1.84167590e+04 | 1.84167590e+04 | yes | 0.065 | 0.034 | 1.87x |
| `scorpion` | 1.87812482e+03 | 1.87812482e+03 | yes | 0.015 | 0.013 | 1.20x |
| `scrs8` | 9.04296954e+02 | 9.04296954e+02 | yes | 0.071 | 0.043 | 1.65x |
| `scsd1` | 8.66666667e+00 | 8.66666667e+00 | yes | 0.026 | 0.012 | 2.15x |
| `scsd6` | 5.05000001e+01 | 5.05000001e+01 | yes | 0.074 | 0.030 | 2.45x |
| `scsd8` | 9.05000000e+02 | 9.05000000e+02 | yes | 0.137 | 0.172 | 0.80x |
| `sctap1` | 1.41225000e+03 | 1.41225000e+03 | yes | 0.025 | 0.024 | 1.02x |
| `share1b` | -7.65893186e+04 | -7.65893186e+04 | yes | 0.005 | 0.014 | 0.33x |
| `share2b` | -4.15732241e+02 | -4.15732241e+02 | yes | 0.005 | 0.009 | 0.49x |
| `ship04l` | 1.79332454e+06 | 1.79332454e+06 | yes | 0.062 | 0.034 | 1.81x |
| `ship04s` | 1.79871470e+06 | 1.79871470e+06 | yes | 0.035 | 0.026 | 1.35x |
| `stair` | -2.51266951e+02 | -2.51266951e+02 | yes | 1.623 | 0.059 | 27.42x |
| `standata` | 1.25769950e+03 | 1.25769950e+03 | yes | 0.011 | 0.022 | 0.50x |
| `standmps` | 1.40601750e+03 | 1.40601750e+03 | yes | 0.018 | 0.024 | 0.78x |
| `stocfor1` | -4.11319762e+04 | -4.11319762e+04 | yes | 0.005 | 0.007 | 0.68x |
| `tuff` | 0.00000000e+00 | 2.92147765e-01 | **NO** | 0.192 | 0.035 | 5.45x |
| `wood1p` | 0.00000000e+00 | 1.44290241e+00 | **NO** | 0.795 | 0.276 | 2.88x |

**Summary**

- SANKHYA shifted geometric mean: **0.191s**
- HiGHS shifted geometric mean: **0.058s**
- SANKHYA is **3.3x** the HiGHS time by that measure

- per-instance ratio: median **1.40x**, worst **39.09x**, faster than HiGHS on **18 of 50** instances

We are **3.27x slower** than HiGHS by this measure, and publish that rather than bury it. HiGHS is a decade of specialist work with presolve, a dual simplex and a mature pricing scheme, and this solver still has neither of the first two. The part that has to be right first is that **the answers agree** - the problem statement asks us to compare, not to win.

---

## 5. What these numbers do not say

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

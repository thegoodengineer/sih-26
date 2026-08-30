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
of Netlib.** Its pass rate is not the headline; section 1c is.

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

### 1b. The medium tier — instances up to 500 rows

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

### 1c. The full set — the honest headline

Every instance in Netlib's summary table. Both tiers above are defined by a row cap, which
makes them the easier half of the library by construction; this is the number Phase 6's
">= 95% of Netlib" exit criterion is measured against, and the one the README quotes.

Source CSV: `bench/results/netlib-full-6a74e68.csv`  
Commit `6a74e68` · machine `Windows-AMD64` · generated 2026-08-30T13:02:40+00:00

**65 of 89 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **89 of the 89 instances** Netlib publishes an optimal value for (set `full`, selected by `fetch_data.py --set full`). Phase 6's "full Netlib >= 95%" exit criterion is measured against this set.

**24 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| basis went singular (#49) | 13 | 25fv47, bnl2, d2q06c, d6cube, greenbea, greenbeb, grow15, modszk1, perold, pilot, pilot4, pilot87, pilotnov |
| disagrees with the published optimum (#75) | 5 | 80bau3b, e226, ganges, nesm, scrs8 |
| hit the time limit | 3 | dfl001, fit2p, maros-r7 |
| point misses feasibility (#72) | 2 | agg3, grow7 |
| duals miss feasibility (#52) | 1 | etamacro |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `25fv47` | 821 | 1571 | numerical_error | 6.1601260000e+01 | 5.5018458883e+03 | 9.9e-01 | 3879 | 2.789 | - |
| `80bau3b` | 2262 | 9799 | optimal | 9.8722419241e+05 | 9.8723216072e+05 | 8.1e-06 | 10216 | 1.632 | yes |
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 158 | 0.016 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 16 | 0.013 | yes |
| `agg` | 488 | 163 | optimal | -3.5991767287e+07 | -3.5991767287e+07 | 1.2e-11 | 141 | 0.022 | yes |
| `agg2` | 516 | 302 | optimal | -2.0239252356e+07 | -2.0239252356e+07 | 1.1e-12 | 218 | 0.027 | yes |
| `agg3` | 516 | 302 | numerical_error | 1.0312115935e+07 | 1.0312115935e+07 | 8.7e-12 | 163 | 0.031 | - |
| `bandm` | 305 | 472 | optimal | -1.5862801845e+02 | -1.5862801845e+02 | 7.6e-13 | 574 | 0.032 | yes |
| `beaconfd` | 173 | 262 | optimal | 3.3592485807e+04 | 3.3592485807e+04 | 6.0e-12 | 96 | 0.018 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 337 | 0.018 | yes |
| `bnl1` | 643 | 1175 | optimal | 1.9776295615e+03 | 1.9776292856e+03 | 1.4e-07 | 3418 | 0.195 | yes |
| `bnl2` | 2324 | 3489 | numerical_error | 1.1489500000e+01 | 1.8112365404e+03 | 9.9e-01 | 6733 | 4.333 | - |
| `boeing1` | 351 | 384 | optimal | -3.3521356751e+02 | -3.3521356751e+02 | 8.6e-12 | 760 | 0.032 | yes |
| `boeing2` | 166 | 143 | optimal | -3.1501872802e+02 | -3.1501872802e+02 | 1.5e-11 | 208 | 0.020 | yes |
| `bore3d` | 233 | 315 | optimal | 1.3730803942e+03 | 1.3730803942e+03 | 6.2e-12 | 167 | 0.021 | yes |
| `brandy` | 220 | 249 | optimal | 1.5185098965e+03 | 1.5185098965e+03 | 7.8e-12 | 323 | 0.022 | yes |
| `capri` | 271 | 353 | optimal | 2.6900129138e+03 | 2.6900129138e+03 | 1.2e-11 | 458 | 0.027 | yes |
| `cycle` | 1903 | 2857 | optimal | -5.2263930249e+00 | -5.2263930249e+00 | 1.1e-12 | 2257 | 0.371 | yes |
| `czprob` | 929 | 3523 | optimal | 2.1851966989e+06 | 2.1851966989e+06 | 2.0e-11 | 1980 | 0.126 | yes |
| `d2q06c` | 2171 | 5167 | numerical_error | 1.1564832950e+03 | 1.2278423615e+05 | 9.9e-01 | 9340 | 4.258 | - |
| `d6cube` | 415 | 6184 | numerical_error | 1.0000000000e+00 | 3.1549166667e+02 | 1.0e+00 | 5230 | 1.175 | - |
| `degen2` | 444 | 534 | optimal | -1.4351780000e+03 | -1.4351780000e+03 | 0.0e+00 | 1539785 | 65.195 | yes |
| `degen3` | 1503 | 1818 | optimal | -9.8729400000e+02 | -9.8729400000e+02 | 4.6e-16 | 11209 | 123.592 | yes |
| `dfl001` | 6071 | 12230 | time_limit | 2.2973684694e+09 | 1.1266400000e+07 | 2.0e+02 | 28809 | 568.515 | - |
| `e226` | 223 | 282 | optimal | -1.1638929066e+01 | -1.8751929066e+01 | 3.8e-01 | 592 | 0.036 | yes |
| `etamacro` | 400 | 688 | feasible | -7.5571523316e+02 | -7.5571521774e+02 | 2.0e-08 | 661 | 0.038 | yes |
| `fffff800` | 524 | 854 | optimal | 5.5567956482e+05 | 5.5567961165e+05 | 8.4e-08 | 605 | 0.049 | yes |
| `finnis` | 497 | 614 | optimal | 1.7279106560e+05 | 1.7279096547e+05 | 5.8e-07 | 643 | 0.041 | yes |
| `fit1d` | 24 | 1026 | optimal | -9.1463780924e+03 | -9.1463780924e+03 | 2.3e-12 | 1622 | 0.081 | yes |
| `fit1p` | 627 | 1677 | optimal | 9.1463780924e+03 | 9.1463780924e+03 | 2.3e-12 | 2092 | 0.903 | yes |
| `fit2d` | 25 | 10500 | optimal | -6.8464293294e+04 | -6.8464293294e+04 | 2.4e-12 | 30210 | 9.109 | yes |
| `fit2p` | 3000 | 13525 | time_limit | 2.0320594492e+06 | 6.8464293232e+04 | 2.9e+01 | 11769 | 669.833 | - |
| `forplan` | 161 | 421 | optimal | -6.6421896127e+02 | -6.6421873953e+02 | 3.3e-07 | 1041 | 0.274 | yes |
| `ganges` | 1309 | 1681 | optimal | -1.0958573613e+05 | -1.0958636356e+05 | 5.7e-06 | 1529 | 0.378 | yes |
| `gfrd-pnc` | 616 | 1092 | optimal | 6.9022359995e+06 | 6.9022359995e+06 | 7.1e-12 | 903 | 0.157 | yes |
| `greenbea` | 2392 | 5405 | numerical_error | 2.0000000000e+02 | -7.2462405908e+07 | 1.0e+00 | 6198 | 5.153 | - |
| `greenbeb` | 2392 | 5405 | numerical_error | 2.0900000000e+02 | -4.3021476065e+06 | 1.0e+00 | 4549 | 4.240 | - |
| `grow15` | 300 | 645 | numerical_error | 0.0000000000e+00 | -1.0687094129e+08 | 1.0e+00 | 577 | 0.123 | - |
| `grow22` | 440 | 946 | optimal | -1.6083433648e+08 | -1.6083433648e+08 | 1.6e-11 | 1246 | 0.141 | yes |
| `grow7` | 140 | 301 | numerical_error | -4.7787811815e+07 | -4.7787811815e+07 | 6.0e-12 | 297 | 0.027 | - |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 251 | 0.027 | yes |
| `kb2` | 43 | 41 | optimal | -1.7499001299e+03 | -1.7499001299e+03 | 3.5e-12 | 71 | 0.022 | yes |
| `lotfi` | 153 | 308 | optimal | -2.5264706062e+01 | -2.5264706062e+01 | 4.7e-12 | 261 | 0.021 | yes |
| `maros` | 846 | 1443 | optimal | -5.8063743701e+04 | -5.8063743701e+04 | 2.2e-12 | 1692 | 0.175 | yes |
| `maros-r7` | 3136 | 9408 | time_limit | 2.0913959570e+11 | 1.4971851665e+06 | 1.4e+05 | 7177 | 240.304 | - |
| `modszk1` | 687 | 1620 | numerical_error | 0.0000000000e+00 | 3.2061972906e+02 | 1.0e+00 | 764045 | 59.937 | - |
| `nesm` | 662 | 2923 | optimal | 1.4076036488e+07 | 1.4076073035e+07 | 2.6e-06 | 5074 | 0.322 | yes |
| `perold` | 625 | 1376 | numerical_error | -6.3332914613e+02 | -9.3807580773e+03 | 9.3e-01 | 3981 | 0.990 | - |
| `pilot` | 1441 | 3652 | numerical_error | -8.5035600000e-01 | -5.5740430007e+02 | 1.0e+00 | 1515 | 2.025 | - |
| `pilot4` | 410 | 1000 | numerical_error | 0.0000000000e+00 | -2.5811392641e+03 | 1.0e+00 | 998 | 0.105 | - |
| `pilot87` | 2030 | 4883 | numerical_error | -1.2178000000e-03 | 3.0171072827e+02 | 1.0e+00 | 358 | 1.858 | - |
| `pilotnov` | 975 | 2172 | numerical_error | 0.0000000000e+00 | -4.4972761882e+03 | 1.0e+00 | 1308 | 0.235 | - |
| `recipe` | 91 | 180 | optimal | -2.6661600000e+02 | -2.6661600000e+02 | 1.1e-15 | 49 | 0.022 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 107 | 0.021 | yes |
| `sc205` | 205 | 203 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 255 | 0.033 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 46 | 0.015 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 4.1e-16 | 50 | 0.020 | yes |
| `scagr25` | 471 | 500 | optimal | -1.4753433061e+07 | -1.4753433061e+07 | 1.6e-11 | 591 | 0.048 | yes |
| `scagr7` | 129 | 140 | optimal | -2.3313898243e+06 | -2.3313892548e+06 | 2.4e-07 | 159 | 0.021 | yes |
| `scfxm1` | 330 | 457 | optimal | 1.8416759028e+04 | 1.8416759028e+04 | 1.9e-11 | 648 | 0.043 | yes |
| `scfxm2` | 660 | 914 | optimal | 3.6660261565e+04 | 3.6660261565e+04 | 3.3e-14 | 1071 | 0.071 | yes |
| `scfxm3` | 990 | 1371 | optimal | 5.4901254550e+04 | 5.4901254550e+04 | 4.5e-12 | 1595 | 0.131 | yes |
| `scorpion` | 388 | 358 | optimal | 1.8781248227e+03 | 1.8781248227e+03 | 2.0e-11 | 323 | 0.027 | yes |
| `scrs8` | 490 | 1169 | optimal | 9.0429695380e+02 | 9.0429998619e+02 | 3.4e-06 | 645 | 0.048 | yes |
| `scsd1` | 77 | 760 | optimal | 8.6666666743e+00 | 8.6666666743e+00 | 3.8e-12 | 534 | 0.031 | yes |
| `scsd6` | 147 | 1350 | optimal | 5.0500000078e+01 | 5.0500000078e+01 | 5.2e-12 | 627 | 0.049 | yes |
| `scsd8` | 397 | 2750 | optimal | 9.0499999993e+02 | 9.0499999993e+02 | 5.0e-12 | 926 | 0.098 | yes |
| `sctap1` | 300 | 480 | optimal | 1.4122500000e+03 | 1.4122500000e+03 | 0.0e+00 | 303 | 0.024 | yes |
| `sctap2` | 1090 | 1880 | optimal | 1.7248071429e+03 | 1.7248071429e+03 | 2.5e-11 | 720 | 0.059 | yes |
| `sctap3` | 1480 | 2480 | optimal | 1.4240000000e+03 | 1.4240000000e+03 | 0.0e+00 | 967 | 0.086 | yes |
| `seba` | 515 | 1028 | optimal | 1.5711600000e+04 | 1.5711600000e+04 | 1.2e-16 | 709 | 0.084 | yes |
| `share1b` | 117 | 225 | optimal | -7.6589318579e+04 | -7.6589318579e+04 | 2.4e-12 | 207 | 0.024 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 114 | 0.021 | yes |
| `shell` | 536 | 1775 | optimal | 1.2088253460e+09 | 1.2088253460e+09 | 0.0e+00 | 739 | 0.042 | yes |
| `ship04l` | 402 | 2118 | optimal | 1.7933245380e+06 | 1.7933245380e+06 | 1.7e-11 | 511 | 0.045 | yes |
| `ship04s` | 402 | 1458 | optimal | 1.7987147004e+06 | 1.7987147004e+06 | 2.5e-11 | 385 | 0.030 | yes |
| `ship08l` | 778 | 4283 | optimal | 1.9090552114e+06 | 1.9090552114e+06 | 5.7e-12 | 860 | 0.080 | yes |
| `ship08s` | 778 | 2387 | optimal | 1.9200982105e+06 | 1.9200982105e+06 | 1.8e-11 | 417 | 0.040 | yes |
| `ship12l` | 1151 | 5427 | optimal | 1.4701879193e+06 | 1.4701879193e+06 | 2.0e-11 | 1337 | 0.131 | yes |
| `ship12s` | 1151 | 2763 | optimal | 1.4892361344e+06 | 1.4892361344e+06 | 4.1e-12 | 691 | 0.051 | yes |
| `sierra` | 1227 | 2036 | optimal | 1.5394362184e+07 | 1.5394362184e+07 | 2.4e-11 | 706 | 0.069 | yes |
| `stair` | 356 | 467 | optimal | -2.5126695119e+02 | -2.5126695119e+02 | 1.2e-11 | 978 | 0.650 | yes |
| `standata` | 359 | 1075 | optimal | 1.2576995000e+03 | 1.2576995000e+03 | 0.0e+00 | 53 | 0.026 | yes |
| `standmps` | 467 | 1075 | optimal | 1.4060175000e+03 | 1.4060175000e+03 | 1.6e-16 | 193 | 0.027 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 68 | 0.021 | yes |
| `stocfor2` | 2157 | 2031 | optimal | -3.9024408538e+04 | -3.9024408538e+04 | 3.0e-12 | 2465 | 1.064 | yes |
| `tuff` | 333 | 587 | optimal | 2.9214776509e-01 | 2.9214776509e-01 | 3.6e-12 | 969 | 0.066 | yes |
| `wood1p` | 244 | 2594 | optimal | 1.4429024116e+00 | 1.4429024116e+00 | 1.8e-11 | 836 | 0.162 | yes |
| `woodw` | 1098 | 8405 | optimal | 1.3044763331e+00 | 1.3044763331e+00 | 1.2e-11 | 3190 | 0.574 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.301s**
- slowest solved instance: 123.592s
- worst relative error against a published optimum: **5.79e-07**
- **failed: `25fv47`, `80bau3b`, `agg3`, `bnl2`, `d2q06c`, `d6cube`, `dfl001`, `e226`, `etamacro`, `fit2p`, `ganges`, `greenbea`, `greenbeb`, `grow15`, `grow7`, `maros-r7`, `modszk1`, `nesm`, `perold`, `pilot`, `pilot4`, `pilot87`, `pilotnov`, `scrs8`** — kept in the table on purpose

---

## 2. MIPLIB — the mixed-integer side

The LP tiers above say nothing about the branch and bound. This is the MILP evidence, and it
is a harder library: MIPLIB instances are chosen to be difficult for mature solvers.

Source CSV: `bench/results/miplib-410789c.csv`  
Commit `410789c` · machine `Windows-AMD64`

**11 of 30** instances reached the published optimum. **6 of 30** also PROVED it - closed the bound rather than stopping at a gap target or a limit.

Those are different claims and are kept apart deliberately. Branch and bound here has no cutting planes - a rounding heuristic and a root dive, but nothing that tightens the relaxation - so it finds good incumbents far more often than it finishes the proof. Collapsing the two columns would hide exactly the thing #23 is meant to improve.

**The time limit decides some of these, not the solver.** `enlight8` proves optimality in about 55 seconds on an idle machine and misses a 60-second budget when the rest of the set is running alongside it - so its row moves with background load rather than with anything about the search. Instances close to the limit should be read as "needs more time than we gave it", not as a capability. The remedy is a longer limit, and the reason this table does not already use one is that the full set takes about half an hour per run as it stands.

Instances are the smallest MIPLIB 2017 instances tagged easy that carry a **proven** optimum (`=opt=` in MIPLIB's own solution file). A `=best=` value is the best anyone has found, not a proof, and scoring against one would let a wrong answer look like a record.

| instance | rows | cols | int | status | our objective | published | rel. gap | nodes | time (s) | matched | proved | verified |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| `b-ball` | 30 | 100 | 88 | feasible | -1.5 | -1.5 | 2.12e-01 | 38096 | 60.1 | yes | **NO** | yes |
| `ej` | 1 | 3 | 3 | feasible | 51015 | 25508 | 1.00e+00 | 70341 | 60.1 | **NO** | **NO** | yes |
| `enlight8` | 64 | 128 | 128 | node_limit | inf | 27 | - | 35778 | 60.1 | **NO** | **NO** | **NO** |
| `enlight_hard` | 100 | 200 | 200 | node_limit | inf | 37 | - | 25638 | 60.1 | **NO** | **NO** | **NO** |
| `f2gap40400` | 40 | 400 | 400 | optimal | 20772 | 20772 | 0.00e+00 | 509 | 2.6 | yes | yes | yes |
| `flugpl` | 18 | 18 | 11 | feasible | 1201500 | 1201500 | 8.74e-05 | 1323 | 0.1 | yes | **NO** | yes |
| `gen-ip016` | 24 | 28 | 28 | feasible | -9430.730724 | -9476.155197 | 7.48e-03 | 63830 | 60.0 | **NO** | **NO** | yes |
| `gen-ip054` | 27 | 30 | 30 | feasible | 6872.821391 | 6840.965642 | 1.33e-02 | 71695 | 60.0 | **NO** | **NO** | yes |
| `gr4x6` | 34 | 48 | 24 | optimal | 202.35 | 202.35 | 0.00e+00 | 142 | 0.1 | yes | yes | yes |
| `gt2` | 29 | 188 | 188 | feasible | 32523 | 21166 | 5.71e-01 | 119004 | 60.0 | **NO** | **NO** | yes |
| `k16x240b` | 256 | 480 | 240 | feasible | 12512 | 11393 | 4.75e-01 | 29929 | 60.0 | **NO** | **NO** | yes |
| `markshare1` | 6 | 62 | 50 | feasible | 28 | 1 | 1.00e+00 | 142973 | 60.0 | **NO** | **NO** | yes |
| `markshare_4_0` | 4 | 34 | 30 | feasible | 5 | 1 | 1.00e+00 | 203867 | 60.0 | **NO** | **NO** | yes |
| `markshare_5_0` | 5 | 45 | 40 | feasible | 15 | 1 | 1.00e+00 | 162195 | 60.0 | **NO** | **NO** | yes |
| `neos-1425699` | 89 | 105 | 85 | optimal | 3179698977 | 3179698977 | 0.00e+00 | 3 | 0.0 | yes | yes | yes |
| `neos-3072252-nete` | 432 | 576 | 144 | feasible | 13790610 | 11807698 | 2.29e-01 | 4559 | 60.0 | **NO** | **NO** | yes |
| `neos-3611689-kaihu` | 323 | 421 | 88 | feasible | 122 | 119 | 1.39e-01 | 6151 | 60.0 | **NO** | **NO** | yes |
| `neos-5140963-mincio` | 184 | 196 | 183 | feasible | 14817 | 14393 | 2.50e-01 | 33842 | 60.0 | **NO** | **NO** | yes |
| `neos-5192052-neckar` | 57 | 180 | 24 | optimal | -11670000 | -11670000 | 0.00e+00 | 9 | 0.0 | yes | yes | yes |
| `neos5` | 63 | 63 | 53 | feasible | 16 | 15 | 1.33e-01 | 38929 | 555.4 | **NO** | **NO** | yes |
| `noswot` | 182 | 128 | 100 | feasible | -39 | -41.00000885 | 1.03e-01 | 86192 | 60.4 | **NO** | **NO** | yes |
| `opt1217` | 64 | 769 | 768 | feasible | -16 | -16 | 2.51e-01 | 27644 | 60.0 | yes | **NO** | yes |
| `p0201` | 133 | 201 | 201 | feasible | 7615 | 7615 | 5.47e-05 | 1922 | 5.7 | yes | **NO** | yes |
| `pk1` | 45 | 86 | 55 | feasible | 19 | 11 | 8.27e-01 | 69012 | 60.0 | **NO** | **NO** | yes |
| `ran12x21` | 285 | 504 | 252 | feasible | 3795 | 3664 | 1.23e-01 | 6587 | 60.0 | **NO** | **NO** | yes |
| `ran13x13` | 195 | 338 | 169 | feasible | 3385 | 3252 | 1.33e-01 | 12145 | 60.0 | **NO** | **NO** | yes |
| `rlp1` | 68 | 461 | 450 | feasible | 15 | 15 | 1.17e-01 | 17256 | 60.0 | yes | **NO** | yes |
| `supportcase14` | 234 | 304 | 304 | optimal | 288 | 288 | 0.00e+00 | 131 | 0.7 | yes | yes | yes |
| `supportcase16` | 130 | 319 | 319 | optimal | 288 | 288 | 0.00e+00 | 258 | 0.6 | yes | yes | yes |
| `timtab1` | 171 | 397 | 171 | node_limit | inf | 764772 | - | 10330 | 60.0 | **NO** | **NO** | **NO** |

**Not proved optimal**, named rather than dropped: `b-ball`, `ej`, `enlight8`, `enlight_hard`, `flugpl`, `gen-ip016`, `gen-ip054`, `gt2`, `k16x240b`, `markshare1`, `markshare_4_0`, `markshare_5_0`, `neos-3072252-nete`, `neos-3611689-kaihu`, `neos-5140963-mincio`, `neos5`, `noswot`, `opt1217`, `p0201`, `pk1`, `ran12x21`, `ran13x13`, `rlp1`, `timtab1`.

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

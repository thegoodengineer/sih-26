# SANKHYA — benchmarks

<!-- GENERATED FILE. Do not edit by hand. -->
<!-- Regenerate with: python bench/runners/make_benchmarks_doc.py -->

This file is generated from the CSVs in `bench/results/`, so it cannot drift from the
evidence. Every number below came out of a run that recorded the instance sha256, the git
commit and the machine tag alongside it.

Times in sections 1a-1c and 2 are wall-clock, measured around the whole process, so they
include reading the model and writing the outputs. That makes them slightly pessimistic and
honest; it is not the figure to quote for algorithmic speed, and no attempt is made to
dress it up. Section 1d prints solver-internal seconds (its instances take minutes, and the
read is not what is being measured) and section 4 is solver-internal on both sides, as it
says.

Reporting follows Mittelmann's conventions: shifted geometric means with a
1-second shift, an explicit time limit, and failures counted and named
rather than dropped.

---

## 1. Netlib LP — accuracy against published optima

The reference optimum for each instance is parsed by `bench/runners/fetch_data.py` from
Netlib's own `readme`. None of these values was typed from memory.

### 1a. The small set — what the demo runs

Nine instances, committed to the repository so a fresh clone can reproduce this with no
network. **This is the set `demo/run_demo.sh` lets a judge pick from, and it is the easy end
of Netlib.** Its pass rate is not the headline; section 1c is.

Source CSV: `bench/results/netlib-small-53cbe16.csv`  
Commit `53cbe16` · machine `Windows-AMD64` · generated 2026-09-09T03:41:33+00:00

**9 of 9 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **9 of the 89 instances** Netlib publishes an optimal value for (set `small`, selected by `fetch_data.py --set small`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

Every instance in this set passed.

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 73 | 0.054 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 13 | 0.024 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 97 | 0.030 | yes |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 176 | 0.044 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 88 | 0.023 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 42 | 0.019 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 2.0e-16 | 42 | 0.029 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 133 | 0.034 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 101 | 0.027 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.031s**
- slowest solved instance: 0.054s
- worst relative error against a published optimum: **1.06e-11**
- no failures on this set

### 1b. The medium tier — instances up to 500 rows

Source CSV: `bench/results/netlib-medium-53cbe16.csv`  
Commit `53cbe16` · machine `Windows-AMD64` · generated 2026-09-09T03:36:04+00:00

**48 of 50 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **50 of the 89 instances** Netlib publishes an optimal value for (set `medium`, selected by `fetch_data.py --set medium`). Phase 6's "full Netlib >= 95%" exit criterion is measured against the full set, not against this one.

**2 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| disagrees with the published optimum (#75) | 2 | e226, scrs8 |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 73 | 0.397 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 13 | 0.040 | yes |
| `agg` | 488 | 163 | optimal | -3.5991767287e+07 | -3.5991767287e+07 | 1.2e-11 | 147 | 0.132 | yes |
| `bandm` | 305 | 472 | optimal | -1.5862801845e+02 | -1.5862801845e+02 | 7.6e-13 | 452 | 0.406 | yes |
| `beaconfd` | 173 | 262 | optimal | 3.3592485807e+04 | 3.3592485807e+04 | 6.0e-12 | 115 | 0.165 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 97 | 0.281 | yes |
| `boeing1` | 351 | 384 | optimal | -3.3521356751e+02 | -3.3521356751e+02 | 8.6e-12 | 357 | 0.223 | yes |
| `boeing2` | 166 | 143 | optimal | -3.1501872802e+02 | -3.1501872802e+02 | 1.5e-11 | 133 | 0.205 | yes |
| `bore3d` | 233 | 315 | optimal | 1.3730803942e+03 | 1.3730803942e+03 | 6.2e-12 | 161 | 0.145 | yes |
| `brandy` | 220 | 249 | optimal | 1.5185098965e+03 | 1.5185098965e+03 | 7.8e-12 | 282 | 0.244 | yes |
| `capri` | 271 | 353 | optimal | 2.6900129138e+03 | 2.6900129138e+03 | 1.2e-11 | 236 | 0.112 | yes |
| `d6cube` | 415 | 6184 | optimal | 3.1549166667e+02 | 3.1549166667e+02 | 1.1e-11 | 1013 | 2.035 | yes |
| `degen2` | 444 | 534 | optimal | -1.4351780000e+03 | -1.4351780000e+03 | 0.0e+00 | 639 | 0.653 | yes |
| `e226` | 223 | 282 | optimal | -1.1638929066e+01 | -1.8751929066e+01 | 3.8e-01 | 507 | 0.911 | yes |
| `etamacro` | 400 | 688 | optimal | -7.5571523312e+02 | -7.5571521774e+02 | 2.0e-08 | 705 | 0.420 | yes |
| `finnis` | 497 | 614 | optimal | 1.7279106560e+05 | 1.7279096547e+05 | 5.8e-07 | 403 | 0.307 | yes |
| `fit1d` | 24 | 1026 | optimal | -9.1463780924e+03 | -9.1463780924e+03 | 2.3e-12 | 56 | 0.292 | yes |
| `fit2d` | 25 | 10500 | optimal | -6.8464293294e+04 | -6.8464293294e+04 | 2.5e-12 | 224 | 2.146 | yes |
| `forplan` | 161 | 421 | optimal | -6.6421896127e+02 | -6.6421873953e+02 | 3.3e-07 | 309 | 0.669 | yes |
| `grow15` | 300 | 645 | optimal | -1.0687094129e+08 | -1.0687094129e+08 | 3.3e-11 | 3630 | 2.191 | yes |
| `grow22` | 440 | 946 | optimal | -1.6083433648e+08 | -1.6083433648e+08 | 1.6e-11 | 3853 | 4.883 | yes |
| `grow7` | 140 | 301 | optimal | -4.7787811815e+07 | -4.7787811815e+07 | 6.0e-12 | 1932 | 0.785 | yes |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 176 | 0.152 | yes |
| `kb2` | 43 | 41 | optimal | -1.7499001299e+03 | -1.7499001299e+03 | 3.5e-12 | 51 | 0.109 | yes |
| `lotfi` | 153 | 308 | optimal | -2.5264706062e+01 | -2.5264706062e+01 | 4.7e-12 | 243 | 0.107 | yes |
| `pilot4` | 410 | 1000 | optimal | -2.5811392589e+03 | -2.5811392641e+03 | 2.0e-09 | 1060 | 1.036 | yes |
| `recipe` | 91 | 180 | optimal | -2.6661600000e+02 | -2.6661600000e+02 | 1.1e-15 | 41 | 0.068 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 88 | 0.112 | yes |
| `sc205` | 205 | 203 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 209 | 0.061 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 42 | 0.098 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 2.0e-16 | 42 | 0.107 | yes |
| `scagr25` | 471 | 500 | optimal | -1.4753433061e+07 | -1.4753433061e+07 | 1.6e-11 | 465 | 0.146 | yes |
| `scagr7` | 129 | 140 | optimal | -2.3313898243e+06 | -2.3313892548e+06 | 2.4e-07 | 111 | 0.095 | yes |
| `scfxm1` | 330 | 457 | optimal | 1.8416759028e+04 | 1.8416759028e+04 | 1.9e-11 | 460 | 0.174 | yes |
| `scorpion` | 388 | 358 | optimal | 1.8781248227e+03 | 1.8781248227e+03 | 2.0e-11 | 242 | 0.207 | yes |
| `scrs8` | 490 | 1169 | optimal | 9.0429695380e+02 | 9.0429998619e+02 | 3.4e-06 | 610 | 0.241 | yes |
| `scsd1` | 77 | 760 | optimal | 8.6666666743e+00 | 8.6666666743e+00 | 3.8e-12 | 155 | 0.179 | yes |
| `scsd6` | 147 | 1350 | optimal | 5.0500000077e+01 | 5.0500000078e+01 | 1.7e-11 | 367 | 0.165 | yes |
| `scsd8` | 397 | 2750 | optimal | 9.0499999993e+02 | 9.0499999993e+02 | 5.0e-12 | 1785 | 1.129 | yes |
| `sctap1` | 300 | 480 | optimal | 1.4122500000e+03 | 1.4122500000e+03 | 0.0e+00 | 269 | 0.109 | yes |
| `share1b` | 117 | 225 | optimal | -7.6589318579e+04 | -7.6589318579e+04 | 2.4e-12 | 136 | 0.206 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 133 | 0.160 | yes |
| `ship04l` | 402 | 2118 | optimal | 1.7933245380e+06 | 1.7933245380e+06 | 1.7e-11 | 449 | 0.441 | yes |
| `ship04s` | 402 | 1458 | optimal | 1.7987147004e+06 | 1.7987147004e+06 | 2.5e-11 | 300 | 0.336 | yes |
| `stair` | 356 | 467 | optimal | -2.5126695119e+02 | -2.5126695119e+02 | 1.2e-11 | 448 | 0.551 | yes |
| `standata` | 359 | 1075 | optimal | 1.2576995000e+03 | 1.2576995000e+03 | 1.8e-16 | 58 | 0.083 | yes |
| `standmps` | 467 | 1075 | optimal | 1.4060175000e+03 | 1.4060175000e+03 | 1.6e-16 | 200 | 0.100 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 101 | 0.095 | yes |
| `tuff` | 333 | 587 | optimal | 2.9214776509e-01 | 2.9214776509e-01 | 3.6e-12 | 239 | 0.133 | yes |
| `wood1p` | 244 | 2594 | optimal | 1.4429024116e+00 | 1.4429024116e+00 | 1.8e-11 | 406 | 1.046 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.382s**
- slowest solved instance: 4.883s
- worst relative error against a published optimum: **5.79e-07**
- **failed: `e226`, `scrs8`** — kept in the table on purpose

### 1c. The full set — the honest headline

Every instance in Netlib's summary table. Both tiers above are defined by a row cap, which
makes them the easier half of the library by construction; this is the number Phase 6's
">= 95% of Netlib" exit criterion is measured against, and the one the README quotes.

Source CSV: `bench/results/netlib-full-53cbe16.csv`  
Commit `53cbe16` · machine `Windows-AMD64` · generated 2026-09-08T11:58:38+00:00

**78 of 89 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

Coverage: this run used **89 of the 89 instances** Netlib publishes an optimal value for (set `full`, selected by `fetch_data.py --set full`). Phase 6's "full Netlib >= 95%" exit criterion is measured against this set.

**11 failed**, grouped by the reason the solver itself gave. They are named here because a pass rate without its failures is a claim, not evidence:

| why it failed | count | instances |
|---|---:|---|
| disagrees with the published optimum (#75) | 7 | 80bau3b, e226, ganges, greenbea, greenbeb, nesm, scrs8 |
| hit the time limit | 2 | dfl001, pilot87 |
| duals miss feasibility (#52) | 1 | pilot |
| numerical_error | 1 | maros-r7 |

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `25fv47` | 821 | 1571 | optimal | 5.5018458883e+03 | 5.5018458883e+03 | 2.4e-12 | 4462 | 2.462 | yes |
| `80bau3b` | 2262 | 9799 | optimal | 9.8722419241e+05 | 9.8723216072e+05 | 8.1e-06 | 4031 | 0.988 | yes |
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 73 | 0.020 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 13 | 0.018 | yes |
| `agg` | 488 | 163 | optimal | -3.5991767287e+07 | -3.5991767287e+07 | 1.2e-11 | 147 | 0.028 | yes |
| `agg2` | 516 | 302 | optimal | -2.0239252356e+07 | -2.0239252356e+07 | 1.1e-12 | 161 | 0.031 | yes |
| `agg3` | 516 | 302 | optimal | 1.0312115935e+07 | 1.0312115935e+07 | 8.7e-12 | 166 | 0.029 | yes |
| `bandm` | 305 | 472 | optimal | -1.5862801845e+02 | -1.5862801845e+02 | 7.6e-13 | 452 | 0.035 | yes |
| `beaconfd` | 173 | 262 | optimal | 3.3592485807e+04 | 3.3592485807e+04 | 6.0e-12 | 115 | 0.020 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 97 | 0.017 | yes |
| `bnl1` | 643 | 1175 | optimal | 1.9776295615e+03 | 1.9776292856e+03 | 1.4e-07 | 1620 | 0.118 | yes |
| `bnl2` | 2324 | 3489 | optimal | 1.8112365404e+03 | 1.8112365404e+03 | 2.3e-11 | 2291 | 0.414 | yes |
| `boeing1` | 351 | 384 | optimal | -3.3521356751e+02 | -3.3521356751e+02 | 8.6e-12 | 357 | 0.032 | yes |
| `boeing2` | 166 | 143 | optimal | -3.1501872802e+02 | -3.1501872802e+02 | 1.5e-11 | 133 | 0.020 | yes |
| `bore3d` | 233 | 315 | optimal | 1.3730803942e+03 | 1.3730803942e+03 | 6.2e-12 | 161 | 0.022 | yes |
| `brandy` | 220 | 249 | optimal | 1.5185098965e+03 | 1.5185098965e+03 | 7.8e-12 | 282 | 0.029 | yes |
| `capri` | 271 | 353 | optimal | 2.6900129138e+03 | 2.6900129138e+03 | 1.2e-11 | 236 | 0.023 | yes |
| `cycle` | 1903 | 2857 | optimal | -5.2263930249e+00 | -5.2263930249e+00 | 1.1e-12 | 310 | 0.079 | yes |
| `czprob` | 929 | 3523 | optimal | 2.1851966989e+06 | 2.1851966989e+06 | 2.0e-11 | 996 | 0.131 | yes |
| `d2q06c` | 2171 | 5167 | optimal | 1.2278421081e+05 | 1.2278423615e+05 | 2.1e-07 | 31857 | 22.802 | yes |
| `d6cube` | 415 | 6184 | optimal | 3.1549166667e+02 | 3.1549166667e+02 | 1.1e-11 | 1013 | 0.411 | yes |
| `degen2` | 444 | 534 | optimal | -1.4351780000e+03 | -1.4351780000e+03 | 0.0e+00 | 639 | 0.057 | yes |
| `degen3` | 1503 | 1818 | optimal | -9.8729400000e+02 | -9.8729400000e+02 | 2.3e-16 | 3276 | 0.875 | yes |
| `dfl001` | 6071 | 12230 | time_limit | 1.0568942128e+07 | 1.1266400000e+07 | 6.2e-02 | 28460 | 120.092 | - |
| `e226` | 223 | 282 | optimal | -1.1638929066e+01 | -1.8751929066e+01 | 3.8e-01 | 507 | 0.041 | yes |
| `etamacro` | 400 | 688 | optimal | -7.5571523312e+02 | -7.5571521774e+02 | 2.0e-08 | 705 | 0.058 | yes |
| `fffff800` | 524 | 854 | optimal | 5.5567956482e+05 | 5.5567961165e+05 | 8.4e-08 | 718 | 0.063 | yes |
| `finnis` | 497 | 614 | optimal | 1.7279106560e+05 | 1.7279096547e+05 | 5.8e-07 | 403 | 0.037 | yes |
| `fit1d` | 24 | 1026 | optimal | -9.1463780924e+03 | -9.1463780924e+03 | 2.3e-12 | 56 | 0.036 | yes |
| `fit1p` | 627 | 1677 | optimal | 9.1463780924e+03 | 9.1463780924e+03 | 2.3e-12 | 1196 | 0.282 | yes |
| `fit2d` | 25 | 10500 | optimal | -6.8464293294e+04 | -6.8464293294e+04 | 2.5e-12 | 224 | 0.297 | yes |
| `fit2p` | 3000 | 13525 | optimal | 6.8464293294e+04 | 6.8464293232e+04 | 9.0e-10 | 10432 | 58.474 | yes |
| `forplan` | 161 | 421 | optimal | -6.6421896127e+02 | -6.6421873953e+02 | 3.3e-07 | 309 | 0.032 | yes |
| `ganges` | 1309 | 1681 | optimal | -1.0958573613e+05 | -1.0958636356e+05 | 5.7e-06 | 1083 | 0.095 | yes |
| `gfrd-pnc` | 616 | 1092 | optimal | 6.9022359995e+06 | 6.9022359995e+06 | 7.1e-12 | 448 | 0.041 | yes |
| `greenbea` | 2392 | 5405 | optimal | -7.2555248130e+07 | -7.2462405908e+07 | 1.3e-03 | 15843 | 6.675 | yes |
| `greenbeb` | 2392 | 5405 | optimal | -4.3022602612e+06 | -4.3021476065e+06 | 2.6e-05 | 9314 | 2.709 | yes |
| `grow15` | 300 | 645 | optimal | -1.0687094129e+08 | -1.0687094129e+08 | 3.3e-11 | 3630 | 0.240 | yes |
| `grow22` | 440 | 946 | optimal | -1.6083433648e+08 | -1.6083433648e+08 | 1.6e-11 | 3853 | 0.325 | yes |
| `grow7` | 140 | 301 | optimal | -4.7787811815e+07 | -4.7787811815e+07 | 6.0e-12 | 1932 | 0.110 | yes |
| `israel` | 174 | 142 | optimal | -8.9664482186e+05 | -8.9664482186e+05 | 3.4e-12 | 176 | 0.028 | yes |
| `kb2` | 43 | 41 | optimal | -1.7499001299e+03 | -1.7499001299e+03 | 3.5e-12 | 51 | 0.016 | yes |
| `lotfi` | 153 | 308 | optimal | -2.5264706062e+01 | -2.5264706062e+01 | 4.7e-12 | 243 | 0.021 | yes |
| `maros` | 846 | 1443 | optimal | -5.8063743701e+04 | -5.8063743701e+04 | 2.2e-12 | 2007 | 0.226 | yes |
| `maros-r7` | 3136 | 9408 | numerical_error | 0.0000000000e+00 | 1.4971851665e+06 | 1.0e+00 | 655 | 5.216 | - |
| `modszk1` | 687 | 1620 | optimal | 3.2061972906e+02 | 3.2061972906e+02 | 1.3e-11 | 658 | 0.062 | yes |
| `nesm` | 662 | 2923 | optimal | 1.4076036488e+07 | 1.4076073035e+07 | 2.6e-06 | 2457 | 0.266 | yes |
| `perold` | 625 | 1376 | optimal | -9.3807552782e+03 | -9.3807580773e+03 | 3.0e-07 | 4953 | 0.587 | yes |
| `pilot` | 1441 | 3652 | feasible | -5.5748956079e+02 | -5.5740430007e+02 | 1.5e-04 | 16753 | 12.812 | yes |
| `pilot4` | 410 | 1000 | optimal | -2.5811392589e+03 | -2.5811392641e+03 | 2.0e-09 | 1060 | 0.102 | yes |
| `pilot87` | 2030 | 4883 | time_limit | 3.3124360035e+02 | 3.0171072827e+02 | 9.8e-02 | 25375 | 120.162 | - |
| `pilotnov` | 975 | 2172 | optimal | -4.4972761882e+03 | -4.4972761882e+03 | 4.2e-12 | 3965 | 1.888 | yes |
| `recipe` | 91 | 180 | optimal | -2.6661600000e+02 | -2.6661600000e+02 | 1.1e-15 | 41 | 0.038 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 88 | 0.176 | yes |
| `sc205` | 205 | 203 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 209 | 0.143 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 42 | 0.097 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 2.0e-16 | 42 | 0.046 | yes |
| `scagr25` | 471 | 500 | optimal | -1.4753433061e+07 | -1.4753433061e+07 | 1.6e-11 | 465 | 0.586 | yes |
| `scagr7` | 129 | 140 | optimal | -2.3313898243e+06 | -2.3313892548e+06 | 2.4e-07 | 111 | 0.316 | yes |
| `scfxm1` | 330 | 457 | optimal | 1.8416759028e+04 | 1.8416759028e+04 | 1.9e-11 | 460 | 0.430 | yes |
| `scfxm2` | 660 | 914 | optimal | 3.6660261565e+04 | 3.6660261565e+04 | 3.3e-14 | 945 | 0.224 | yes |
| `scfxm3` | 990 | 1371 | optimal | 5.4901254550e+04 | 5.4901254550e+04 | 4.5e-12 | 1577 | 0.655 | yes |
| `scorpion` | 388 | 358 | optimal | 1.8781248227e+03 | 1.8781248227e+03 | 2.0e-11 | 242 | 0.118 | yes |
| `scrs8` | 490 | 1169 | optimal | 9.0429695380e+02 | 9.0429998619e+02 | 3.4e-06 | 610 | 0.088 | yes |
| `scsd1` | 77 | 760 | optimal | 8.6666666743e+00 | 8.6666666743e+00 | 3.8e-12 | 155 | 0.046 | yes |
| `scsd6` | 147 | 1350 | optimal | 5.0500000077e+01 | 5.0500000078e+01 | 1.7e-11 | 367 | 0.061 | yes |
| `scsd8` | 397 | 2750 | optimal | 9.0499999993e+02 | 9.0499999993e+02 | 5.0e-12 | 1785 | 0.247 | yes |
| `sctap1` | 300 | 480 | optimal | 1.4122500000e+03 | 1.4122500000e+03 | 0.0e+00 | 269 | 0.034 | yes |
| `sctap2` | 1090 | 1880 | optimal | 1.7248071429e+03 | 1.7248071429e+03 | 2.5e-11 | 738 | 0.092 | yes |
| `sctap3` | 1480 | 2480 | optimal | 1.4240000000e+03 | 1.4240000000e+03 | 0.0e+00 | 1076 | 0.145 | yes |
| `seba` | 515 | 1028 | optimal | 1.5711600000e+04 | 1.5711600000e+04 | 1.2e-16 | 439 | 0.048 | yes |
| `share1b` | 117 | 225 | optimal | -7.6589318579e+04 | -7.6589318579e+04 | 2.4e-12 | 136 | 0.022 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 133 | 0.025 | yes |
| `shell` | 536 | 1775 | optimal | 1.2088253460e+09 | 1.2088253460e+09 | 0.0e+00 | 457 | 0.046 | yes |
| `ship04l` | 402 | 2118 | optimal | 1.7933245380e+06 | 1.7933245380e+06 | 1.7e-11 | 449 | 0.050 | yes |
| `ship04s` | 402 | 1458 | optimal | 1.7987147004e+06 | 1.7987147004e+06 | 2.5e-11 | 300 | 0.035 | yes |
| `ship08l` | 778 | 4283 | optimal | 1.9090552114e+06 | 1.9090552114e+06 | 5.7e-12 | 743 | 0.099 | yes |
| `ship08s` | 778 | 2387 | optimal | 1.9200982105e+06 | 1.9200982105e+06 | 1.8e-11 | 428 | 0.050 | yes |
| `ship12l` | 1151 | 5427 | optimal | 1.4701879193e+06 | 1.4701879193e+06 | 2.0e-11 | 1065 | 0.145 | yes |
| `ship12s` | 1151 | 2763 | optimal | 1.4892361344e+06 | 1.4892361344e+06 | 4.1e-12 | 584 | 0.062 | yes |
| `sierra` | 1227 | 2036 | optimal | 1.5394362184e+07 | 1.5394362184e+07 | 2.4e-11 | 552 | 0.084 | yes |
| `stair` | 356 | 467 | optimal | -2.5126695119e+02 | -2.5126695119e+02 | 1.2e-11 | 448 | 0.057 | yes |
| `standata` | 359 | 1075 | optimal | 1.2576995000e+03 | 1.2576995000e+03 | 1.8e-16 | 58 | 0.027 | yes |
| `standmps` | 467 | 1075 | optimal | 1.4060175000e+03 | 1.4060175000e+03 | 1.6e-16 | 200 | 0.038 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 101 | 0.020 | yes |
| `stocfor2` | 2157 | 2031 | optimal | -3.9024408538e+04 | -3.9024408538e+04 | 3.0e-12 | 1839 | 0.298 | yes |
| `tuff` | 333 | 587 | optimal | 2.9214776509e-01 | 2.9214776509e-01 | 3.6e-12 | 239 | 0.038 | yes |
| `wood1p` | 244 | 2594 | optimal | 1.4429024116e+00 | 1.4429024116e+00 | 1.8e-11 | 406 | 0.204 | yes |
| `woodw` | 1098 | 8405 | optimal | 1.3044763331e+00 | 1.3044763331e+00 | 1.2e-11 | 2955 | 1.224 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.273s**
- slowest solved instance: 58.474s
- worst relative error against a published optimum: **5.79e-07**
- **failed: `80bau3b`, `dfl001`, `e226`, `ganges`, `greenbea`, `greenbeb`, `maros-r7`, `nesm`, `pilot`, `pilot87`, `scrs8`** — kept in the table on purpose

### 1d. Beyond Netlib — Mittelmann's LP set

Netlib's largest instance has about 6,000 rows. PS26119 asks about "thousands to millions
of variables", and the only honest way to say where this solver stands on that is to run
instances of that size and name what happens. These are the eight smallest archives in
Mittelmann's LP test set (`bench/runners/fetch_mittelmann.py`, provenance in
`data/mittelmann/reference.json`).

Source CSV: `bench/results/mittelmann-592aea3.csv`  
Commit `592aea3` · machine `Windows-AMD64` · time limit 300 s per instance, both solvers

**0 of 8** instances reached `optimal` inside the limit; **0 of 8** also passed the independent verifier and agree with HiGHS. HiGHS, run as a separate process under the same limit, finished **2 of 8**.

These are the smallest archives in Mittelmann's LP directory; against Netlib's largest instance (dfl001, 6,071 rows, 35,632 nonzeros) they range from the same row count with 2.7x the nonzeros (qap15) to 62x the rows and 42x the nonzeros (bdry2). No published optimum exists for them, so there is no pass-against-a-number column: the outcome is the status, the verifier's verdict where a solution was written, and HiGHS's objective where HiGHS finished. `our objective` on a `time_limit` row is the last iterate's value, not a bound, and is printed only so that a later run can be compared with it.

| instance | rows | cols | nonzeros | status | our objective | HiGHS objective | rel. diff | iters | solver time (s) | verified |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `Linf_520c` | 93326 | 69004 | 566193 | time_limit | 0.1049570954 | Time limit reached | - | 588 | 308.9 | - |
| `bdry2` | 376500 | 250998 | 1500003 | time_limit | -0.0043 | Time limit reached | - | 1 | 648.7 | - |
| `brazil3` | 14646 | 23968 | 133184 | time_limit | 0 | 2 | - | 10922 | 300.5 | - |
| `chromaticindex1024-7` | 67583 | 73728 | 270324 | time_limit | 2 | Time limit reached | - | 1302 | 303.5 | - |
| `irish-electricity` | 104259 | 61728 | 523257 | time_limit | 0 | 2546254.563 | - | 3121 | 304.0 | - |
| `qap15` | 6330 | 22275 | 94950 | numerical_error | 0 | Time limit reached | - | 13954 | 297.5 | - |
| `rmine15` | 358395 | 42438 | 879732 | time_limit | -8443.961371 | Time limit reached | - | 28 | 309.2 | - |
| `supportcase10` | 165684 | 14770 | 555082 | time_limit | 0 | Time limit reached | - | 1266 | 306.2 | - |

**Not solved inside the limit**, named rather than dropped: `Linf_520c`, `bdry2`, `brazil3`, `chromaticindex1024-7`, `irish-electricity`, `qap15`, `rmine15`, `supportcase10`.

### 1e. The first-order engine — PDHG

The simplex is not the only continuous engine. Restarted PDHG (`--option algorithm=pdhg`) is
a first-order method: no basis, no factorization, and a cost that depends enormously on the
accuracy asked of it - which is why this section reports two tolerances separately rather
than one blended number. It is also the engine the GPU work targets, so its CPU behaviour is
the baseline every GPU claim will be measured against.

Source CSV: `bench/results/pdhg-53cbe16.csv`  
Commit `53cbe16` · machine `Windows-AMD64` · 9 instances, the ones committed to the repository

- **8 of 9** reach `optimal` at a requested 0.0001 with restarts on.
- **8 of 9** reach `optimal` at a requested 1e-08 with restarts on, **7 of 9** with restarts off.

`optimal` here means what it means everywhere else in this document: the point also survives the project's absolute tolerances, not merely the relative ones the first-order loop converges on. That distinction is the whole of #179 - the loop used to stop on the relative measure and the report then downgraded the point it stopped on, so the engine gave up early and handed back the weaker answer.

**Read the two tolerance columns together, because they are the same run.** Since #179 the loop stops only where the absolute standard is met, so a request looser than that standard no longer stops the solve any earlier - ask for 1e-4 and you get the 1e-8 point, at the 1e-8 cost. That is the honest reading of the identical columns below, and it is a real trade: the old behaviour honoured a loose request and returned a point it then had to label `feasible`. #180 made that the opt-in: `--option pdhg_stop_at_request=true` waives the dual, gap and complementarity halves of the standard - absolute primal feasibility is kept, so `feasible` still means a feasible point - and reports the point as `feasible` unless it meets the full standard anyway. Measured on these instances at 1e-4 it costs 0.85x the iterations (`bench/results/pdhg-stop-at-request-f18d4b0.csv`) and turns `share2b` from an iteration limit into a usable point at 807,760. The two tolerance columns stay identical on `adlittle`, `israel` and `sc50b` even with the switch on, because on those the kept primal clause is what binds.

| instance | simplex | PDHG 0.0001: objective / iterations | PDHG 1e-08: objective / iterations |
|---|---:|---:|---:|
| `adlittle` | 225494.9632 | 225494.9632 / 192080 | 225494.9632 / 192080 |
| `afiro` | -464.7531429 | -464.7531428 / 1040 | -464.7531428 / 1040 |
| `blend` | -30.81214985 | -30.81214988 / 44520 | -30.81214988 / 44520 |
| `israel` | -896644.8219 | -896644.8219 / 368720 | -896644.8219 / 368720 |
| `sc105` | -52.20206121 | -52.20206122 / 61440 | -52.20206122 / 61440 |
| `sc50a` | -64.57507706 | -64.57507705 / 7680 | -64.57507705 / 7680 |
| `sc50b` | -70 | -69.99999999 / 8360 | -69.99999999 / 8360 |
| `share2b` | -415.7322407 | -415.7322735 / 1000000 (iteration_limit) | -415.7322735 / 1000000 (iteration_limit) |
| `stocfor1` | -41131.97622 | -41131.97619 / 321000 | -41131.97619 / 321000 |

**Restarts, measured at 1e-08.** The claim that restarting the averaging helps is checked rather than repeated:

| instance | restarts on | restarts off | ratio |
|---|---:|---:|---:|
| `adlittle` | 192080 | 207040 | 1.08x |
| `afiro` | 1040 | 2800 | 2.69x |
| `blend` | 44520 | 76400 | 1.72x |
| `israel` | 368720 | 1000000 | 2.71x |
| `sc105` | 61440 | 291200 | 4.74x |
| `sc50a` | 7680 | 28280 | 3.68x |
| `sc50b` | 8360 | 33200 | 3.97x |
| `share2b` | 1000000 | 1000000 | 1.00x |
| `stocfor1` | 321000 | 496320 | 1.55x |

A ratio above 1 means restarts saved iterations on that instance.

---

## 2. MIPLIB — the mixed-integer side

The LP tiers above say nothing about the branch and bound. This is the MILP evidence, and it
is a harder library: MIPLIB instances are chosen to be difficult for mature solvers.

Source CSV: `bench/results/miplib-53cbe16.csv`  
Commit `53cbe16` · machine `Windows-AMD64`

**13 of 30** instances reached the published optimum. **6 of 30** also PROVED it - closed the bound to within the requested gap target rather than stopping at a node or time limit.

**This CSV predates #188.** 3 of these rows stopped on the gap target and were recorded `feasible`, so they are counted above as NOT proved: `f2gap40400`, `flugpl`, `p0201`. Since #188 such a stop reports `optimal` - the incumbent is within the tolerance the caller asked for, which is what the word means everywhere else in the field - so a rerun would count them as proved. That is a renamed status, not a better search, and the number above is left as the run measured it.
Those are different claims and are kept apart deliberately. Branch and bound here finds good incumbents far more often than it finishes the proof: reliability branching (#69) and warm-started dual node LPs (#65) do the searching, and the root cutting planes that exist (#159: Gomory mixed-integer and lifted knapsack cover) are off by default, for the reason measured below. Collapsing the two columns would hide exactly the thing cuts are meant to improve.

**Root cuts, on versus off** (`bench/results/miplib-cuts-off.csv` and `miplib-cuts-on.csv`, both at `adf4f20` on the PR branch, 30 instances, the same time limit): with cuts on, 12 of 30 reach the published optimum and 5 prove it, against 12 and 7 with them off. Over the 28 instances that end the same way either way, the cuts take the total node count to 0.887x (per instance from 0.257x to 1.209x). The outcome changed on 2: `enlight8` optimal -> node limit (stopped at the time limit after 60.00s and 51627 nodes); `f2gap40400` optimal -> feasible (stopped on a relative gap target (2.043e+00 absolute, 9.835e-05 relative) after 321 nodes). A cut row makes every node LP dearer, so at this limit the cuts buy nodes and cost proofs, and a run that reaches the gap target with them stops as `feasible` where the run without them exhausted its tree. That is why `enable_root_cuts` is off by default: a measurement, not caution.

**The time limit decides some of these, not the solver.** A row that stops at the limit with a small gap says "needs more time than we gave it", not "cannot"; which side of the limit such a row lands on moves with the machine's speed rather than with anything about the search. The remedy is a longer limit, and the reason this table does not already use one is that the set already adds up to 21 minutes of solve time per run at this one.

Instances are the smallest MIPLIB 2017 instances tagged easy that carry a **proven** optimum (`=opt=` in MIPLIB's own solution file). A `=best=` value is the best anyone has found, not a proof, and scoring against one would let a wrong answer look like a record.

| instance | rows | cols | int | status | our objective | published | rel. gap | nodes | time (s) | matched | proved | verified |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| `b-ball` | 30 | 100 | 88 | feasible | -1.5 | -1.5 | 2.12e-01 | 71480 | 60.3 | yes | **NO** | yes |
| `ej` | 1 | 3 | 3 | feasible | 51015 | 25508 | 1.00e+00 | 65307 | 60.1 | **NO** | **NO** | yes |
| `enlight8` | 64 | 128 | 128 | node_limit | inf | 27 | - | 122275 | 60.1 | **NO** | **NO** | **NO** |
| `enlight_hard` | 100 | 200 | 200 | node_limit | inf | 37 | - | 102964 | 60.0 | **NO** | **NO** | **NO** |
| `f2gap40400` | 40 | 400 | 400 | feasible | 20772 | 20772 | 5.23e-05 | 543 | 3.9 | yes | **NO** | yes |
| `flugpl` | 18 | 18 | 11 | feasible | 1201500 | 1201500 | 1.87e-05 | 545 | 0.1 | yes | **NO** | yes |
| `gen-ip016` | 24 | 28 | 28 | feasible | -9431.212409 | -9476.155197 | 7.38e-03 | 76469 | 60.0 | **NO** | **NO** | yes |
| `gen-ip054` | 27 | 30 | 30 | feasible | 6859.867147 | 6840.965642 | 1.08e-02 | 90620 | 60.0 | **NO** | **NO** | yes |
| `gr4x6` | 34 | 48 | 24 | optimal | 202.35 | 202.35 | 0.00e+00 | 71 | 0.1 | yes | yes | yes |
| `gt2` | 29 | 188 | 188 | optimal | 21166 | 21166 | 0.00e+00 | 478 | 0.2 | yes | yes | yes |
| `k16x240b` | 256 | 480 | 240 | feasible | 12066 | 11393 | 4.23e-01 | 78287 | 60.1 | **NO** | **NO** | yes |
| `markshare1` | 6 | 62 | 50 | feasible | 40 | 1 | 1.00e+00 | 85564 | 60.0 | **NO** | **NO** | yes |
| `markshare_4_0` | 4 | 34 | 30 | feasible | 4 | 1 | 1.00e+00 | 117649 | 60.1 | **NO** | **NO** | yes |
| `markshare_5_0` | 5 | 45 | 40 | feasible | 27 | 1 | 1.00e+00 | 94708 | 60.0 | **NO** | **NO** | yes |
| `neos-1425699` | 89 | 105 | 85 | optimal | 3179698977 | 3179698977 | 0.00e+00 | 3 | 0.0 | yes | yes | yes |
| `neos-3072252-nete` | 432 | 576 | 144 | feasible | 11928124 | 11807698 | 1.10e-01 | 13549 | 60.0 | **NO** | **NO** | yes |
| `neos-3611689-kaihu` | 323 | 421 | 88 | feasible | 120 | 119 | 6.38e-02 | 26355 | 60.8 | **NO** | **NO** | yes |
| `neos-5140963-mincio` | 184 | 196 | 183 | feasible | 15178 | 14393 | 2.53e-01 | 26011 | 60.2 | **NO** | **NO** | yes |
| `neos-5192052-neckar` | 57 | 180 | 24 | optimal | -11670000 | -11670000 | 0.00e+00 | 9 | 0.1 | yes | yes | yes |
| `neos5` | 63 | 63 | 53 | feasible | 15.5 | 15 | 1.13e-01 | 10421 | 60.1 | **NO** | **NO** | yes |
| `noswot` | 182 | 128 | 100 | feasible | -41 | -41.00000885 | 4.88e-02 | 47572 | 60.1 | yes | **NO** | yes |
| `opt1217` | 64 | 769 | 768 | feasible | -16 | -16 | 2.51e-01 | 61231 | 60.0 | yes | **NO** | yes |
| `p0201` | 133 | 201 | 201 | feasible | 7615 | 7615 | 1.19e-16 | 860 | 5.2 | yes | **NO** | yes |
| `pk1` | 45 | 86 | 55 | feasible | 18 | 11 | 7.46e-01 | 68183 | 60.1 | **NO** | **NO** | yes |
| `ran12x21` | 285 | 504 | 252 | feasible | 3794 | 3664 | 9.74e-02 | 21447 | 60.1 | **NO** | **NO** | yes |
| `ran13x13` | 195 | 338 | 169 | feasible | 3319 | 3252 | 8.55e-02 | 34421 | 60.0 | **NO** | **NO** | yes |
| `rlp1` | 68 | 461 | 450 | feasible | 15 | 15 | 1.15e-01 | 44075 | 60.1 | yes | **NO** | yes |
| `supportcase14` | 234 | 304 | 304 | optimal | 288 | 288 | 0.00e+00 | 89 | 1.5 | yes | yes | yes |
| `supportcase16` | 130 | 319 | 319 | optimal | 288 | 288 | 0.00e+00 | 70 | 0.9 | yes | yes | yes |
| `timtab1` | 171 | 397 | 171 | feasible | 1176264 | 764772 | 7.51e-01 | 27963 | 60.1 | **NO** | **NO** | yes |

**Not proved optimal**, named rather than dropped: `b-ball`, `ej`, `enlight8`, `enlight_hard`, `f2gap40400`, `flugpl`, `gen-ip016`, `gen-ip054`, `k16x240b`, `markshare1`, `markshare_4_0`, `markshare_5_0`, `neos-3072252-nete`, `neos-3611689-kaihu`, `neos-5140963-mincio`, `neos5`, `noswot`, `opt1217`, `p0201`, `pk1`, `ran12x21`, `ran13x13`, `rlp1`, `timtab1`.

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

The comparison below is run on **the same tier as section 1b**, not on the nine-instance
demo set. Comparing only where we pass would be the easy version of this table and would say
nothing: the instances we fail are exactly the ones a reader should want to see against a
mature solver.

Source CSV: `bench/results/compare-highs-medium-53cbe16.csv`  
Commit `53cbe16` · machine `Windows-AMD64`

**50 of 50** instances where the two solvers agree on the objective.

Times are **solver-internal on both sides** - HiGHS's own `getRunTime()` against our `effort.solve_seconds` - so process start-up is excluded for both. At this instance size start-up would otherwise dominate and the comparison would measure the wrong thing entirely.

| instance | SANKHYA obj | HiGHS obj | agree | SANKHYA (s) | HiGHS (s) | ratio |
|---|---:|---:|:--:|---:|---:|---:|
| `adlittle` | 2.25494963e+05 | 2.25494963e+05 | yes | 0.001 | 0.001 | 1.08x |
| `afiro` | -4.64753143e+02 | -4.64753143e+02 | yes | 0.000 | 0.000 | 0.94x |
| `agg` | -3.59917673e+07 | -3.59917673e+07 | yes | 0.008 | 0.007 | 1.23x |
| `bandm` | -1.58628018e+02 | -1.58628018e+02 | yes | 0.020 | 0.011 | 1.81x |
| `beaconfd` | 3.35924858e+04 | 3.35924858e+04 | yes | 0.011 | 0.005 | 2.11x |
| `blend` | -3.08121498e+01 | -3.08121498e+01 | yes | 0.003 | 0.002 | 1.34x |
| `boeing1` | -3.35213568e+02 | -3.35213568e+02 | yes | 0.024 | 0.020 | 1.21x |
| `boeing2` | -3.15018728e+02 | -3.15018728e+02 | yes | 0.004 | 0.003 | 1.42x |
| `bore3d` | 1.37308039e+03 | 1.37308039e+03 | yes | 0.006 | 0.002 | 2.26x |
| `brandy` | 1.51850990e+03 | 1.51850990e+03 | yes | 0.012 | 0.007 | 1.68x |
| `capri` | 2.69001291e+03 | 2.69001291e+03 | yes | 0.052 | 0.011 | 4.90x |
| `d6cube` | 3.15491667e+02 | 3.15491667e+02 | yes | 0.479 | 0.137 | 3.49x |
| `degen2` | -1.43517800e+03 | -1.43517800e+03 | yes | 0.047 | 0.015 | 3.05x |
| `e226` | -1.16389291e+01 | -1.16389291e+01 | yes | 0.020 | 0.008 | 2.63x |
| `etamacro` | -7.55715233e+02 | -7.55715233e+02 | yes | 0.032 | 0.010 | 3.14x |
| `finnis` | 1.72791066e+05 | 1.72791066e+05 | yes | 0.020 | 0.007 | 2.70x |
| `fit1d` | -9.14637809e+03 | -9.14637809e+03 | yes | 0.012 | 0.014 | 0.91x |
| `fit2d` | -6.84642933e+04 | -6.84642933e+04 | yes | 0.287 | 0.193 | 1.49x |
| `forplan` | -6.64218961e+02 | -6.64218961e+02 | yes | 0.017 | 0.007 | 2.41x |
| `grow15` | -1.06870941e+08 | -1.06870941e+08 | yes | 0.290 | 0.045 | 6.39x |
| `grow22` | -1.60834336e+08 | -1.60834336e+08 | yes | 0.363 | 0.092 | 3.94x |
| `grow7` | -4.77878118e+07 | -4.77878118e+07 | yes | 0.089 | 0.012 | 7.21x |
| `israel` | -8.96644822e+05 | -8.96644822e+05 | yes | 0.010 | 0.004 | 2.54x |
| `kb2` | -1.74990013e+03 | -1.74990013e+03 | yes | 0.001 | 0.001 | 1.61x |
| `lotfi` | -2.52647061e+01 | -2.52647061e+01 | yes | 0.006 | 0.002 | 2.58x |
| `pilot4` | -2.58113926e+03 | -2.58113926e+03 | yes | 0.098 | 0.028 | 3.53x |
| `recipe` | -2.66616000e+02 | -2.66616000e+02 | yes | 0.001 | 0.001 | 1.02x |
| `sc105` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.002 | 0.001 | 1.96x |
| `sc205` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.005 | 0.002 | 2.34x |
| `sc50a` | -6.45750771e+01 | -6.45750771e+01 | yes | 0.001 | 0.001 | 0.94x |
| `sc50b` | -7.00000000e+01 | -7.00000000e+01 | yes | 0.001 | 0.001 | 0.78x |
| `scagr25` | -1.47534331e+07 | -1.47534331e+07 | yes | 0.026 | 0.007 | 3.70x |
| `scagr7` | -2.33138982e+06 | -2.33138982e+06 | yes | 0.002 | 0.002 | 1.20x |
| `scfxm1` | 1.84167590e+04 | 1.84167590e+04 | yes | 0.021 | 0.009 | 2.50x |
| `scorpion` | 1.87812482e+03 | 1.87812482e+03 | yes | 0.009 | 0.003 | 2.65x |
| `scrs8` | 9.04296954e+02 | 9.04296954e+02 | yes | 0.039 | 0.013 | 3.09x |
| `scsd1` | 8.66666667e+00 | 8.66666667e+00 | yes | 0.009 | 0.003 | 3.46x |
| `scsd6` | 5.05000001e+01 | 5.05000001e+01 | yes | 0.024 | 0.008 | 2.91x |
| `scsd8` | 9.05000000e+02 | 9.05000000e+02 | yes | 0.175 | 0.071 | 2.47x |
| `sctap1` | 1.41225000e+03 | 1.41225000e+03 | yes | 0.013 | 0.009 | 1.47x |
| `share1b` | -7.65893186e+04 | -7.65893186e+04 | yes | 0.006 | 0.004 | 1.70x |
| `share2b` | -4.15732241e+02 | -4.15732241e+02 | yes | 0.004 | 0.002 | 1.77x |
| `ship04l` | 1.79332454e+06 | 1.79332454e+06 | yes | 0.030 | 0.010 | 3.08x |
| `ship04s` | 1.79871470e+06 | 1.79871470e+06 | yes | 0.015 | 0.007 | 2.11x |
| `stair` | -2.51266951e+02 | -2.51266951e+02 | yes | 0.038 | 0.016 | 2.33x |
| `standata` | 1.25769950e+03 | 1.25769950e+03 | yes | 0.005 | 0.005 | 0.96x |
| `standmps` | 1.40601750e+03 | 1.40601750e+03 | yes | 0.013 | 0.006 | 2.05x |
| `stocfor1` | -4.11319762e+04 | -4.11319762e+04 | yes | 0.003 | 0.002 | 1.62x |
| `tuff` | 2.92147765e-01 | 2.92147765e-01 | yes | 0.013 | 0.012 | 1.08x |
| `wood1p` | 1.44290241e+00 | 1.44290241e+00 | yes | 0.154 | 0.092 | 1.67x |

**Summary**

- SANKHYA shifted geometric mean: **0.047s**
- HiGHS shifted geometric mean: **0.018s**
- SANKHYA is **2.6x** the HiGHS time by that measure

- per-instance ratio: median **2.11x**, worst **7.21x**, faster than HiGHS on **5 of 50** instances

We are **2.57x slower** than HiGHS by this measure, and publish that rather than bury it. HiGHS is a decade of specialist work with presolve, a dual simplex and a mature pricing scheme. This solver now has a presolve (#43, #92) and a dual simplex (#65) of its own, both defaults, so what remains between the two is the pricing and the years. The part that has to be right first is that **the answers agree** - the problem statement asks us to compare, not to win.

---

## 5. Robustness — where the solver stops working

PS26119 asks for "a clear demonstration of numerical robustness ... involving degeneracy,
weak LP relaxations or ill-conditioned constraint matrices". `data/casestudies/` demonstrates
each hazard on one chosen instance; this section is the sweep that finds the case we do not
handle. Every instance is built from a chosen primal-dual pair, so its optimum is known
before it is solved (the construction is `tests/oracles/lp_generator.hpp`'s, in
`bench/runners/robustness.py`), and each family pushes one hazard until the answer, or the
certificate, moves. The reduced version runs in CI (`tests/robustness/`), together with the
adversarial families judged by the exact rational oracle and the classic cycling examples
of Beale and Kuhn.

Measured on commit `53cbe16` (Windows-AMD64), 192 solves, 5 families. Source: `robustness-53cbe16.csv`.

| family | parameter | last k that passed on every instance | first k that failed | what failed |
|---|---|---|---|---|
| `conditioning` | entry spread 10^k | 10 | 12 | infeasible, relative error 1.0e+00, verified no: row 7 needs activity of at least 8e-06 but the column bounds cap it at 0 |
| `near_parallel` | twin rows differing by a relative 10^-k | 16 | passes the whole sweep (k up to 16) | - |
| `cost_ratio` | costs spanning 10^k | 9 | 10 | optimal, relative error 5.7e-16, verified 0: [FAIL] complementary slackness     worst /multiplier/ * slack = 1.599e-05 on R4 |
| `redundancy` | k times the row count of implied rows | 32 | passes the whole sweep (k up to 32) | - |
| `degeneracy` | k times the column count of rows, all active | 64 | passes the whole sweep (k up to 64) | - |

Reading the table: the `conditioning` cliff is `kZeroDrop` (`tolerances.hpp`), the threshold below which a coefficient is treated as zero everywhere in the solver. At an entry spread of 1e12 the smallest coefficients fall under 1e-11, the model that gets solved is not the model that was written, and presolve then reports - correctly, about the truncated model - that a row cannot reach its bound. A model whose answer depends on a coefficient below 1e-11 is outside this solver's range; lowering the threshold would move the cliff, not remove it. The other limits are limits of the CERTIFICATE, not the answer: where the objective is right to 1e-15 and the verifier still rejects, the reduced costs or multipliers carry more rounding than its tolerances allow, which is worth knowing exactly because those tolerances are what a downstream consumer of the duals gets.

---

## 6. What these numbers do not say

- **Nothing here supports a claim about large models.** Section 1d is the evidence at
  the scale PS26119's "thousands to millions of variables" means, and it is a table of
  named time limits: the solver reaches Netlib's largest instances and stops there. No
  pass rate above substitutes for that table. Tracked as part of #54.
- Wall-clock times at this size are dominated by process start-up and file reading, so
  ratios between solvers are not meaningful until the instances get big enough to matter.
  The comparison in section 4 uses solver-internal time on both sides for that reason.
- The failures in section 1b are real and are not going to be quietly dropped from a later
  edition of this file. Each one carries the issue tracking it.
- Two engines named in PS26119 are not measured on this page. The interior-point method
  (`algorithm=ipm`, #56) is opt-in and produces no basis, so it is not the engine behind any
  table above; its own Netlib run is committed as `netlib-full-*-ipm.csv` and quoted in
  `docs/PS26119_COVERAGE.md`, not here, because a run made with a non-default option is a
  measurement of that option rather than the tier's evidence. There is no GPU backend on
  `main` (#16-#19). `docs/PROVENANCE.md` and issue #54 carry the full accounting.

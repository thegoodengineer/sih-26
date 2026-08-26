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

Source CSV: `bench/results/netlib-37e08f9.csv`  
Commit `37e08f9` · machine `Windows-AMD64` · generated 2026-08-26T05:29:59+00:00

**8 of 8 instances in this working set** matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

To be plain about coverage: this is **8 of the 89 instances** Netlib publishes, chosen as the small, well conditioned end of the set. It is not a claim about the other 81, and it is not a claim about large models. Widening the set is tracked as an issue.

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 139 | 0.072 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 16 | 0.053 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 510 | 0.122 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 108 | 0.055 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 48 | 0.049 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 4.1e-16 | 48 | 0.067 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 121 | 0.084 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 79 | 0.038 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.067s**
- slowest solved instance: 0.122s
- worst relative error against a published optimum: **1.06e-11**
- no failures on this set

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

Source CSV: `bench/results/compare-highs-30d7c6c.csv`  
Commit `30d7c6c` · machine `Windows-AMD64`

**8 of 8** instances where the two solvers agree on the objective.

| instance | SANKHYA obj | HiGHS obj | agree | SANKHYA (s) | HiGHS (s) | ratio |
|---|---:|---:|:--:|---:|---:|---:|
| `adlittle` | 2.25494963e+05 | 2.25494963e+05 | yes | 0.033 | 0.115 | 0.29x |
| `afiro` | -4.64753143e+02 | -4.64753143e+02 | yes | 0.068 | 0.044 | 1.53x |
| `blend` | -3.08121498e+01 | -3.08121498e+01 | yes | 0.069 | 0.058 | 1.20x |
| `sc105` | -5.22020612e+01 | -5.22020612e+01 | yes | 0.151 | 0.142 | 1.06x |
| `sc50a` | -6.45750771e+01 | -6.45750771e+01 | yes | 0.089 | 0.138 | 0.65x |
| `sc50b` | -7.00000000e+01 | -7.00000000e+01 | yes | 0.082 | 0.152 | 0.54x |
| `share2b` | -4.15732241e+02 | -4.15732241e+02 | yes | 0.087 | 0.060 | 1.46x |
| `stocfor1` | -4.11319762e+04 | -4.11319762e+04 | yes | 0.100 | 0.136 | 0.73x |

**Summary**

- SANKHYA shifted geometric mean: **0.084s**
- HiGHS shifted geometric mean: **0.105s**
- SANKHYA is **0.8x** the HiGHS time by that measure

---

## 4. What these numbers do not say

- The instances here are the small end of Netlib. Nothing on this page supports a claim
  about large models.
- Wall-clock times at this size are dominated by process start-up and file reading, so
  ratios between solvers are not meaningful until the instances get big enough to matter.
- The simplex still refactorizes a dense basis from scratch every iteration (Phase 2 by
  design). Phase 6 replaces it with a sparse LU and Forrest–Tomlin updates, and the speed
  numbers here are the baseline that work will be measured against.

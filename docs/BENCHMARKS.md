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

Source CSV: `bench/results/netlib-aa68b8a.csv`  
Commit `aa68b8a` · machine `Windows-AMD64` · generated 2026-08-25T18:20:28+00:00

**8 of 8** instances matched their published optimum to a relative 1e-6 **and** passed independent verification by `tools/verify_solution.py`.

| instance | rows | cols | status | our objective | published optimum | rel. error | iters | time (s) | verified |
|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|
| `adlittle` | 56 | 97 | optimal | 2.2549496316e+05 | 2.2549496316e+05 | 1.1e-11 | 140 | 0.061 | yes |
| `afiro` | 27 | 32 | optimal | -4.6475314286e+02 | -4.6475314286e+02 | 6.1e-12 | 16 | 0.084 | yes |
| `blend` | 74 | 83 | optimal | -3.0812149846e+01 | -3.0812149846e+01 | 5.6e-12 | 468 | 0.099 | yes |
| `sc105` | 105 | 103 | optimal | -5.2202061212e+01 | -5.2202061212e+01 | 5.6e-12 | 107 | 0.080 | yes |
| `sc50a` | 50 | 48 | optimal | -6.4575077059e+01 | -6.4575077059e+01 | 6.7e-12 | 47 | 0.058 | yes |
| `sc50b` | 50 | 48 | optimal | -7.0000000000e+01 | -7.0000000000e+01 | 2.0e-16 | 50 | 0.081 | yes |
| `share2b` | 96 | 79 | optimal | -4.1573224074e+02 | -4.1573224074e+02 | 3.4e-12 | 123 | 0.090 | yes |
| `stocfor1` | 117 | 111 | optimal | -4.1131976219e+04 | -4.1131976219e+04 | 1.1e-11 | 79 | 0.089 | yes |

**Summary**

- shifted geometric mean solve time (shift 1s): **0.080s**
- slowest solved instance: 0.099s
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

No comparison has been run yet, so **this section states no numbers**.

`bench/runners/compare.py` is written and ready; it needs a HiGHS binary on the
machine, which is invoked purely as an external subprocess and is never linked
into SANKHYA (see the red line in `CLAUDE.md`).

```bash
apt-get install highs      # or conda install -c conda-forge highs
python bench/runners/compare.py --time-limit 60
```

Once run, this section regenerates itself from the emitted CSV.

---

## 4. What these numbers do not say

- The instances here are the small end of Netlib. Nothing on this page supports a claim
  about large models.
- Wall-clock times at this size are dominated by process start-up and file reading, so
  ratios between solvers are not meaningful until the instances get big enough to matter.
- The simplex still refactorizes a dense basis from scratch every iteration (Phase 2 by
  design). Phase 6 replaces it with a sparse LU and Forrest–Tomlin updates, and the speed
  numbers here are the baseline that work will be measured against.

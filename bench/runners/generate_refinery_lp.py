#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a T-period refinery planning LP with a KNOWN optimum, as free-format MPS (#211).

Every scale measurement in docs/BENCHMARKS.md is on a generated shape - random, or an abstract
staircase - and every document says the same caveat: no industrial model of that size has
been run. This is that model, in the problem statement's own vocabulary.

THE MODEL. A refinery buys crudes, runs them through a distillation unit, and the yields make
products that sit in tanks until they are sold. Per period t and crude k, the columns are
buy[k,t] (purchase), run[k,t] (throughput) and cs[k,t] (crude in storage at the end of t); per
product j, prod[j,t] (made), sell[j,t] (sold) and ps[j,t] (product in storage). The rows are:

    crude balance      cs[k,t] - cs[k,t-1] - buy[k,t] + run[k,t]  = 0     (E, couples t-1 to t)
    CDU capacity       sum_k run[k,t]                             <= cap   (L)
    production         prod[j,t] - sum_k Y[j,k] run[k,t]           = 0     (E, the yields)
    product balance    ps[j,t] - ps[j,t-1] - prod[j,t] + sell[j,t] = 0     (E, couples t-1 to t)
    unit capacity      sum_k H[u,k] run[k,t]                      <= cap_u (L, u = reformer, ...)
    quality budget     sum_k q[s,k] Y[j_s,k] run[k,t]           <= budget[s,t] (L)
    commitment         sell[j,t]                                  >= commit[j,t]     (G)

Demand is an upper bound on sell[j,t]; tank sizes are upper bounds on cs and ps. The
inventory rows are what make this a staircase: each period's columns touch their own rows and
the next period's balances, and nothing else. That is the sparsity pattern a multi-period
planning model has and the one a direct factorization exploits (#206, #193).

THE OPTIMUM IS EXACT BY CONSTRUCTION, the same way bench/runners/generate_large_lp.py and
tests/oracles/lp_generator.cpp::kkt_lp do it: choose the operating plan x* first (feasible for
every balance by construction - stocks are what is left over), choose which capacities,
specifications, demands and commitments are tight at that plan, choose multipliers y* on the
tight rows with the sign their sense demands and free multipliers on the equalities, and then
DERIVE the prices from dual feasibility:

    c = d + A^T y*,   d_j >= 0 where x*_j sits at its lower bound, d_j <= 0 at its upper bound,
                      d_j  = 0 where x*_j is strictly between its bounds.

With that, (x*, y*, d) satisfies every KKT condition of  min c^T x  s.t.  A x {<=,=,>=} b,
l <= x <= u,  so x* is optimal and c^T x* is the optimum - the crude prices, product prices
and holding costs are the ones that make this plan the best one. Every coefficient is a
decimal with a small denominator (yields in hundredths, qualities in thousandths), so the
arithmetic is exact in Fractions and the optimum is printed from the exact value.

The script checks all of that itself before writing anything: primal feasibility of x* on
every row and bound, the sign of every multiplier, and complementary slackness on every row
and every bound. A generator that could be wrong about its own optimum would be worse than
none.

Usage:
    python bench/runners/generate_refinery_lp.py --periods 12 --seed 7 --out refinery-12.mps
    python bench/runners/generate_refinery_lp.py --periods 365 --crudes 30 --products 16 ...

Sizes with the defaults: T = 12 is 1,068 rows (monthly), T = 365 is 32,485 (daily),
T = 8,760 is 779,640 (hourly).
"""
from __future__ import annotations

import argparse
import random
import sys
from fractions import Fraction
from pathlib import Path

# ---- Coefficient ranges: decimals with small denominators, so the optimum is exact -------
YIELD_DENOMINATOR = 100      # yields in hundredths; each crude's yields sum to at most 1
QUALITY_DENOMINATOR = 1000   # qualities (sulphur, density, ...) in thousandths
PLAN_RANGE = 40              # run[k,t] and buy[k,t] drawn from [0, this]
STOCK_SLACK = 5              # tank size = largest stock held + [0, this]; 0 makes it tight
MULTIPLIER_RANGE = 9         # |y*_i| and d_j drawn from [1, this]
TIGHT_PROBABILITY = 0.5      # fraction of capacity / spec / demand / commitment rows made tight


class Instance:
    """Everything the MPS needs, plus what the caller reports and verifies."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str]] = []             # (name, sense) sense in {E, L, G}
        self.row_entries: list[dict[int, Fraction]] = []  # row -> {col: coefficient}
        self.rhs: list[Fraction] = []
        self.cols: list[str] = []
        self.lower: list[Fraction] = []
        self.upper: list[Fraction | None] = []            # None = +inf
        self.x: list[Fraction] = []                       # the chosen optimal plan
        self.y: list[Fraction] = []                       # multipliers, minimisation sense
        self.cost: list[Fraction] = []

    def add_col(self, name: str, value: Fraction, upper: Fraction | None) -> int:
        self.cols.append(name)
        self.lower.append(Fraction(0))
        self.upper.append(upper)
        self.x.append(value)
        return len(self.cols) - 1

    def add_row(self, name: str, sense: str, entries: dict[int, Fraction],
                rhs: Fraction) -> int:
        self.rows.append((name, sense))
        self.row_entries.append(entries)
        self.rhs.append(rhs)
        self.y.append(Fraction(0))
        return len(self.rows) - 1


def build(periods: int, crudes: int, products: int, units: int, specs: int, commits: int,
          seed: int) -> Instance:
    rng = random.Random(seed)
    inst = Instance()
    fr = Fraction

    # ---- The refinery's fixed data --------------------------------------------------------
    # Yields Y[j][k] in hundredths, summing to at most 1 per crude: the rest is loss and fuel.
    yields: list[list[Fraction]] = [[fr(0)] * crudes for _ in range(products)]
    for k in range(crudes):
        cut = sorted(rng.sample(range(1, 100), products - 1))
        parts = [b - a for a, b in zip([0] + cut, cut + [rng.randint(cut[-1] + 1, 100)])]
        # `parts` sums to the last endpoint <= 100; the balance is loss.
        for j in range(products):
            yields[j][k] = fr(parts[j], YIELD_DENOMINATOR)
    # Unit loads H[u][k] in hundredths of a barrel per barrel run, zero for most crudes.
    loads: list[list[Fraction]] = [[fr(0)] * crudes for _ in range(units)]
    for u in range(units):
        for k in rng.sample(range(crudes), max(1, crudes // 2)):
            loads[u][k] = fr(rng.randint(5, 60), 100)
    # Qualities q[s][k] of each crude for each spec, in thousandths, and which product pool
    # each spec applies to.
    qualities: list[list[Fraction]] = [[fr(rng.randint(1, 999), QUALITY_DENOMINATOR)
                                        for _ in range(crudes)] for _ in range(specs)]
    spec_product = [rng.randrange(products) for _ in range(specs)]
    commit_products = rng.sample(range(products), min(commits, products))

    # ---- The plan x*, feasible for every balance by construction --------------------------
    # Stocks are what is left over, so the balances hold exactly; run never exceeds what is
    # in the tank plus what was bought. Half the periods run the CDU at capacity.
    cdu_capacity = fr(PLAN_RANGE * crudes // 2)
    run = [[fr(0)] * periods for _ in range(crudes)]
    buy = [[fr(0)] * periods for _ in range(crudes)]
    cs = [[fr(0)] * periods for _ in range(crudes)]
    cs_initial = [fr(rng.randint(0, PLAN_RANGE)) for _ in range(crudes)]
    cdu_tight = [rng.random() < TIGHT_PROBABILITY for _ in range(periods)]
    for t in range(periods):
        # Draw a throughput per crude, then scale the period to capacity when it is tight.
        draw = [fr(rng.randint(0, PLAN_RANGE)) for _ in range(crudes)]
        total = sum(draw)
        if cdu_tight[t] and total > 0:
            # Scale to hit capacity exactly with integer-friendly arithmetic: give the
            # remainder to the first crude that runs. Then no draw exceeds capacity.
            scale = cdu_capacity / total
            draw = [d * scale for d in draw]
            draw = [fr(int(d)) for d in draw]  # floor each; add the remainder below
            short = cdu_capacity - sum(draw)
            for k in range(crudes):
                if draw[k] > 0 or short > 0:
                    draw[k] += short
                    break
        elif total > cdu_capacity:
            draw = [d * (cdu_capacity - 1) / total for d in draw]
            draw = [fr(int(d)) for d in draw]
        for k in range(crudes):
            previous = cs_initial[k] if t == 0 else cs[k][t - 1]
            run[k][t] = draw[k]
            need = max(fr(0), run[k][t] - previous)
            buy[k][t] = need + fr(rng.randint(0, PLAN_RANGE // 4))
            cs[k][t] = previous + buy[k][t] - run[k][t]
            assert cs[k][t] >= 0
    prod = [[sum(yields[j][k] * run[k][t] for k in range(crudes)) for t in range(periods)]
            for j in range(products)]
    sell = [[fr(0)] * periods for _ in range(products)]
    ps = [[fr(0)] * periods for _ in range(products)]
    ps_initial = [fr(rng.randint(0, PLAN_RANGE)) for _ in range(products)]
    for j in range(products):
        for t in range(periods):
            previous = ps_initial[j] if t == 0 else ps[j][t - 1]
            available = previous + prod[j][t]
            # Sell a random share of what is available, rounded DOWN to a hundredth, and keep
            # the rest. The rounding matters: production is in hundredths (yields times
            # integer throughputs), and a share taken as an exact tenth of a stock that is
            # itself a tenth of a stock compounds into 10^t denominators by period t - no
            # longer a short decimal, and the file must carry every number exactly.
            share = fr(rng.randint(0, 10), 10)
            sell[j][t] = fr(int(available * share * 100), 100)
            ps[j][t] = available - sell[j][t]
            assert ps[j][t] >= 0

    # ---- Columns, with the bounds that hold at x* (tight where chosen) -------------------
    crude_tank = [max(cs[k]) + fr(rng.randint(0, STOCK_SLACK)) for k in range(crudes)]
    product_tank = [max(ps[j]) + fr(rng.randint(0, STOCK_SLACK)) for j in range(products)]
    col_buy = [[0] * periods for _ in range(crudes)]
    col_run = [[0] * periods for _ in range(crudes)]
    col_cs = [[0] * periods for _ in range(crudes)]
    col_prod = [[0] * periods for _ in range(products)]
    col_sell = [[0] * periods for _ in range(products)]
    col_ps = [[0] * periods for _ in range(products)]
    for t in range(periods):
        for k in range(crudes):
            col_buy[k][t] = inst.add_col(f"BUY_{k}_{t}", buy[k][t], None)
            col_run[k][t] = inst.add_col(f"RUN_{k}_{t}", run[k][t], cdu_capacity)
            col_cs[k][t] = inst.add_col(f"CS_{k}_{t}", cs[k][t], crude_tank[k])
        for j in range(products):
            col_prod[j][t] = inst.add_col(f"PROD_{j}_{t}", prod[j][t], None)
            # Demand: the upper bound on sales, tight in about half the periods.
            demand = sell[j][t] + (fr(0) if rng.random() < TIGHT_PROBABILITY
                                   else fr(rng.randint(1, PLAN_RANGE)))
            col_sell[j][t] = inst.add_col(f"SELL_{j}_{t}", sell[j][t], demand)
            col_ps[j][t] = inst.add_col(f"PS_{j}_{t}", ps[j][t], product_tank[j])

    # ---- Rows, with the right-hand sides that hold at x* ---------------------------------
    for t in range(periods):
        for k in range(crudes):
            entries = {col_cs[k][t]: fr(1), col_buy[k][t]: fr(-1), col_run[k][t]: fr(1)}
            rhs = fr(0)
            if t == 0:
                rhs = cs_initial[k]
            else:
                entries[col_cs[k][t - 1]] = fr(-1)
            inst.add_row(f"CBAL_{k}_{t}", "E", entries, rhs)
        entries = {col_run[k][t]: fr(1) for k in range(crudes)}
        inst.add_row(f"CDU_{t}", "L", entries, cdu_capacity)
        for j in range(products):
            entries = {col_prod[j][t]: fr(1)}
            for k in range(crudes):
                if yields[j][k] != 0:
                    entries[col_run[k][t]] = -yields[j][k]
            inst.add_row(f"MAKE_{j}_{t}", "E", entries, fr(0))
        for j in range(products):
            entries = {col_ps[j][t]: fr(1), col_prod[j][t]: fr(-1), col_sell[j][t]: fr(1)}
            rhs = fr(0)
            if t == 0:
                rhs = ps_initial[j]
            else:
                entries[col_ps[j][t - 1]] = fr(-1)
            inst.add_row(f"PBAL_{j}_{t}", "E", entries, rhs)
        for u in range(units):
            entries = {col_run[k][t]: loads[u][k] for k in range(crudes) if loads[u][k] != 0}
            load = sum(loads[u][k] * run[k][t] for k in range(crudes))
            tight = rng.random() < TIGHT_PROBABILITY
            inst.add_row(f"UNIT_{u}_{t}", "L", entries,
                         load if tight else load + fr(rng.randint(1, PLAN_RANGE)))
        for s in range(specs):
            j = spec_product[s]
            # A quality budget for the pool this period: the mass of the contaminant the
            # crudes bring into product j, sum_k q[s,k] Y[j,k] run[k,t], may not exceed the
            # budget - tight (the blend sits exactly on it) in about half the periods. The
            # budget form is used rather than a concentration cap, qmax * prod[j,t], because
            # a concentration that is exactly met at an integer plan is generally not a
            # short decimal, and every coefficient in this file must be one so that the
            # optimum stays exact. The sparsity is the same: one row over the crudes run.
            entries = {}
            blended = fr(0)
            for k in range(crudes):
                coefficient = qualities[s][k] * yields[j][k]
                if coefficient != 0:
                    entries[col_run[k][t]] = coefficient
                    blended += coefficient * run[k][t]
            tight = rng.random() < TIGHT_PROBABILITY
            budget = blended if tight else blended + fr(rng.randint(1, PLAN_RANGE), 100)
            inst.add_row(f"SPEC_{s}_{t}", "L", entries, budget)
        for j in commit_products:
            tight = rng.random() < TIGHT_PROBABILITY
            commit = sell[j][t] if tight else max(fr(0), sell[j][t] - fr(rng.randint(1, 10)))
            inst.add_row(f"COMMIT_{j}_{t}", "G", {col_sell[j][t]: fr(1)}, commit)

    # ---- Multipliers on the tight rows, then the prices that make x* optimal -------------
    # Minimisation sense: a G row active at its bound takes y >= 0, an L row y <= 0, an
    # equality any sign. Slack inequality rows take 0 (complementary slackness).
    activity = [sum(c * inst.x[col] for col, c in entries.items())
                for entries in inst.row_entries]
    for i, (name, sense) in enumerate(inst.rows):
        tight = activity[i] == inst.rhs[i]
        if sense == "E":
            assert tight, f"{name}: balance does not hold at the plan"
            inst.y[i] = fr(rng.choice([-1, 1]) * rng.randint(1, MULTIPLIER_RANGE))
        elif sense == "L":
            assert activity[i] <= inst.rhs[i], f"{name}: plan violates the row"
            inst.y[i] = fr(-rng.randint(1, MULTIPLIER_RANGE)) if tight else fr(0)
        else:
            assert activity[i] >= inst.rhs[i], f"{name}: plan violates the row"
            inst.y[i] = fr(rng.randint(1, MULTIPLIER_RANGE)) if tight else fr(0)
    a_transpose_y = [fr(0)] * len(inst.cols)
    for i, entries in enumerate(inst.row_entries):
        if inst.y[i] == 0:
            continue
        for col, c in entries.items():
            a_transpose_y[col] += c * inst.y[i]
    for j in range(len(inst.cols)):
        at_lower = inst.x[j] == inst.lower[j]
        at_upper = inst.upper[j] is not None and inst.x[j] == inst.upper[j]
        if at_lower and at_upper:
            d = fr(rng.choice([-1, 1]) * rng.randint(0, MULTIPLIER_RANGE))
        elif at_lower:
            d = fr(rng.randint(0, MULTIPLIER_RANGE))
        elif at_upper:
            d = fr(-rng.randint(0, MULTIPLIER_RANGE))
        else:
            d = fr(0)
        inst.cost.append(d + a_transpose_y[j])
    return inst


def verify(inst: Instance) -> None:
    """Every KKT condition, checked exactly, before a single line is written."""
    n = len(inst.cols)
    for j in range(n):
        assert inst.lower[j] <= inst.x[j], f"{inst.cols[j]} below its lower bound"
        assert inst.upper[j] is None or inst.x[j] <= inst.upper[j], (
            f"{inst.cols[j]} above its upper bound")
    a_transpose_y = [Fraction(0)] * n
    for i, entries in enumerate(inst.row_entries):
        activity = sum(c * inst.x[col] for col, c in entries.items())
        name, sense = inst.rows[i]
        if sense == "E":
            assert activity == inst.rhs[i], f"{name}: equality violated"
        elif sense == "L":
            assert activity <= inst.rhs[i], f"{name}: <= violated"
            assert inst.y[i] <= 0, f"{name}: L row needs a non-positive multiplier"
        else:
            assert activity >= inst.rhs[i], f"{name}: >= violated"
            assert inst.y[i] >= 0, f"{name}: G row needs a non-negative multiplier"
        if sense != "E" and activity != inst.rhs[i]:
            assert inst.y[i] == 0, f"{name}: slack row carries a multiplier"
        for col, c in entries.items():
            a_transpose_y[col] += c * inst.y[i]
    for j in range(n):
        d = inst.cost[j] - a_transpose_y[j]
        at_lower = inst.x[j] == inst.lower[j]
        at_upper = inst.upper[j] is not None and inst.x[j] == inst.upper[j]
        if not at_lower and not at_upper:
            assert d == 0, f"{inst.cols[j]}: interior column with reduced cost {d}"
        elif at_lower and not at_upper:
            assert d >= 0, f"{inst.cols[j]}: at lower bound with reduced cost {d}"
        elif at_upper and not at_lower:
            assert d <= 0, f"{inst.cols[j]}: at upper bound with reduced cost {d}"


def decimal(v: Fraction) -> str:
    """Exact decimal text for a Fraction whose denominator divides a power of ten."""
    if v.denominator == 1:
        return str(v.numerator)
    scale = 1
    digits = 0
    while (scale % v.denominator) != 0:
        scale *= 10
        digits += 1
        if digits > 12:
            raise ValueError(f"{v} is not a short decimal")
    sign = "-" if v < 0 else ""
    integer, fraction = divmod(abs(v.numerator) * (scale // v.denominator), scale)
    text = f"{sign}{integer}.{fraction:0{digits}d}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def write_mps(inst: Instance, out: Path, periods: int, seed: int) -> None:
    optimum = sum(c * x for c, x in zip(inst.cost, inst.x))
    col_rows: list[list[tuple[int, Fraction]]] = [[] for _ in inst.cols]
    for i, entries in enumerate(inst.row_entries):
        for col, c in entries.items():
            col_rows[col].append((i, c))
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"NAME          REFINERY_T{periods}_S{seed}\n")
        f.write(f"* generator: bench/runners/generate_refinery_lp.py (#211)\n")
        f.write(f"* structure: refinery, {periods} periods\n")
        f.write(f"* analytic optimum: {float(optimum)!r}\n")
        f.write("ROWS\n N  COST\n")
        for name, sense in inst.rows:
            f.write(f" {sense}  {name}\n")
        f.write("COLUMNS\n")
        for j, name in enumerate(inst.cols):
            if inst.cost[j] != 0:
                f.write(f"    {name}  COST  {decimal(inst.cost[j])}\n")
            for i, c in col_rows[j]:
                f.write(f"    {name}  {inst.rows[i][0]}  {decimal(c)}\n")
            if inst.cost[j] == 0 and not col_rows[j]:
                f.write(f"    {name}  COST  0\n")
        f.write("RHS\n")
        for i, (name, _) in enumerate(inst.rows):
            if inst.rhs[i] != 0:
                f.write(f"    RHS  {name}  {decimal(inst.rhs[i])}\n")
        f.write("BOUNDS\n")
        for j, name in enumerate(inst.cols):
            if inst.upper[j] is not None:
                f.write(f" UP BND  {name}  {decimal(inst.upper[j])}\n")
        f.write("ENDATA\n")
    print(f"wrote {out}")
    print(f"  rows: {len(inst.rows)}  columns: {len(inst.cols)}  "
          f"nonzeros: {sum(len(e) for e in inst.row_entries)}")
    print(f"  analytic optimum: {float(optimum)!r}")
    print(f"  structure: refinery, {periods} periods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--periods", type=int, required=True, help="T: 12, 365, 8760")
    parser.add_argument("--crudes", type=int, default=30)
    parser.add_argument("--products", type=int, default=16)
    parser.add_argument("--units", type=int, default=8)
    parser.add_argument("--specs", type=int, default=10)
    parser.add_argument("--commits", type=int, default=8,
                        help="products with a delivery commitment row per period")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.periods < 1 or args.crudes < 2 or args.products < 2:
        print("periods >= 1, crudes >= 2 and products >= 2", file=sys.stderr)
        return 2
    inst = build(args.periods, args.crudes, args.products, args.units, args.specs,
                 args.commits, args.seed)
    verify(inst)
    write_mps(inst, args.out, args.periods, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())

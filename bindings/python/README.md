# SANKHYA — Python bindings

```python
import sankhya

model = sankhya.Model(maximize=True)
x = model.add_column(cost=3.0, upper=3.0, name="x")
y = model.add_column(cost=2.0, name="y")
model.add_row({x: 1.0, y: 1.0}, upper=4.0)
model.add_row({x: 1.0, y: 3.0}, upper=6.0)

result = model.solve()
print(result.status, result.objective, result.x)
# optimal 11.0 [3.0, 1.0]
```

Or read a file:

```python
model = sankhya.Model.read("data/netlib/afiro.mps")
print(model.solve().objective)   # -464.7531428571429
```

## Install

There is nothing to install and nothing to compile. Build the solver, then put the package
on your path:

```bash
scripts/configure.sh build Release && cmake --build build -j
export PYTHONPATH=bindings/python
```

The bindings load `libsankhya.so` / `libsankhya.dll` from `build/` (and a few other usual
directories). Point `SANKHYA_LIBRARY` at a specific file to override the search.

**On Windows**, the library imports `zlib1.dll`, which lives beside the compiler rather than
anywhere Windows searches by default — Python 3.8 stopped honouring `PATH` for this. The
loader adds the usual MSYS2 and MinGW directories itself; if yours is elsewhere, set
`SANKHYA_DLL_DIR`.

## Why ctypes and not pybind11

A compiled extension has to be built against the exact Python that imports it, which turns
"use the solver from Python" into a second build system, a wheel, and an ABI to keep
matching. ctypes needs none of that: the shared library the C++ build already produces is
the whole dependency, and the same file works from CPython, PyPy, or anything else with an
FFI.

It also means these bindings exercise `include/sankhya/sankhya.h` exactly as a third-party
caller would. A defect in that boundary surfaces here rather than being hidden by a
C++-aware binding layer — which is not hypothetical: writing the C API found that an unknown
option name reached a `std::abort()` in the C++ accessors, which across an FFI would have
taken the interpreter down with it.

## What you get

| | |
|---|---|
| `Model` | `add_column`, `add_row`, `set_coefficient`, `set_quadratic`, `read`, `validate`, `solve` |
| `Options` | any option the CLI accepts, by name; types dispatched from the Python value |
| `Result` | `status`, `objective`, `x`, `row_duals`, `reduced_costs`, `row_activities`, `iterations`, `nodes`, `seconds`, `absolute_gap`, `relative_gap`, `claims_a_point`, `farkas_dual` and `primal_ray` (each `None` when the solver attached no proof) |

`Result` also exposes the **measured** quality of the point — `primal_infeasibility`,
`dual_infeasibility`, `integrality_violation`. These are recomputed from the returned vectors
rather than asserted by the engine about itself, and a caller writing its own acceptance test
should read them rather than trusting `status` alone.

## Two behaviours worth knowing

**A failed call raises; a solved model returns.** An infeasible model is a successful call
whose `status` is `"infeasible"` — it does not raise. Conflating the two would make an
ordinary modelling outcome indistinguishable from a bug in your code.

**`set_coefficient` replaces, it does not accumulate.** Unlike an MPS file, where a repeated
entry is an error, setting the same cell twice through the API simply overwrites it.
Summing would silently double a coefficient, and that is not a change you can see in the
answer.

## Tests

```bash
PYTHONPATH=bindings/python python bindings/python/test_bindings.py
```

Every expected value is derived by hand in a comment above it. A binding is a translation
layer, and the mistake it can make that the solver cannot is reading the right number out of
the wrong field — which a test that captured its expectations from a previous run would
happily freeze in.

## Watching a solve, and stopping it

`solve()` takes a `callback`, called at most every 100 ms with a `Progress` namedtuple:
`phase` (`"presolve"`, `"lp"` or `"tree"`), `iterations`, `nodes`, `open_nodes`,
`objective`, `best_bound`, `gap` and `elapsed_seconds`. Return a non-zero value to stop
the solve at its next safe point; the result then has status `interrupted` and carries the
point in hand, exactly as a time limit does. A progress bar for a MILP is a few lines:

```python
import sys
import sankhya

model = sankhya.Model.read("demo/blend_milp.mps")

def progress(p):
    gap = "" if p.gap == float("inf") else f"  gap {p.gap:.3g}"
    sys.stderr.write(f"\r[{p.phase:>8}] {p.elapsed_seconds:6.1f}s  nodes {p.nodes:6d}"
                     f"  open {p.open_nodes:5d}  incumbent {p.objective:.6g}{gap}")
    return 1 if p.elapsed_seconds > 30.0 else 0   # non-zero interrupts the solve

result = model.solve(callback=progress, log_to_console=False)
sys.stderr.write("\n")
print(result.status, result.objective)
```

(`log_to_console=False` keeps the solver's own log off the terminal so the bar is what you
see. The demo MILP solves at its root node, so it reports once; a search of thousands of
nodes reports every 100 ms.)

An exception raised inside the callback interrupts the solve and is re-raised from
`solve()`. `Model.interrupt()` does the same from another thread or a signal handler.

## Not yet

Warm starts and basis in/out. Each is a surface worth designing rather than accreting; the
C API does not expose them either.

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

## Progress and interruption (#223)

A solve on a real planning model runs for minutes; `callback` is how you watch one, and how
you stop one without losing what it has found so far. It is called at a bounded rate (by
iteration/node count and to at most roughly every 100 ms), so even a chatty callback does not
slow the solve down:

```python
def progress_bar(p: sankhya.Progress) -> bool:
    width = 30
    filled = int(width * min(1.0, 1.0 - min(p.gap, 1.0))) if p.gap else width
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {p.phase:>8}  node {p.nodes:>6}  "
          f"incumbent {p.objective:.6g}  gap {p.gap:.2e}  {p.elapsed_seconds:6.1f}s",
          end="", flush=True)
    return False  # never asks the solve to stop; return True to do that

result = model.solve(callback=progress_bar, mip_relative_gap=1e-6)
print(f"\n{result.status} {result.objective}")
```

Ctrl-C during `solve()` is handled the same way: a progress callback is always installed
internally (even when you don't pass one), which is what lets Python notice the SIGINT and
ask the solve to stop while it is otherwise blocked inside the C library - so `solve()`
returns the incumbent with `status == "interrupted"` rather than the raw `KeyboardInterrupt`
tearing down whatever was running.

`model.interrupt()` does the same thing from another thread, on purpose - it is meant to be
called while a `solve()` on a different thread is running, not before one starts (each
`solve()` begins with a clear flag, precisely so a stale `interrupt()` from an earlier,
already-finished solve cannot silently kill a later, unrelated one on a reused `Model`):

```python
import threading

worker = threading.Thread(target=lambda: results.append(model.solve()))
worker.start()
...
model.interrupt()   # from the main thread, or anywhere else
worker.join()
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
| `Model` | `add_column`, `add_row`, `set_coefficient`, `set_quadratic`, `read`, `validate`, `solve`, `interrupt` |
| `Options` | any option the CLI accepts, by name; types dispatched from the Python value |
| `Result` | `status`, `objective`, `x`, `row_duals`, `reduced_costs`, `row_activities`, `iterations`, `nodes`, `seconds` |
| `Progress` | `phase`, `iterations`, `nodes`, `open_nodes`, `objective`, `best_bound`, `gap`, `elapsed_seconds` - handed to `solve(callback=...)` |

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

## Not yet

Warm starts and basis in/out. Each is a surface worth designing rather than accreting; the
C API does not expose them either.

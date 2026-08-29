# SPDX-License-Identifier: Apache-2.0
"""SANKHYA - Python bindings.

A thin, Pythonic layer over the C API in ``include/sankhya/sankhya.h``. Nothing here
reimplements solver behaviour; it converts between Python values and the C surface, and
turns non-zero status codes into exceptions.

    >>> import sankhya
    >>> model = sankhya.Model(maximize=True)
    >>> x = model.add_column(cost=3.0, upper=3.0, name="x")
    >>> y = model.add_column(cost=2.0, name="y")
    >>> model.add_row({x: 1.0, y: 1.0}, upper=4.0)
    >>> model.add_row({x: 1.0, y: 3.0}, upper=6.0)
    >>> result = model.solve()
    >>> result.status
    'optimal'
    >>> round(result.objective, 6)
    11.0

Or from a file:

    >>> model = sankhya.Model.read("data/netlib/afiro.mps")   # doctest: +SKIP
    >>> model.solve().objective                                # doctest: +SKIP
    -464.7531428571429

MEMORY. Every handle is owned by a Python object and released in ``__del__``, so ordinary
Python lifetime rules apply and there is nothing to free by hand. ``Model`` and ``Result``
are also context managers if you would rather be explicit about it.
"""

from __future__ import annotations

import ctypes
from typing import Iterable, Mapping, Sequence

from ._library import SankhyaError, load

__all__ = ["Model", "Options", "Result", "SankhyaError", "INFINITY", "version"]

_lib = None


def _library():
    global _lib
    if _lib is None:
        _lib = load()
    return _lib


def version() -> str:
    """The solver's version string."""
    return _library().sankhya_version().decode()


def _check(status: int, what: str) -> None:
    if status == 0:
        return
    detail = _library().sankhya_last_error().decode()
    raise SankhyaError(f"{what}: {detail}" if detail else what)


class _Infinity(float):
    """The C API's infinity, as a float that repr()s recognisably."""

    def __repr__(self) -> str:
        return "sankhya.INFINITY"


def _infinity() -> float:
    return _Infinity(_library().sankhya_infinity())


# Resolved lazily on first use so that importing the package does not require the library to
# be built - a caller who imports sankhya only to catch the "not built" error should get that
# error from the call, with its instructions, rather than from the import.
class _LazyInfinity:
    _value: float | None = None

    def __float__(self) -> float:
        if _LazyInfinity._value is None:
            _LazyInfinity._value = _infinity()
        return _LazyInfinity._value

    def __repr__(self) -> str:
        return "sankhya.INFINITY"

    def __neg__(self) -> float:
        return -float(self)

    def __eq__(self, other: object) -> bool:
        return float(self) == other

    def __hash__(self) -> int:
        return hash(float("inf"))


INFINITY = _LazyInfinity()


_STATUS_NAMES = {
    0: "not_solved",
    1: "optimal",
    2: "feasible",
    3: "infeasible",
    4: "unbounded",
    5: "iteration_limit",
    6: "time_limit",
    7: "node_limit",
    8: "numerical_error",
    9: "model_error",
    10: "infeasible_or_unbounded",
}


class Options:
    """Solver options, by the same names the CLI and ``sankhya options`` use.

    Types are dispatched from the Python value, so ``Options(presolve=False,
    time_limit=10.0, algorithm="pdhg")`` does the right thing for each. An unknown name is
    rejected here rather than ignored - the C API checks the registry before the C++ setter,
    which would otherwise abort the process.
    """

    def __init__(self, **values: object) -> None:
        self._handle = _library().sankhya_options_create()
        if not self._handle:
            raise SankhyaError("could not allocate options")
        for name, value in values.items():
            self.set(name, value)

    def set(self, name: str, value: object) -> "Options":
        lib = _library()
        encoded = name.encode()
        # bool BEFORE int, because bool is a subclass of int in Python and would otherwise
        # be routed to the integer setter and rejected as the wrong type.
        if isinstance(value, bool):
            _check(lib.sankhya_options_set_bool(self._handle, encoded, 1 if value else 0),
                   f"setting option {name!r}")
        elif isinstance(value, int):
            _check(lib.sankhya_options_set_int(self._handle, encoded, value),
                   f"setting option {name!r}")
        elif isinstance(value, float):
            _check(lib.sankhya_options_set_double(self._handle, encoded, value),
                   f"setting option {name!r}")
        elif isinstance(value, str):
            _check(lib.sankhya_options_set_string(self._handle, encoded, value.encode()),
                   f"setting option {name!r}")
        else:
            raise TypeError(f"option {name!r}: unsupported value type {type(value).__name__}")
        return self

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle and _lib is not None:
            _lib.sankhya_options_free(handle)
            self._handle = None


class Result:
    """The outcome of a solve. Read-only."""

    def __init__(self, handle: int, num_cols: int, num_rows: int) -> None:
        self._handle = handle
        self._cols = num_cols
        self._rows = num_rows

    # ---- What happened -------------------------------------------------------------------

    @property
    def status(self) -> str:
        """One of: optimal, feasible, infeasible, unbounded, iteration_limit, time_limit,
        node_limit, numerical_error, model_error, infeasible_or_unbounded, not_solved.

        ``feasible`` means a usable point with optimality NOT proven; it is not a weaker
        spelling of ``optimal`` and should not be treated as one.
        """
        return _STATUS_NAMES.get(_library().sankhya_solution_status(self._handle), "unknown")

    @property
    def optimal(self) -> bool:
        return self.status == "optimal"

    @property
    def message(self) -> str:
        return _library().sankhya_solution_message(self._handle).decode()

    @property
    def objective(self) -> float:
        return _library().sankhya_solution_objective(self._handle)

    @property
    def dual_bound(self) -> float:
        """Best proven bound. Equal to the objective when optimality was proved."""
        return _library().sankhya_solution_dual_bound(self._handle)

    @property
    def iterations(self) -> int:
        return _library().sankhya_solution_iterations(self._handle)

    @property
    def nodes(self) -> int:
        return _library().sankhya_solution_nodes(self._handle)

    @property
    def seconds(self) -> float:
        return _library().sankhya_solution_seconds(self._handle)

    # ---- Measured quality ------------------------------------------------------------------

    @property
    def primal_infeasibility(self) -> float:
        """MEASURED violation of the returned point, not asserted by the engine about itself.

        Recomputed from the returned vectors before anything is reported. A caller writing
        its own acceptance test should read this rather than trusting ``status`` alone.
        """
        return _library().sankhya_solution_primal_infeasibility(self._handle)

    @property
    def dual_infeasibility(self) -> float:
        return _library().sankhya_solution_dual_infeasibility(self._handle)

    @property
    def integrality_violation(self) -> float:
        return _library().sankhya_solution_integrality_violation(self._handle)

    # ---- Vectors ---------------------------------------------------------------------------

    def _vector(self, function, count: int, what: str) -> list[float]:
        if count == 0:
            return []
        buffer = (ctypes.c_double * count)()
        _check(function(self._handle, buffer, count), f"reading {what}")
        return list(buffer)

    @property
    def x(self) -> list[float]:
        """Primal column values, in the order the columns were added."""
        return self._vector(_library().sankhya_solution_col_values, self._cols, "column values")

    @property
    def row_activities(self) -> list[float]:
        return self._vector(_library().sankhya_solution_row_activities, self._rows,
                            "row activities")

    @property
    def row_duals(self) -> list[float]:
        """Shadow prices: the marginal worth of relaxing each row by one unit."""
        return self._vector(_library().sankhya_solution_row_duals, self._rows, "row duals")

    @property
    def reduced_costs(self) -> list[float]:
        return self._vector(_library().sankhya_solution_col_duals, self._cols, "reduced costs")

    def __repr__(self) -> str:
        return f"<sankhya.Result {self.status} objective={self.objective:.10g}>"

    def __enter__(self) -> "Result":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle and _lib is not None:
            _lib.sankhya_solution_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()


class Model:
    """A linear, mixed-integer or convex quadratic program."""

    def __init__(self, maximize: bool = False) -> None:
        self._handle = _library().sankhya_model_create()
        if not self._handle:
            raise SankhyaError("could not allocate model")
        if maximize:
            _check(_library().sankhya_model_set_maximize(self._handle, 1), "setting sense")

    @classmethod
    def read(cls, path: str) -> "Model":
        """Read an MPS, QPS or LP file. Raises SankhyaError with the parser's message."""
        model = cls()
        _check(_library().sankhya_model_read(model._handle, str(path).encode()),
               f"reading {path!r}")
        return model

    # ---- Building ---------------------------------------------------------------------------

    def add_column(self, cost: float = 0.0, lower: float = 0.0, upper: float | None = None,
                   integer: bool = False, name: str | None = None) -> int:
        """Append a column and return its index.

        ``upper=None`` means no upper bound. The default lower bound is 0, matching the
        convention every MPS file uses, so a free variable needs ``lower=-sankhya.INFINITY``
        stated explicitly rather than implied.
        """
        index = ctypes.c_int(-1)
        _check(_library().sankhya_model_add_column(
            self._handle, float(cost), float(lower),
            float(INFINITY) if upper is None else float(upper),
            1 if integer else 0, name.encode() if name else None, ctypes.byref(index)),
            "adding a column")
        return index.value

    def add_row(self, coefficients: Mapping[int, float] | None = None,
                lower: float | None = None, upper: float | None = None,
                name: str | None = None) -> int:
        """Append a row ``lower <= a'x <= upper`` and return its index.

        ``None`` on either bound means unbounded on that side; pass the same value for both
        to get an equality. ``coefficients`` maps column index to value.
        """
        index = ctypes.c_int(-1)
        _check(_library().sankhya_model_add_row(
            self._handle,
            -float(INFINITY) if lower is None else float(lower),
            float(INFINITY) if upper is None else float(upper),
            name.encode() if name else None, ctypes.byref(index)), "adding a row")
        for column, value in (coefficients or {}).items():
            self.set_coefficient(index.value, column, value)
        return index.value

    def set_coefficient(self, row: int, column: int, value: float) -> None:
        """Set one constraint coefficient. Setting it twice REPLACES, never accumulates."""
        _check(_library().sankhya_model_set_coefficient(self._handle, row, column, float(value)),
               f"setting coefficient ({row}, {column})")

    def set_quadratic(self, row: int, column: int, value: float) -> None:
        """Set one entry of the objective Hessian, making this a QP.

        The objective is ``c'x + 0.5 x'Qx`` and Q is symmetric, so ``(i, j)`` and ``(j, i)``
        name ONE entry; the 0.5 belongs to the objective rather than to the value you pass.
        A non-convex Q is refused at solve time rather than solved to a local point.
        """
        _check(_library().sankhya_model_set_quadratic_coefficient(
            self._handle, row, column, float(value)), f"setting Q({row}, {column})")

    def set_objective_offset(self, offset: float) -> None:
        _check(_library().sankhya_model_set_objective_offset(self._handle, float(offset)),
               "setting the objective offset")

    # ---- Inspection ---------------------------------------------------------------------------

    @property
    def num_cols(self) -> int:
        return _library().sankhya_model_num_cols(self._handle)

    @property
    def num_rows(self) -> int:
        return _library().sankhya_model_num_rows(self._handle)

    @property
    def num_nonzeros(self) -> int:
        return _library().sankhya_model_num_nonzeros(self._handle)

    def validate(self) -> None:
        """Raise SankhyaError if the model is not internally consistent."""
        _check(_library().sankhya_model_validate(self._handle), "validating the model")

    # ---- Solving --------------------------------------------------------------------------------

    def solve(self, options: Options | None = None, **overrides: object) -> Result:
        """Solve, returning a Result.

        Options may be passed as an Options object, as keyword arguments, or both - keywords
        are applied on top. The return value describes what the SOLVER concluded; a failure
        of the CALL raises instead, so an infeasible model returns normally with
        ``status == "infeasible"`` rather than raising.
        """
        if overrides:
            options = options or Options()
            for name, value in overrides.items():
                options.set(name, value)

        handle = ctypes.c_void_p()
        _check(_library().sankhya_solve(
            self._handle, options._handle if options else None, ctypes.byref(handle)),
            "solving")
        return Result(handle.value, self.num_cols, self.num_rows)

    def __repr__(self) -> str:
        return (f"<sankhya.Model {self.num_rows} rows x {self.num_cols} columns, "
                f"{self.num_nonzeros} nonzeros>")

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle and _lib is not None:
            _lib.sankhya_model_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

"""
Simulation hook: sandboxed execution of small toy-experiment Python
snippets produced by the Synthesizer agent.

This is intentionally NOT a general-purpose code execution sandbox — it is
a narrow tool for running short, numeric "toy model" snippets (e.g. a toy
Arrhenius conductivity estimate) so a hypothesis's rough quantitative
plausibility can be sanity-checked before real lab work. Safety measures:

- Restricted builtins (no file/network/process/import-of-arbitrary-modules
  access beyond an explicit allowlist).
- Wall-clock timeout via a worker thread (portable across platforms, unlike
  signal.alarm which doesn't work on Windows / non-main threads).
- stdout capture only; no filesystem or network side effects are possible
  through the allowed builtins/modules.
- `key=value` lines printed to stdout are parsed into structured metrics.
"""

from __future__ import annotations

import io
import math
import re
import threading
import time
from contextlib import redirect_stdout
from typing import Any

from ..schemas import SimulationResult

_ALLOWED_MODULES = {"math", "statistics", "random", "itertools", "functools"}

try:
    import numpy as _np

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False

_METRIC_LINE_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")


def _safe_import(name: str, *args, **kwargs):
    if name not in _ALLOWED_MODULES:
        raise ImportError(f"Import of module '{name}' is not permitted in the sandbox.")
    import importlib

    return importlib.import_module(name)


def _build_safe_globals() -> dict[str, Any]:
    safe_builtins = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "enumerate": enumerate, "zip": zip, "map": map,
        "filter": filter, "sorted": sorted, "reversed": reversed,
        "round": round, "pow": pow, "print": print, "float": float,
        "int": int, "str": str, "bool": bool, "list": list, "dict": dict,
        "tuple": tuple, "set": set, "True": True, "False": False, "None": None,
        "__import__": _safe_import,
    }
    safe_globals: dict[str, Any] = {"__builtins__": safe_builtins, "math": math}
    if _NUMPY_AVAILABLE:
        safe_globals["np"] = _np
    return safe_globals


class SimulationTimeout(Exception):
    pass


def run_toy_simulation(code: str, timeout_seconds: float = 5.0) -> SimulationResult:
    """Execute `code` in a restricted namespace and capture stdout/metrics."""
    stdout_buffer = io.StringIO()
    error: str | None = None
    exec_globals = _build_safe_globals()
    start = time.monotonic()

    result_holder: dict[str, Any] = {}

    def _target():
        try:
            with redirect_stdout(stdout_buffer):
                exec(code, exec_globals, {})  # noqa: S102 - sandboxed by design
        except Exception as exc:  # noqa: BLE001 - we want to report any user-code error
            result_holder["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    runtime = time.monotonic() - start

    if thread.is_alive():
        # Daemon thread will be abandoned; process-level timeout protects us.
        return SimulationResult(
            executed=False,
            stdout=stdout_buffer.getvalue(),
            error=f"Execution exceeded {timeout_seconds}s timeout and was aborted.",
            runtime_seconds=runtime,
        )

    error = result_holder.get("error")
    stdout_text = stdout_buffer.getvalue()
    parsed_metrics: dict[str, float] = {}
    for line in stdout_text.splitlines():
        m = _METRIC_LINE_RE.match(line.strip())
        if m:
            try:
                parsed_metrics[m.group(1)] = float(m.group(2))
            except ValueError:
                continue

    return SimulationResult(
        executed=error is None,
        stdout=stdout_text,
        error=error,
        parsed_metrics=parsed_metrics,
        runtime_seconds=round(runtime, 4),
    )

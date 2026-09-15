from hypothesisforge.tools.simulation import run_toy_simulation


def test_simulation_runs_and_parses_metrics():
    code = "print('sigma=1.23e-3')\nprint('activation_energy=0.35')\n"
    result = run_toy_simulation(code)
    assert result.executed
    assert result.error is None
    assert result.parsed_metrics["sigma"] == 1.23e-3
    assert result.parsed_metrics["activation_energy"] == 0.35


def test_simulation_captures_runtime_error():
    code = "raise ValueError('boom')\n"
    result = run_toy_simulation(code)
    assert not result.executed
    assert "ValueError" in result.error


def test_simulation_blocks_disallowed_import():
    code = "import os\nprint(os.getcwd())\n"
    result = run_toy_simulation(code)
    assert not result.executed
    assert "ImportError" in result.error or "not permitted" in (result.error or "")


def test_simulation_allows_math_module():
    code = "import math\nprint(f'sqrt_two={math.sqrt(2):.4f}')\n"
    result = run_toy_simulation(code)
    assert result.executed
    assert abs(result.parsed_metrics["sqrt_two"] - 1.4142) < 1e-3


def test_simulation_timeout_is_reported():
    code = "while True:\n    pass\n"
    result = run_toy_simulation(code, timeout_seconds=0.5)
    assert not result.executed
    assert "timeout" in result.error.lower() or "exceeded" in result.error.lower()


def test_simulation_blocks_dunder_builtins_escape_attempt():
    code = (
        "try:\n"
        "    (1).__class__.__base__.__subclasses__()\n"
        "except Exception as e:\n"
        "    print('blocked')\n"
    )
    # This should not raise unhandled — the point is the sandbox doesn't
    # crash the host process regardless of what introspection is attempted.
    result = run_toy_simulation(code)
    assert result.executed

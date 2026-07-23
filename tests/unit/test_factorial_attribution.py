import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[2] / "scripts" / "run_factorial_attribution.py"
    spec = importlib.util.spec_from_file_location("factorial_attribution", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exact_shapley_recovers_additive_effects():
    module = _module()
    factors = ("a", "b", "c")
    weights = (0.2, -0.1, 0.4)
    values = {
        mask: 0.7 + sum(weight for index, weight in enumerate(weights) if mask & (1 << index))
        for mask in range(1 << len(factors))
    }

    result = module._exact_shapley(values, factors)

    assert result == pytest.approx({"a": weights[0], "b": weights[1], "c": weights[2]})


def test_subsets_cover_power_set():
    module = _module()

    subsets = module._subsets(("a", "b", "c"))

    assert len(subsets) == 8
    assert set() in subsets
    assert {"a", "b", "c"} in subsets

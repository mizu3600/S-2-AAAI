from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluation = load_module(
    "official_evaluation", "scripts/official_baselines/evaluate_official_baselines.py"
)
comparison = load_module("official_comparison", "scripts/official_baselines/compare_results.py")


def test_passage_metrics_distinguish_partial_and_complete_recall() -> None:
    ranking = ["distractor", "gold-a", "gold-b"]
    gold = {"gold-a", "gold-b"}

    assert evaluation.recall_at(ranking, gold, 2) == 0.5
    assert evaluation.complete_at(ranking, gold, 2) == 0.0
    assert evaluation.complete_at(ranking, gold, 3) == 1.0
    assert evaluation.reciprocal_rank(ranking, gold) == 0.5


def test_qmsxe_seed_scores_are_averaged_per_example() -> None:
    records = [
        {"framework": "qmsxe:test", "example_id": "a", "seed": 1, **_metrics(1.0)},
        {"framework": "qmsxe:test", "example_id": "a", "seed": 2, **_metrics(0.0)},
        {"framework": "qmsxe:test", "example_id": "b", "seed": 1, **_metrics(1.0)},
    ]

    result = comparison.per_example(records)

    assert result["qmsxe:test"]["a"]["mrr"] == 0.5
    assert result["qmsxe:test"]["b"]["mrr"] == 1.0


def test_paired_test_reports_exact_observed_difference() -> None:
    left = {"a": _metrics(1.0), "b": _metrics(0.5)}
    right = {"a": _metrics(0.0), "b": _metrics(0.5)}

    result = comparison.paired_test(
        left,
        right,
        "mrr",
        samples=100,
        rng=np.random.default_rng(7),
    )

    assert result["example_count"] == 2
    assert result["mean_difference"] == 0.5


def _metrics(value: float) -> dict[str, float]:
    return {metric: value for metric in comparison.METRICS}

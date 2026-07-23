from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np


METRICS = (
    "recall_at_1",
    "recall_at_2",
    "recall_at_5",
    "recall_at_10",
    "complete_at_2",
    "complete_at_5",
    "mrr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-records", type=Path, required=True)
    parser.add_argument("--qmsxe-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def per_example(records: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("error") is None:
            grouped[(record["framework"], record["example_id"])].append(record)
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (framework, example_id), rows in grouped.items():
        output[framework][example_id] = {
            metric: mean(float(row[metric]) for row in rows) for metric in METRICS
        }
    return dict(output)


def bootstrap_interval(values: np.ndarray, samples: int, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def paired_test(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    ids = sorted(set(left) & set(right))
    differences = np.asarray([left[item][metric] - right[item][metric] for item in ids])
    signs = rng.choice((-1.0, 1.0), size=(samples, len(ids)))
    null_differences = (signs * differences).mean(axis=1)
    observed = float(differences.mean())
    p_value = float(
        (np.count_nonzero(np.abs(null_differences) >= abs(observed)) + 1) / (samples + 1)
    )
    return {
        "example_count": len(ids),
        "mean_difference": observed,
        "difference_ci_95": bootstrap_interval(differences, samples, rng),
        "paired_randomization_p_value": p_value,
    }


def main() -> None:
    args = parse_args()
    official = json.loads(args.official_records.read_text(encoding="utf-8"))
    qmsxe = json.loads(args.qmsxe_records.read_text(encoding="utf-8"))
    systems = per_example(official + qmsxe)
    rng = np.random.default_rng(args.seed)
    summaries = {}
    for system, examples in systems.items():
        summaries[system] = {"example_count": len(examples)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in examples.values()])
            summaries[system][metric] = float(values.mean())
            summaries[system][f"{metric}_ci_95"] = bootstrap_interval(
                values, args.bootstrap_samples, rng
            )

    qmsxe_systems = sorted(system for system in systems if system.startswith("qmsxe:"))
    official_systems = sorted(system for system in systems if not system.startswith("qmsxe:"))
    comparisons = {}
    for official_system in official_systems:
        for qmsxe_system in qmsxe_systems:
            pair = f"{qmsxe_system} minus {official_system}"
            comparisons[pair] = {
                metric: paired_test(
                    systems[qmsxe_system],
                    systems[official_system],
                    metric,
                    args.bootstrap_samples,
                    rng,
                )
                for metric in ("recall_at_2", "complete_at_2", "mrr")
            }
    for left, right in combinations(qmsxe_systems, 2):
        pair = f"{left} minus {right}"
        comparisons[pair] = {
            metric: paired_test(
                systems[left],
                systems[right],
                metric,
                args.bootstrap_samples,
                rng,
            )
            for metric in ("recall_at_2", "complete_at_2", "mrr")
        }

    result = {
        "protocol": {
            "bootstrap_samples": args.bootstrap_samples,
            "random_seed": args.seed,
            "qmsxe_seeds_averaged_per_example": True,
        },
        "summary": summaries,
        "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {len(summaries)} systems to {args.output}")


if __name__ == "__main__":
    main()

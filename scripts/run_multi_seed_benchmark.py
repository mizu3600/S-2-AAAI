from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev

import typer

from s2rag.benchmarks import load_benchmark
from s2rag.evaluation.experiment import BenchmarkExperimentRunner
from s2rag.evaluation.report import aggregate
from s2rag.evaluation.statistics import (
    cohens_d_paired,
    holm_adjust,
    paired_bootstrap_difference_ci,
    paired_randomization_pvalue,
)


PRIMARY_METRICS = (
    "passage_recall_at_5",
    "fact_recall_at_5",
    "answer_f1",
)


def main(
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(...),
    output_dir: Path = typer.Option(Path("data/experiments/multi_seed")),
    seeds: str = typer.Option("42", help="Exactly one evaluation seed"),
    limit: int = typer.Option(0, help="0 means all examples"),
    split: str = typer.Option("validation"),
    methods: str = typer.Option("bm25,dense,reified_fact_hybrid"),
    generate_methods: str = typer.Option("bm25,dense,reified_fact_hybrid"),
) -> None:
    parsed_seeds = [int(item.strip()) for item in seeds.split(",") if item.strip()]
    if len(parsed_seeds) != 1:
        raise typer.BadParameter(
            "exactly one seed is supported; use --seeds 42",
            param_hint="--seeds",
        )
    all_records = []
    from s2rag.evaluation.internal_baselines import (
        ALL_EXPERIMENT_METHODS,
        BENCHMARK_METHODS,
    )

    selected = (
        ALL_EXPERIMENT_METHODS
        if methods == "all_ablations"
        else BENCHMARK_METHODS
        if methods == "all"
        else tuple(item.strip() for item in methods.split(",") if item.strip())
    )
    generated = tuple(
        item.strip()
        for item in generate_methods.split(",")
        if item.strip() in selected
    )
    for seed in parsed_seeds:
        suite = load_benchmark(
            dataset,
            input_path,
            split=split,
            limit=None if limit == 0 else limit,
        )
        run_dir = output_dir / f"{dataset}_seed_{seed}"
        all_records.extend(
            BenchmarkExperimentRunner(
                methods=selected,
                generate_for_methods=generated,
            ).run(suite, run_dir, seed=seed)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(_collapse_records_across_seeds(all_records, expected_seeds=parsed_seeds))
    (output_dir / f"{dataset}_all_records.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / f"{dataset}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / f"{dataset}_seed_variation.json").write_text(
        json.dumps(
            _seed_run_variation(all_records),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_paired_statistics(all_records, output_dir / f"{dataset}_paired_statistics.json")
    typer.echo(f"wrote {len(all_records)} records across seeds {parsed_seeds}")


def _write_paired_statistics(records: list[dict], path: Path) -> None:
    expected_seeds = {int(record["seed"]) for record in records}
    by_example: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record in records:
        system_key = record["system"]
        system_values = by_example.setdefault(record["example_id"], {}).setdefault(system_key, {})
        for metric in PRIMARY_METRICS:
            value = record.get(metric)
            if value is not None:
                system_values.setdefault(metric, []).append(float(value))
    systems = sorted({record["system"] for record in records})
    comparisons: dict[str, dict] = {}
    all_raw_pvalues: dict[str, float] = {}
    all_pending: dict[str, tuple[str, str]] = {}
    for metric in PRIMARY_METRICS:
        eligible_examples = [
            (example_id, values)
            for example_id, values in by_example.items()
            if all(
                len(values.get(system, {}).get(metric, [])) == len(expected_seeds)
                for system in systems
            )
        ]
        paired_ids = [example_id for example_id, _ in eligible_examples]
        pending = {}
        for left, right in itertools.combinations(systems, 2):
            paired_left, paired_right = [], []
            for _, values in eligible_examples:
                left_values = values.get(left, {}).get(metric, [])
                right_values = values.get(right, {}).get(metric, [])
                paired_left.append(sum(left_values) / len(left_values))
                paired_right.append(sum(right_values) / len(right_values))
            if not paired_left:
                continue
            key = f"{left}_vs_{right}"
            global_key = f"{metric}:{key}"
            raw_pvalue = paired_randomization_pvalue(paired_left, paired_right)
            all_raw_pvalues[global_key] = raw_pvalue
            effect = cohens_d_paired(paired_left, paired_right)
            ci_low, ci_high = paired_bootstrap_difference_ci(paired_left, paired_right)
            pending[key] = {
                "n_questions": len(paired_left),
                "paired_example_ids_sha256": __import__("hashlib")
                .sha256("\n".join(sorted(paired_ids)).encode())
                .hexdigest(),
                "left_mean": sum(paired_left) / len(paired_left),
                "right_mean": sum(paired_right) / len(paired_right),
                "mean_difference": (
                    sum(paired_left) / len(paired_left) - sum(paired_right) / len(paired_right)
                ),
                "difference_ci95_low": ci_low,
                "difference_ci95_high": ci_high,
                "randomization_p": raw_pvalue,
                "cohens_d_paired": None if math.isnan(effect) else effect,
                "zero_variance": math.isnan(effect),
            }
            all_pending[global_key] = (metric, key)
        comparisons[metric] = pending
    adjusted = holm_adjust(all_raw_pvalues)
    for global_key, (metric, key) in all_pending.items():
        comparisons[metric][key]["holm_adjusted_p"] = adjusted[global_key]
    path.write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _seed_run_variation(records: list[dict]) -> dict:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for record in records:
        for metric in PRIMARY_METRICS:
            value = record.get(metric)
            if value is not None:
                grouped.setdefault(
                    (record["system"], metric, int(record["seed"])),
                    [],
                ).append(float(value))
    output: dict[str, dict[str, dict]] = {}
    by_system_metric: dict[tuple[str, str], list[dict]] = {}
    for (system, metric, seed), values in grouped.items():
        by_system_metric.setdefault((system, metric), []).append(
            {"seed": seed, "run_mean": mean(values), "n_questions": len(values)}
        )
    for (system, metric), runs in by_system_metric.items():
        runs.sort(key=lambda item: item["seed"])
        run_means = [item["run_mean"] for item in runs]
        output.setdefault(system, {})[metric] = {
            "runs": runs,
            "across_seed_mean": mean(run_means),
            "across_seed_sd": stdev(run_means) if len(run_means) > 1 else None,
            "n_seeds": len(runs),
        }
    return output


def _collapse_records_across_seeds(
    records: list[dict],
    expected_seeds: list[int],
) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault(
            (record["system"], record["example_id"]),
            [],
        ).append(record)
    collapsed = []
    for rows in grouped.values():
        item = dict(rows[0])
        item["seed"] = -1
        present_seeds = {int(row["seed"]) for row in rows}
        item["seed_complete"] = present_seeds == set(expected_seeds)
        numeric_names = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "seed"
        }
        for name in numeric_names:
            values = [row.get(name) for row in rows]
            item[name] = (
                sum(float(value) for value in values) / len(values)
                if item["seed_complete"] and all(value is not None for value in values)
                else None
            )
        collapsed.append(item)
    return collapsed


if __name__ == "__main__":
    typer.run(main)

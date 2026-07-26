from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.experiment import BenchmarkExperimentRunner
from qmshe.evaluation.report import aggregate
from qmshe.evaluation.statistics import (
    cohens_d_paired,
    holm_adjust,
    paired_randomization_pvalue,
)


PRIMARY_METRICS = (
    "passage_recall_at_5",
    "fact_recall_at_5",
    "fact_mrr",
    "answer_f1",
    "generated_fact_citation_f1",
    "joint_f1",
)


def main(
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(...),
    output_dir: Path = typer.Option(Path("data/experiments/multi_seed")),
    seeds: str = typer.Option("13,42,73"),
    limit: int = typer.Option(0, help="0 means all examples"),
    split: str = typer.Option("validation"),
    methods: str = typer.Option("bm25,dense,bm25_dense_rrf,reified_fact_hybrid"),
) -> None:
    parsed_seeds = [int(item.strip()) for item in seeds.split(",") if item.strip()]
    all_records = []
    from qmshe.evaluation.internal_baselines import INTERNAL_BASELINES

    selected = INTERNAL_BASELINES if methods == "all" else tuple(
        item.strip() for item in methods.split(",") if item.strip()
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
            BenchmarkExperimentRunner(methods=selected).run(suite, run_dir, seed=seed)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(_collapse_records_across_seeds(all_records))
    (output_dir / f"{dataset}_all_records.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / f"{dataset}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_paired_statistics(all_records, output_dir / f"{dataset}_paired_statistics.json")
    typer.echo(f"wrote {len(all_records)} records across seeds {parsed_seeds}")


def _write_paired_statistics(records: list[dict], path: Path) -> None:
    by_example: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record in records:
        system_values = by_example.setdefault(record["example_id"], {}).setdefault(
            record["system"], {}
        )
        for metric in PRIMARY_METRICS:
            value = record.get(metric)
            if value is not None:
                system_values.setdefault(metric, []).append(float(value))
    systems = sorted({record["system"] for record in records})
    comparisons: dict[str, dict] = {}
    for metric in PRIMARY_METRICS:
        raw_pvalues = {}
        pending = {}
        for left, right in itertools.combinations(systems, 2):
            paired_left, paired_right = [], []
            for values in by_example.values():
                left_values = values.get(left, {}).get(metric, [])
                right_values = values.get(right, {}).get(metric, [])
                if left_values and right_values:
                    paired_left.append(sum(left_values) / len(left_values))
                    paired_right.append(sum(right_values) / len(right_values))
            if not paired_left:
                continue
            key = f"{left}_vs_{right}"
            raw_pvalues[key] = paired_randomization_pvalue(
                paired_left, paired_right
            )
            effect = cohens_d_paired(paired_left, paired_right)
            pending[key] = {
                "n_questions": len(paired_left),
                "left_mean": sum(paired_left) / len(paired_left),
                "right_mean": sum(paired_right) / len(paired_right),
                "randomization_p": raw_pvalues[key],
                "cohens_d_paired": None if math.isnan(effect) else effect,
            }
        adjusted = holm_adjust(raw_pvalues)
        for key, values in pending.items():
            values["holm_adjusted_p"] = adjusted[key]
        comparisons[metric] = pending
    path.write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _collapse_records_across_seeds(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["system"], record["example_id"]), []).append(record)
    collapsed = []
    for rows in grouped.values():
        item = dict(rows[0])
        item["seed"] = -1
        numeric_names = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "seed"
        }
        for name in numeric_names:
            values = [float(row[name]) for row in rows if row.get(name) is not None]
            item[name] = sum(values) / len(values) if values else None
        collapsed.append(item)
    return collapsed


if __name__ == "__main__":
    typer.run(main)

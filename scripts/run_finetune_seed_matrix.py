from __future__ import annotations

import gc
import json
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev

import torch
import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.dual_mode import DualModeExperimentRunner, DualModeRecord
from qmshe.evaluation.local_models import LocalBGEEncoder, LocalBGEReranker
from qmshe.evaluation.splits import fixed_partition
from qmshe.graph.ordinary import GraphProfile


def main(
    embedding_model: Path = typer.Option(...), reranker_model: Path = typer.Option(...),
    embedding_adapters: str = typer.Option(..., help="seed=path comma-separated mapping"),
    reranker_adapters: str = typer.Option(..., help="seed=path comma-separated mapping"),
    seeds: str = typer.Option("13,42,73"), dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output_dir: Path = typer.Option(Path("reports/finetune_seed_matrix")),
    limit: int = typer.Option(500), validation_fraction: float = typer.Option(0.15),
    test_fraction: float = typer.Option(0.15), embedding_batch_size: int = typer.Option(32),
    reranker_batch_size: int = typer.Option(32), device: str = typer.Option("cuda"),
) -> None:
    seed_values = [int(item) for item in seeds.split(",") if item.strip()]
    embedding_paths = _parse_adapters(embedding_adapters)
    reranker_paths = _parse_adapters(reranker_adapters)
    missing = [seed for seed in seed_values if seed not in embedding_paths or seed not in reranker_paths]
    if missing:
        raise typer.BadParameter(f"missing embedding or reranker adapter for seeds: {missing}")
    suite = load_benchmark(dataset, input_path, split="test", limit=limit)
    partitions = fixed_partition(suite.examples, validation_fraction, test_fraction)
    suite.examples = partitions["test"]
    output_dir.mkdir(parents=True, exist_ok=True)
    import peft
    import transformers
    experiment_config = {
        "dataset": dataset, "input_path": str(input_path), "limit": limit,
        "fixed_partition_sizes": {name: len(items) for name, items in partitions.items()},
        "seeds": seed_values, "embedding_model": str(embedding_model),
        "reranker_model": str(reranker_model), "embedding_adapters": embedding_paths,
        "reranker_adapters": reranker_paths, "torch": torch.__version__,
        "transformers": transformers.__version__, "peft": peft.__version__,
        "cuda": torch.version.cuda, "device": device,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2), encoding="utf-8"
    )
    all_records: list[DualModeRecord] = []
    for seed in seed_values:
        for embedding_variant, embedding_adapter in (
            ("base", None), ("tuned", embedding_paths[seed]),
        ):
            encoder = LocalBGEEncoder(
                str(embedding_model), embedding_adapter, batch_size=embedding_batch_size,
                device=device,
            )
            for reranker_variant, reranker_adapter in (
                ("base", None), ("tuned", reranker_paths[seed]),
            ):
                reranker = LocalBGEReranker(
                    str(reranker_model), reranker_adapter, batch_size=reranker_batch_size,
                    device=device,
                )
                condition = f"seed_{seed}/{embedding_variant}_embedding__{reranker_variant}_reranker"
                runner = DualModeExperimentRunner(
                    encoder=encoder, reranker=reranker, seed=seed,
                    embedding_variant=embedding_variant, reranker_variant=reranker_variant,
                )
                records = runner.run(suite, output_dir / condition)
                all_records.extend(records)
                del runner, reranker
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    payload = [asdict(record) for record in all_records]
    (output_dir / "all_records.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = _aggregate(all_records)
    effects = _ablation_effects(summary)
    significance = _paired_significance(all_records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "significance.json").write_text(
        json.dumps(significance, indent=2), encoding="utf-8"
    )
    (output_dir / "ablation_effects.json").write_text(
        json.dumps(effects, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        _render(summary, effects, significance, len(suite.examples), seed_values), encoding="utf-8"
    )
    typer.echo(f"wrote {len(all_records)} records to {output_dir}")


def _parse_adapters(value: str) -> dict[int, str]:
    output = {}
    for item in value.split(","):
        seed, path = item.split("=", 1)
        output[int(seed)] = path
    return output


def _aggregate(records):
    grouped = {}
    for record in records:
        key = (record.embedding_variant, record.reranker_variant, record.mode, record.profile)
        grouped.setdefault(key, {}).setdefault(record.seed, []).append(record)
    metrics = (
        "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_30", "recall_at_40",
        "precision_at_5", "precision_at_10", "precision_at_20", "precision_at_30",
        "precision_at_40", "hit_at_5", "hit_at_10", "hit_at_20", "hit_at_30",
        "hit_at_40", "complete_at_5", "complete_at_10", "complete_at_20",
        "complete_at_30", "complete_at_40", "mrr", "ndcg_at_10", "bridge_recall_at_20",
    )
    output = {}
    for key, seed_groups in grouped.items():
        name = ":".join(map(str, key))
        output[name] = {"seed_count": len(seed_groups)}
        for metric in metrics:
            seed_means = [mean(getattr(item, metric) for item in items) for items in seed_groups.values()]
            output[name][f"{metric}_mean"] = mean(seed_means)
            output[name][f"{metric}_std"] = stdev(seed_means) if len(seed_means) > 1 else 0.0
            output[name][f"{metric}_by_seed"] = dict(zip(map(str, seed_groups), seed_means, strict=True))
    return output


def _paired_significance(records):
    lookup = {}
    for record in records:
        key = (record.embedding_variant, record.reranker_variant, record.seed, record.example_id)
        lookup.setdefault(key, {})[(record.mode, record.profile)] = record.recall_at_20
    comparisons = {
        "entity_relation_minus_hypergraph": ("graph", GraphProfile.ENTITY_RELATION.value),
        "reified_fact_minus_hypergraph": ("graph", GraphProfile.REIFIED_FACT.value),
    }
    output = {}
    for (embedding, reranker) in {
        (record.embedding_variant, record.reranker_variant) for record in records
    }:
        for label, target in comparisons.items():
            by_example = {}
            for key, values in lookup.items():
                if key[:2] != (embedding, reranker):
                    continue
                hyper = values.get(("hypergraph", "evidence_hypergraph"))
                graph = values.get(target)
                if hyper is not None and graph is not None:
                    by_example.setdefault(key[3], []).append(graph - hyper)
            diffs = [mean(values) for values in by_example.values()]
            output[f"{embedding}:{reranker}:{label}"] = _randomization_test(diffs)
    return output


def _ablation_effects(summary):
    systems = {
        (name.split(":", 3)[2], name.split(":", 3)[3]) for name in summary
    }
    output = {}
    for mode, profile in systems:
        cells = {
            (embedding, reranker): summary[f"{embedding}:{reranker}:{mode}:{profile}"]
            for embedding in ("base", "tuned") for reranker in ("base", "tuned")
        }
        output[f"{mode}:{profile}"] = {}
        for metric in ("recall_at_20", "mrr", "ndcg_at_10"):
            bb = cells[("base", "base")][f"{metric}_mean"]
            tb = cells[("tuned", "base")][f"{metric}_mean"]
            bt = cells[("base", "tuned")][f"{metric}_mean"]
            tt = cells[("tuned", "tuned")][f"{metric}_mean"]
            output[f"{mode}:{profile}"][metric] = {
                "embedding_only": tb - bb, "reranker_only": bt - bb,
                "joint": tt - bb, "interaction": tt - tb - bt + bb,
            }
    return output


def _randomization_test(differences, samples=10000):
    rng = random.Random(20260721)
    observed = mean(differences)
    permutations = [
        mean(value if rng.random() < 0.5 else -value for value in differences)
        for _ in range(samples)
    ]
    bootstrap = [
        mean(rng.choice(differences) for _ in differences) for _ in range(samples)
    ]
    bootstrap.sort()
    return {
        "paired_samples": len(differences), "mean_difference": observed,
        "ci95_low": bootstrap[int(samples * 0.025)],
        "ci95_high": bootstrap[int(samples * 0.975)],
        "randomization_p_two_sided": (
            1 + sum(abs(value) >= abs(observed) for value in permutations)
        ) / (samples + 1),
    }


def _render(summary, effects, significance, examples, seeds):
    rows = [
        "# Embedding/Reranker fine-tuning × random-seed ablation", "",
        f"Held-out examples: {examples}; seeds: {', '.join(map(str, seeds))}", "",
        "All models use the same fixed ID-based train/validation/test partition. Mean±SD is across seeds.", "",
        "| Embedding | Reranker | System | R@20 | MRR | nDCG@10 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name, item in summary.items():
        embedding, reranker, mode, profile = name.split(":", 3)
        rows.append(
            f"| {embedding} | {reranker} | {mode}/{profile} | "
            f"{item['recall_at_20_mean']:.4f}±{item['recall_at_20_std']:.4f} | "
            f"{item['mrr_mean']:.4f}±{item['mrr_std']:.4f} | "
            f"{item['ndcg_at_10_mean']:.4f}±{item['ndcg_at_10_std']:.4f} |"
        )
    rows.extend([
        "", "## Multi-k retrieval metrics", "",
        "| Embedding | Reranker | System | R@5 | R@10 | R@20 | R@30 | R@40 | "
        "P@10 | P@20 | Hit@10 | Complete@20 | Complete@40 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, item in summary.items():
        embedding, reranker, mode, profile = name.split(":", 3)
        rows.append(
            f"| {embedding} | {reranker} | {mode}/{profile} | "
            f"{item['recall_at_5_mean']:.4f} | {item['recall_at_10_mean']:.4f} | "
            f"{item['recall_at_20_mean']:.4f} | {item['recall_at_30_mean']:.4f} | "
            f"{item['recall_at_40_mean']:.4f} | {item['precision_at_10_mean']:.4f} | "
            f"{item['precision_at_20_mean']:.4f} | {item['hit_at_10_mean']:.4f} | "
            f"{item['complete_at_20_mean']:.4f} | {item['complete_at_40_mean']:.4f} |"
        )
    rows.extend(["", "## Recall@20 ablation effects", "",
                 "| System | Embedding only | Reranker only | Joint | Interaction |",
                 "|---|---:|---:|---:|---:|"])
    for name, metrics in effects.items():
        item = metrics["recall_at_20"]
        rows.append(
            f"| {name} | {item['embedding_only']:+.4f} | {item['reranker_only']:+.4f} | "
            f"{item['joint']:+.4f} | {item['interaction']:+.4f} |"
        )
    rows.extend(["", "## Paired randomization tests on Recall@20", "",
                 "| Condition | Difference | 95% bootstrap CI | p |", "|---|---:|---:|---:|"])
    for name, item in significance.items():
        rows.append(
            f"| {name} | {item['mean_difference']:.4f} | "
            f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | "
            f"{item['randomization_p_two_sided']:.4f} |"
        )
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    typer.run(main)

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as functional
import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.local_models import LocalBGEEncoder
from qmshe.evaluation.retrieval_metrics import (
    complete_at_k,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from qmshe.evaluation.splits import fixed_partition
from qmshe.training.inductive_stage_a import _prepare, _rank_facts, _score_graph


KS = (1, 2, 5, 10, 20, 30, 40)
CONDITIONS = (
    "dense_bge_identity",
    "untrained_raw",
    "trained_raw",
    "untrained_full",
    "trained_full",
)
COMPARISONS = {
    "raw_training_effect": ("trained_raw", "untrained_raw"),
    "full_training_effect": ("trained_full", "untrained_full"),
    "untrained_band_effect": ("untrained_full", "untrained_raw"),
    "trained_band_effect": ("trained_full", "trained_raw"),
    "trained_full_vs_dense_bge": ("trained_full", "dense_bge_identity"),
}


def main(
    base_model: Path = typer.Option(...),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    raw_checkpoint_root: Path = typer.Option(Path("data/models/core_ablation")),
    full_checkpoint_root: Path = typer.Option(Path("data/models/stage_ab")),
    output_dir: Path = typer.Option(Path("reports/stage_a_training_effect")),
    dataset: str = typer.Option("hotpotqa"),
    limit: int = typer.Option(500),
    seeds: str = typer.Option("13,42,73"),
    device: str = typer.Option("cuda"),
    embedding_batch_size: int = typer.Option(32),
) -> None:
    suite = load_benchmark(dataset, input_path, split="test", limit=limit)
    test_examples = fixed_partition(suite.examples)["test"]
    seed_values = [int(item) for item in seeds.split(",") if item.strip()]
    records = []
    for seed in seed_values:
        random.seed(seed)
        torch.manual_seed(seed)
        encoder = LocalBGEEncoder(str(base_model), batch_size=embedding_batch_size, device=device)
        graphs, template, roles = _prepare(
            test_examples, encoder, "hypergraph", "evidence_hypergraph", seed, "full"
        )
        initial_model = _clone_state(template.model.state_dict())
        initial_relation_gate = _clone_state(template.relation_gate.state_dict())
        states = {
            "untrained_raw": (initial_model, initial_relation_gate, "raw_only"),
            "untrained_full": (initial_model, initial_relation_gate, "full"),
            "trained_raw": (
                *_checkpoint(
                    raw_checkpoint_root / "raw_only" / f"seed_{seed}" / "stage_a.pt"
                ),
                "raw_only",
            ),
            "trained_full": (
                *_checkpoint(
                    full_checkpoint_root
                    / "hypergraph_evidence_hypergraph"
                    / f"seed_{seed}"
                    / "stage_a.pt"
                ),
                "full",
            ),
        }
        records.extend(_evaluate_dense(graphs, seed, device))
        for condition in CONDITIONS[1:]:
            model_state, relation_state, variant = states[condition]
            records.extend(
                _evaluate_stage_a(
                    graphs,
                    template,
                    roles,
                    model_state,
                    relation_state,
                    variant,
                    condition,
                    seed,
                    device,
                )
            )
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = _aggregate(records)
    effects = _paired_effects(records)
    manifest = {
        "track": "stage_a_training_effect_2x2",
        "held_out_examples": len(test_examples),
        "seeds": seed_values,
        "ks": list(KS),
        "architecture_matched_initialization": True,
        "same_initialization_within_seed": True,
        "dense_bge_identity_is_reference_not_a_2x2_cell": True,
        "no_bm25_graph_rerank_or_neural_reranker": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render(summary, effects, manifest), encoding="utf-8")
    typer.echo(f"wrote {len(records)} records")


def _clone_state(state):
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["model"], payload["relation_gate"]


def _evaluate_dense(graphs, seed, device):
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    rows = []
    with torch.no_grad():
        for graph in graphs:
            _synchronize(target)
            started = time.perf_counter()
            query = functional.normalize(graph.query_vector.to(target), dim=0)
            nodes = functional.normalize(graph.raw_features.to(target), dim=1)
            scores = nodes @ query
            ranking = torch.argsort(scores, descending=True).cpu().tolist()
            _synchronize(target)
            rows.append(
                _record(
                    graph,
                    _rank_facts([graph.node_ids[index] for index in ranking], graph),
                    "dense_bge_identity",
                    seed,
                    time.perf_counter() - started,
                )
            )
    return rows


def _evaluate_stage_a(
    graphs,
    template,
    roles,
    model_state,
    relation_state,
    variant,
    condition,
    seed,
    device,
):
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    model = template.model.to(target)
    relation_gate = template.relation_gate.to(target)
    model.load_state_dict(model_state)
    relation_gate.load_state_dict(relation_state)
    model.eval()
    relation_gate.eval()
    rows = []
    with torch.no_grad():
        for graph in graphs:
            _synchronize(target)
            started = time.perf_counter()
            scores = _score_graph(
                model,
                relation_gate,
                graph,
                graph.query_vector.to(target),
                roles,
                target,
                variant,
            )
            ranking = torch.argsort(scores, descending=True).cpu().tolist()
            _synchronize(target)
            rows.append(
                _record(
                    graph,
                    _rank_facts([graph.node_ids[index] for index in ranking], graph),
                    condition,
                    seed,
                    time.perf_counter() - started,
                )
            )
    return rows


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _record(graph, ranking, condition, seed, elapsed):
    row = {
        "example_id": graph.example_id,
        "condition": condition,
        "seed": seed,
        "mrr": reciprocal_rank(ranking, graph.gold_fact_ids),
        "ndcg_at_10": ndcg_at_k(ranking, graph.gold_fact_ids, 10),
        "seconds": elapsed,
    }
    for k in KS:
        row[f"recall_at_{k}"] = recall_at_k(ranking, graph.gold_fact_ids, k)
        row[f"hit_at_{k}"] = hit_at_k(ranking, graph.gold_fact_ids, k)
        row[f"complete_at_{k}"] = complete_at_k(ranking, graph.gold_fact_ids, k)
    return row


def _aggregate(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[row["condition"]][row["seed"]].append(row)
    metrics = ["mrr", "ndcg_at_10", "seconds"] + [
        f"{name}_at_{k}" for k in KS for name in ("recall", "hit", "complete")
    ]
    output = {}
    for condition, seed_rows in sorted(grouped.items()):
        item = {"condition": condition, "seed_count": len(seed_rows)}
        for metric in metrics:
            values = [mean(row[metric] for row in rows) for rows in seed_rows.values()]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output[condition] = item
    return output


def _paired_effects(records):
    indexed = defaultdict(dict)
    for row in records:
        indexed[(row["seed"], row["example_id"])][row["condition"]] = row
    rng = random.Random(20260722)
    output = {}
    for name, (treatment, control) in COMPARISONS.items():
        differences = [
            values[treatment]["recall_at_20"] - values[control]["recall_at_20"]
            for values in indexed.values()
            if treatment in values and control in values
        ]
        observed = mean(differences)
        samples = 10000
        permutations = [
            mean(value if rng.random() < 0.5 else -value for value in differences)
            for _ in range(samples)
        ]
        bootstrap = sorted(
            mean(rng.choice(differences) for _ in differences) for _ in range(samples)
        )
        output[name] = {
            "treatment": treatment,
            "control": control,
            "recall_at_20_delta": observed,
            "ci95_low": bootstrap[int(samples * 0.025)],
            "ci95_high": bootstrap[int(samples * 0.975)],
            "p_two_sided": (1 + sum(abs(value) >= abs(observed) for value in permutations))
            / (samples + 1),
            "paired_rows": len(differences),
        }
    return output


def _render(summary, effects, manifest):
    lines = [
        "# Stage A training-effect 2×2 control",
        "",
        "All cells rank Facts directly before BM25, graph reranking and the neural reranker.",
        "",
        "| Condition | R@5 | R@10 | R@20 mean±SD | R@40 | Hit@10 | Complete@20 | MRR | nDCG@10 | ms/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = summary[condition]
        lines.append(
            f"| {condition} | {item['recall_at_5_mean']:.4f} | "
            f"{item['recall_at_10_mean']:.4f} | "
            f"{item['recall_at_20_mean']:.4f}±{item['recall_at_20_std']:.4f} | "
            f"{item['recall_at_40_mean']:.4f} | {item['hit_at_10_mean']:.4f} | "
            f"{item['complete_at_20_mean']:.4f} | {item['mrr_mean']:.4f} | "
            f"{item['ndcg_at_10_mean']:.4f} | {item['seconds_mean'] * 1000:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired effects on Recall@20",
            "",
            "| Question | Treatment − control | Delta | 95% CI | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for name, item in effects.items():
        lines.append(
            f"| {name} | {item['treatment']} − {item['control']} | "
            f"{item['recall_at_20_delta']:+.4f} | "
            f"[{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}] | "
            f"{item['p_two_sided']:.4f} |"
        )
    lines.extend(["", "## Manifest", "", "```json", json.dumps(manifest, indent=2), "```", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    typer.run(main)

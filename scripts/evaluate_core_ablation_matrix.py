from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch
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
VARIANTS = (
    "full",
    "raw_only",
    "no_low",
    "no_mid",
    "no_high",
    "fixed_gate",
    "no_role_gate",
    "no_semantic_graph",
    "no_bridge_loss",
    "no_hard_negatives",
)


def main(
    base_model: Path = typer.Option(...),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    checkpoint_root: Path = typer.Option(Path("data/models/core_ablation")),
    full_checkpoint_root: Path = typer.Option(Path("data/models/stage_ab")),
    output_dir: Path = typer.Option(Path("reports/core_ablation_matrix")),
    dataset: str = typer.Option("hotpotqa"),
    limit: int = typer.Option(500),
    seeds: str = typer.Option("13,42,73"),
    device: str = typer.Option("cuda"),
    embedding_batch_size: int = typer.Option(32),
    variants: str = typer.Option(",".join(VARIANTS)),
) -> None:
    suite = load_benchmark(dataset, input_path, split="test", limit=limit)
    test_examples = fixed_partition(suite.examples)["test"]
    seed_values = [int(item) for item in seeds.split(",") if item.strip()]
    selected_variants = [item.strip() for item in variants.split(",") if item.strip()]
    unknown = sorted(set(selected_variants) - set(VARIANTS))
    if unknown:
        raise typer.BadParameter(f"unknown variants: {unknown}")
    records = []
    for seed in seed_values:
        encoder = LocalBGEEncoder(str(base_model), batch_size=embedding_batch_size, device=device)
        hyper_graphs, hyper_template, roles = _prepare(
            test_examples, encoder, "hypergraph", "evidence_hypergraph", seed, "full"
        )
        no_semantic_graphs, no_semantic_template, no_semantic_roles = _prepare(
            test_examples,
            encoder,
            "hypergraph",
            "evidence_hypergraph",
            seed,
            "no_semantic_graph",
        )
        for variant in selected_variants:
            checkpoint = _checkpoint(
                full_checkpoint_root,
                checkpoint_root,
                "hypergraph",
                "evidence_hypergraph",
                variant,
                seed,
            )
            graphs, template, active_roles = (
                (no_semantic_graphs, no_semantic_template, no_semantic_roles)
                if variant == "no_semantic_graph"
                else (hyper_graphs, hyper_template, roles)
            )
            records.extend(
                _evaluate_condition(
                    graphs,
                    template,
                    active_roles,
                    checkpoint,
                    variant,
                    seed,
                    "hypergraph:evidence_hypergraph",
                    device,
                )
            )
        for profile in ("entity_relation", "reified_fact"):
            graphs, template, graph_roles = _prepare(
                test_examples, encoder, "graph", profile, seed, "full"
            )
            checkpoint = _checkpoint(
                full_checkpoint_root, checkpoint_root, "graph", profile, "full", seed
            )
            records.extend(
                _evaluate_condition(
                    graphs,
                    template,
                    graph_roles,
                    checkpoint,
                    "full",
                    seed,
                    f"graph:{profile}",
                    device,
                )
            )
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _aggregate(records)
    effects = _paired_effects(records, selected_variants)
    manifest = {
        "track": "stage_a_intrinsic_pre_fusion",
        "held_out_examples": len(test_examples),
        "seeds": seed_values,
        "ks": list(KS),
        "fixed_encoder": str(base_model),
        "no_bm25_or_reranker_in_this_track": True,
        "note": "Training variants are independently trained; runtime fusion ablations are reported separately.",
    }
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render(summary, effects, manifest), encoding="utf-8")
    typer.echo(f"wrote {len(records)} records")


def _checkpoint(full_root, variant_root, mode, profile, variant, seed):
    if variant == "full":
        path = full_root / f"{mode}_{profile}" / f"seed_{seed}" / "stage_a.pt"
    else:
        path = variant_root / variant / f"seed_{seed}" / "stage_a.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _evaluate_condition(graphs, template, roles, checkpoint, variant, seed, system, device):
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    model = template.model.to(target)
    model.load_state_dict(checkpoint["model"])
    relation_gate = template.relation_gate.to(target) if system.startswith("hypergraph") else None
    if relation_gate is not None:
        relation_gate.load_state_dict(checkpoint["relation_gate"])
        relation_gate.eval()
    model.eval()
    rows = []
    with torch.no_grad():
        for graph in graphs:
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
            facts = _rank_facts([graph.node_ids[index] for index in ranking], graph)
            elapsed = time.perf_counter() - started
            row = {
                "example_id": graph.example_id,
                "system": system,
                "variant": variant,
                "seed": seed,
                "mrr": reciprocal_rank(facts, graph.gold_fact_ids),
                "ndcg_at_10": ndcg_at_k(facts, graph.gold_fact_ids, 10),
                "spectral_seconds": elapsed,
            }
            for k in KS:
                row[f"recall_at_{k}"] = recall_at_k(facts, graph.gold_fact_ids, k)
                row[f"hit_at_{k}"] = hit_at_k(facts, graph.gold_fact_ids, k)
                row[f"complete_at_{k}"] = complete_at_k(facts, graph.gold_fact_ids, k)
            rows.append(row)
    return rows


def _aggregate(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[(row["system"], row["variant"])][row["seed"]].append(row)
    metrics = ["mrr", "ndcg_at_10", "spectral_seconds"] + [
        f"{name}_at_{k}" for k in KS for name in ("recall", "hit", "complete")
    ]
    output = {}
    for (system, variant), seed_rows in sorted(grouped.items()):
        item = {"system": system, "variant": variant, "seed_count": len(seed_rows)}
        for metric in metrics:
            values = [mean(row[metric] for row in rows) for rows in seed_rows.values()]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output[f"{system}:{variant}"] = item
    return output


def _paired_effects(records, variants):
    indexed = defaultdict(dict)
    for row in records:
        if row["system"] == "hypergraph:evidence_hypergraph":
            indexed[(row["seed"], row["example_id"])][row["variant"]] = row
    output = {}
    rng = random.Random(20260722)
    for variant in variants:
        if variant == "full":
            continue
        differences = []
        for values in indexed.values():
            if "full" in values and variant in values:
                differences.append(values[variant]["recall_at_20"] - values["full"]["recall_at_20"])
        observed = mean(differences)
        samples = 10000
        permutations = [
            mean(value if rng.random() < 0.5 else -value for value in differences)
            for _ in range(samples)
        ]
        bootstrap = sorted(
            mean(rng.choice(differences) for _ in differences) for _ in range(samples)
        )
        output[variant] = {
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
        "# Core Stage A ablation matrix",
        "",
        "This is the intrinsic spectral-ranking track before BM25, graph reranking and neural reranking.",
        "",
        "| System | Variant | R@5 | R@10 | R@20 | R@40 | Hit@10 | Complete@20 | MRR | nDCG@10 | ms/query |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.values():
        lines.append(
            f"| {item['system']} | {item['variant']} | "
            f"{item['recall_at_5_mean']:.4f} | {item['recall_at_10_mean']:.4f} | "
            f"{item['recall_at_20_mean']:.4f} | {item['recall_at_40_mean']:.4f} | "
            f"{item['hit_at_10_mean']:.4f} | {item['complete_at_20_mean']:.4f} | "
            f"{item['mrr_mean']:.4f} | {item['ndcg_at_10_mean']:.4f} | "
            f"{item['spectral_seconds_mean'] * 1000:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired effects versus Full on Recall@20",
            "",
            "| Variant | Delta | 95% CI | p |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant, item in effects.items():
        lines.append(
            f"| {variant} | {item['recall_at_20_delta']:+.4f} | "
            f"[{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}] | "
            f"{item['p_two_sided']:.4f} |"
        )
    lines.extend(["", "## Manifest", "", "```json", json.dumps(manifest, indent=2), "```", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    typer.run(main)

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
from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.evaluation.local_models import LocalBGEEncoder, LocalBGEReranker
from qmshe.evaluation.retrieval_metrics import (
    complete_at_k,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from qmshe.evaluation.splits import fixed_partition
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline


KS = (1, 2, 5, 10, 20, 30, 40)
RUNTIME_VARIANTS = {
    "full": {"use_graph_rerank": True, "use_bm25": True, "dense_only": False},
    "no_graph_rerank": {
        "use_graph_rerank": False,
        "use_bm25": True,
        "dense_only": False,
    },
    "dense_only": {
        "use_graph_rerank": False,
        "use_bm25": False,
        "dense_only": True,
    },
    "no_bm25": {"use_graph_rerank": True, "use_bm25": False, "dense_only": False},
}


def main(
    embedding_model: Path = typer.Option(...),
    reranker_model: Path = typer.Option(...),
    reranker_adapters: str = typer.Option(...),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    checkpoint_root: Path = typer.Option(Path("data/models/stage_ab")),
    output_dir: Path = typer.Option(Path("reports/runtime_ablation_matrix")),
    dataset: str = typer.Option("hotpotqa"),
    limit: int = typer.Option(500),
    seeds: str = typer.Option("13,42,73"),
    device: str = typer.Option("cuda"),
    embedding_batch_size: int = typer.Option(16),
    reranker_batch_size: int = typer.Option(8),
) -> None:
    suite = load_benchmark(dataset, input_path, split="test", limit=limit)
    test_examples = fixed_partition(suite.examples)["test"]
    adapters = _mapping(reranker_adapters)
    seed_values = [int(item) for item in seeds.split(",") if item.strip()]
    records = []
    for seed in seed_values:
        encoder = LocalBGEEncoder(
            str(embedding_model), batch_size=embedding_batch_size, device=device
        )
        reranker = LocalBGEReranker(
            str(reranker_model),
            adapters[seed],
            batch_size=reranker_batch_size,
            device=device,
        )
        checkpoints = {
            ("hypergraph", "evidence_hypergraph"): _load_checkpoint(
                checkpoint_root, "hypergraph", "evidence_hypergraph", seed
            ),
            **{
                ("graph", profile): _load_checkpoint(checkpoint_root, "graph", profile, seed)
                for profile in ("entity_relation", "reified_fact")
            },
        }
        for number, example in enumerate(test_examples, 1):
            built = build_example_corpus(example)
            pipelines = {
                "hypergraph:evidence_hypergraph": QMSHEPipeline(
                    built.corpus,
                    text_encoder=encoder,
                    reranker=reranker,
                    seed=seed,
                    enable_remote_reranker=False,
                ),
                **{
                    f"graph:{profile}": QMSGEGraphPipeline(
                        built.corpus,
                        text_encoder=encoder,
                        reranker=reranker,
                        profile=GraphProfile(profile),
                        seed=seed,
                        enable_remote_reranker=False,
                    )
                    for profile in ("entity_relation", "reified_fact")
                },
            }
            pipelines["hypergraph:evidence_hypergraph"].load_stage_a_checkpoint(
                checkpoints[("hypergraph", "evidence_hypergraph")]
            )
            for profile in ("entity_relation", "reified_fact"):
                pipelines[f"graph:{profile}"].load_stage_a_checkpoint(
                    checkpoints[("graph", profile)]
                )
            for system, pipeline in pipelines.items():
                pipeline.generator.client = None
                for variant, flags in RUNTIME_VARIANTS.items():
                    for name, value in flags.items():
                        setattr(pipeline, name, value)
                    pipeline.query_cache.clear()
                    records.append(
                        _query_record(
                            example.example_id,
                            example.question,
                            system,
                            variant,
                            seed,
                            pipeline,
                            built,
                        )
                    )
                if system.startswith("graph:"):
                    pipeline.use_graph_rerank = True
                    pipeline.use_bm25 = True
                    pipeline.dense_only = False
                    for strategy in ("single", "multi", "hybrid"):
                        pipeline.index_strategy = strategy
                        pipeline.query_cache.clear()
                        records.append(
                            _query_record(
                                example.example_id,
                                example.question,
                                system,
                                f"index_{strategy}",
                                seed,
                                pipeline,
                                built,
                            )
                        )
            if number % 10 == 0:
                typer.echo(f"seed={seed} {number}/{len(test_examples)}")
        del encoder, reranker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _aggregate(records)
    effects = _paired_effects(records)
    manifest = {
        "track": "end_to_end_retrieval_runtime_ablation",
        "held_out_examples": len(test_examples),
        "seeds": seed_values,
        "fixed_tuned_reranker": True,
        "generator_disabled": True,
        "candidate_budget": 60,
        "top_k": 40,
        "timing_cache_policy": "query_embedding_and_reranker_caches_cleared_per_condition",
        "cuda_synchronized": True,
    }
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render(summary, effects, manifest), encoding="utf-8")
    typer.echo(f"wrote {len(records)} records")


def _query_record(example_id, question, system, variant, seed, pipeline, built):
    # Every condition pays for its own query encoding and reranker scoring.  The
    # local providers intentionally cache both, which is useful in production
    # but would otherwise make later ablation cells look artificially cheap.
    query_cache = getattr(pipeline.text_encoder, "_query_cache", None)
    if query_cache is not None:
        query_cache.clear()
    reranker_cache = getattr(pipeline.reranker, "_cache", None)
    if reranker_cache is not None:
        reranker_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = pipeline.query(question, 40, False, 60)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    facts = (
        result.retrieved_hyperedges if system.startswith("hypergraph") else result.retrieved_facts
    )
    gold = built.gold_fact_ids
    chunk_to_document = {chunk.chunk_id: chunk.document_id for chunk in built.corpus.chunks}
    fact_to_document = {
        fact.hyperedge_id: chunk_to_document[fact.evidence_chunk_ids[0]]
        for fact in built.corpus.evidence_hyperedges
        if fact.evidence_chunk_ids and fact.evidence_chunk_ids[0] in chunk_to_document
    }
    documents = list(
        dict.fromkeys(fact_to_document[item] for item in facts if item in fact_to_document)
    )
    gold_documents = {fact_to_document[item] for item in gold if item in fact_to_document}
    path_em, path_precision, path_recall, path_f1 = _set_scores(set(documents[:2]), gold_documents)
    row = {
        "example_id": example_id,
        "system": system,
        "variant": variant,
        "seed": seed,
        "seconds": time.perf_counter() - started,
        "mrr": reciprocal_rank(facts, gold),
        "ndcg_at_10": ndcg_at_k(facts, gold, 10),
        "passage_mrr": reciprocal_rank(documents, gold_documents),
        "path_em": path_em,
        "path_precision": path_precision,
        "path_recall": path_recall,
        "path_f1": path_f1,
    }
    row.update({f"band_{name}": value for name, value in result.band_weights.items()})
    for k in KS:
        row[f"recall_at_{k}"] = recall_at_k(facts, gold, k)
        row[f"hit_at_{k}"] = hit_at_k(facts, gold, k)
        row[f"complete_at_{k}"] = complete_at_k(facts, gold, k)
        row[f"passage_recall_at_{k}"] = recall_at_k(documents, gold_documents, k)
        row[f"passage_hit_at_{k}"] = hit_at_k(documents, gold_documents, k)
        row[f"passage_complete_at_{k}"] = complete_at_k(documents, gold_documents, k)
    return row


def _load_checkpoint(root, mode, profile, seed):
    return torch.load(
        root / f"{mode}_{profile}" / f"seed_{seed}" / "stage_a.pt",
        map_location="cpu",
        weights_only=True,
    )


def _mapping(value):
    return {int(seed): path for seed, path in (item.split("=", 1) for item in value.split(","))}


def _aggregate(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[(row["system"], row["variant"])][row["seed"]].append(row)
    metrics = (
        [
            "mrr",
            "ndcg_at_10",
            "passage_mrr",
            "path_em",
            "path_precision",
            "path_recall",
            "path_f1",
            "seconds",
            "band_raw",
            "band_low",
            "band_mid",
            "band_high",
        ]
        + [f"{name}_at_{k}" for k in KS for name in ("recall", "hit", "complete")]
        + [f"passage_{name}_at_{k}" for k in KS for name in ("recall", "hit", "complete")]
    )
    output = {}
    for (system, variant), by_seed in sorted(grouped.items()):
        item = {"system": system, "variant": variant, "seed_count": len(by_seed)}
        for metric in metrics:
            values = [mean(row[metric] for row in rows) for rows in by_seed.values()]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output[f"{system}:{variant}"] = item
    return output


def _paired_effects(records):
    indexed = defaultdict(dict)
    for row in records:
        indexed[(row["system"], row["seed"], row["example_id"])][row["variant"]] = row
    rng = random.Random(20260722)
    output = {}
    variants = sorted({row["variant"] for row in records} - {"full", "index_hybrid"})
    for system in sorted({row["system"] for row in records}):
        for variant in variants:
            baseline = "index_hybrid" if variant.startswith("index_") else "full"
            differences = []
            for (row_system, _, _), values in indexed.items():
                if row_system == system and baseline in values and variant in values:
                    differences.append(
                        values[variant]["recall_at_20"] - values[baseline]["recall_at_20"]
                    )
            if not differences:
                continue
            observed = mean(differences)
            samples = 10000
            permutations = [
                mean(value if rng.random() < 0.5 else -value for value in differences)
                for _ in range(samples)
            ]
            bootstrap = sorted(
                mean(rng.choice(differences) for _ in differences) for _ in range(samples)
            )
            output[f"{system}:{variant}"] = {
                "system": system,
                "variant": variant,
                "baseline": baseline,
                "recall_at_20_delta": observed,
                "ci95_low": bootstrap[int(samples * 0.025)],
                "ci95_high": bootstrap[int(samples * 0.975)],
                "p_two_sided": (1 + sum(abs(value) >= abs(observed) for value in permutations))
                / (samples + 1),
            }
    return output


def _set_scores(predicted, gold):
    precision = len(predicted & gold) / len(predicted) if predicted else float(not gold)
    recall = len(predicted & gold) / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(predicted == gold), precision, recall, f1


def _render(summary, effects, manifest):
    lines = [
        "# Runtime retrieval ablation matrix",
        "",
        "| System | Variant | Fact R@5 | Fact R@10 | Fact R@20 | Fact R@40 | Fact MRR | Passage R@5 | Passage MRR | Path F1 | Raw | Low | Mid | High | s/query |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.values():
        lines.append(
            f"| {item['system']} | {item['variant']} | {item['recall_at_5_mean']:.4f} | "
            f"{item['recall_at_10_mean']:.4f} | {item['recall_at_20_mean']:.4f} | "
            f"{item['recall_at_40_mean']:.4f} | {item['mrr_mean']:.4f} | "
            f"{item['passage_recall_at_5_mean']:.4f} | "
            f"{item['passage_mrr_mean']:.4f} | {item['path_f1_mean']:.4f} | "
            f"{item['band_raw_mean']:.4f} | {item['band_low_mean']:.4f} | "
            f"{item['band_mid_mean']:.4f} | {item['band_high_mean']:.4f} | "
            f"{item['seconds_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired effects on Recall@20",
            "",
            "| System | Variant | Baseline | Delta | 95% CI | p |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for item in effects.values():
        lines.append(
            f"| {item['system']} | {item['variant']} | {item['baseline']} | "
            f"{item['recall_at_20_delta']:+.4f} | "
            f"[{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}] | "
            f"{item['p_two_sided']:.4f} |"
        )
    lines.extend(["", "## Manifest", "", "```json", json.dumps(manifest, indent=2), "```", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    typer.run(main)

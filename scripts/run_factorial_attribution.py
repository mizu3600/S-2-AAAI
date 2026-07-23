from __future__ import annotations

import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import typer

from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.embedding.text_encoder import encode_queries
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
from qmshe.graph_pipeline import QMSGEGraphPipeline, _weighted_band_fusion
from qmshe.pipeline import QMSHEPipeline
from qmshe.retrieval.ann_retriever import ExactVectorIndex
from qmshe.retrieval.graph_reranker import graph_rerank
from qmshe.retrieval.seed_retriever import reciprocal_rank_fusion


KS = (1, 2, 5, 10, 20, 30, 40)
GRAPH_FACTORS = (
    "entity_expansion",
    "bm25",
    "spectral",
    "graph_rerank",
    "base_reranker",
)
HYPERGRAPH_FACTORS = ("bm25", "spectral", "graph_rerank", "base_reranker")
ATTRIBUTION_METRICS = ("recall_at_20", "mrr", "passage_recall_at_5", "path_f1")


def main(
    embedding_model: Path = typer.Option(...),
    reranker_model: Path = typer.Option(...),
    reranker_adapters: str = typer.Option(...),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    checkpoint_root: Path = typer.Option(Path("data/models/stage_ab")),
    output_dir: Path = typer.Option(Path("reports/factorial_attribution")),
    dataset: str = typer.Option("hotpotqa"),
    limit: int = typer.Option(500),
    seeds: str = typer.Option("13,42,73"),
    device: str = typer.Option("cuda"),
    embedding_batch_size: int = typer.Option(16),
    reranker_batch_size: int = typer.Option(8),
    candidate_budget: int = typer.Option(60),
    top_k: int = typer.Option(40),
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
        base_reranker = LocalBGEReranker(
            str(reranker_model), batch_size=reranker_batch_size, device=device
        )
        lora_reranker = LocalBGEReranker(
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
            base_reranker._cache.clear()
            lora_reranker._cache.clear()
            built = build_example_corpus(example)
            pipelines = {
                "hypergraph:evidence_hypergraph": QMSHEPipeline(
                    built.corpus,
                    text_encoder=encoder,
                    seed=seed,
                    enable_remote_reranker=False,
                ),
                **{
                    f"graph:{profile}": QMSGEGraphPipeline(
                        built.corpus,
                        text_encoder=encoder,
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
                factors = HYPERGRAPH_FACTORS if system.startswith("hypergraph") else GRAPH_FACTORS
                sources = (
                    _hypergraph_sources(pipeline, example.question, candidate_budget)
                    if system.startswith("hypergraph")
                    else _graph_sources(pipeline, example.question, candidate_budget)
                )
                records.extend(
                    _evaluate_factorial(
                        example.example_id,
                        example.question,
                        system,
                        seed,
                        pipeline,
                        sources,
                        factors,
                        built,
                        base_reranker,
                        lora_reranker,
                        candidate_budget,
                        top_k,
                    )
                )
            if number % 10 == 0:
                typer.echo(f"seed={seed} {number}/{len(test_examples)}")
        del encoder, base_reranker, lora_reranker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = _aggregate_conditions(records)
    shapley = _shapley_report(records)
    interactions = _interaction_report(records)
    lora_effects = _lora_report(records)
    manifest = {
        "track": "exact_factorial_retrieval_attribution",
        "held_out_examples": len(test_examples),
        "seeds": seed_values,
        "graph_factors": list(GRAPH_FACTORS),
        "hypergraph_factors": list(HYPERGRAPH_FACTORS),
        "raw_dense_source_always_present": True,
        "fusion": "reciprocal_rank_fusion",
        "lora_effect": "conditional Base BGE reranker to LoRA reranker increment",
        "candidate_budget": candidate_budget,
        "top_k": top_k,
        "generator_disabled": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "records.json": records,
        "summary.json": summary,
        "shapley.json": shapley,
        "interactions.json": interactions,
        "lora_effects.json": lora_effects,
        "manifest.json": manifest,
    }
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(
        _render(summary, shapley, interactions, lora_effects, manifest), encoding="utf-8"
    )
    typer.echo(f"wrote {len(records)} records")


def _mapping(value):
    return {int(seed): path for seed, path in (item.split("=", 1) for item in value.split(","))}


def _load_checkpoint(root, mode, profile, seed):
    return torch.load(
        root / f"{mode}_{profile}" / f"seed_{seed}" / "stage_a.pt",
        map_location="cpu",
        weights_only=True,
    )


def _graph_sources(pipeline, question, budget):
    query_np = encode_queries(pipeline.text_encoder, [question])[0]
    query_tensor = torch.tensor(query_np, dtype=torch.float32)
    with torch.no_grad():
        query_parts, gate = pipeline.model.encode_query_parts(
            query_tensor, pipeline.raw_features, pipeline.node_bands, top_m=64, temperature=0.05
        )
        query_vector = torch.cat(
            [
                gate[index] * query_parts[name]
                for index, name in enumerate(("raw", "low", "mid", "high"))
            ]
        )
    single = pipeline.graph_index.search(query_vector.numpy(), budget, "spectral-single")
    band_hits = {
        name: pipeline.band_indices[name].search(query_parts[name].numpy(), budget, name)
        for name in ("raw", "low", "mid", "high")
    }
    multi = _weighted_band_fusion(band_hits, gate, budget)
    return {
        "raw": pipeline.raw_index.search(query_np, budget, "raw"),
        "spectral": reciprocal_rank_fusion([single, multi])[:budget],
        "bm25": pipeline.bm25.search(question, budget),
    }


def _hypergraph_sources(pipeline, question, budget):
    query_np = encode_queries(pipeline.text_encoder, [question])[0]
    query_tensor = torch.tensor(query_np, dtype=torch.float32)
    relation_weights, query_node_bands = pipeline._relation_conditioned_bands(query_tensor)
    with torch.no_grad():
        query_vector, _ = pipeline.model.encode_query(
            query_tensor, pipeline.raw_features, query_node_bands, top_m=64, temperature=0.05
        )
    relation_full = torch.cat(
        [
            pipeline.node_bands["raw"],
            query_node_bands["low"],
            query_node_bands["mid"],
            query_node_bands["high"],
        ],
        dim=-1,
    ).numpy()
    relation_hits = ExactVectorIndex(pipeline.object_ids, relation_full).search(
        query_vector.numpy(), budget, "spectral-relation"
    )
    spectral_hits = pipeline.qmshe_index.search(query_vector.numpy(), budget, "spectral-full")
    del relation_weights
    return {
        "raw": pipeline.raw_index.search(query_np, budget, "raw"),
        "spectral": reciprocal_rank_fusion([spectral_hits, relation_hits])[:budget],
        "bm25": pipeline.bm25.search(question, budget),
    }


def _evaluate_factorial(
    example_id,
    question,
    system,
    seed,
    pipeline,
    sources,
    factors,
    built,
    base_reranker,
    lora_reranker,
    candidate_budget,
    top_k,
):
    rows = []
    non_reranker_factors = tuple(item for item in factors if item != "base_reranker")
    for enabled in _subsets(factors):
        level = "base" if "base_reranker" in enabled else "none"
        rows.append(
            _evaluate_configuration(
                example_id,
                question,
                system,
                seed,
                pipeline,
                sources,
                factors,
                enabled,
                level,
                built,
                base_reranker,
                lora_reranker,
                candidate_budget,
                top_k,
            )
        )
    for enabled in _subsets(non_reranker_factors):
        rows.append(
            _evaluate_configuration(
                example_id,
                question,
                system,
                seed,
                pipeline,
                sources,
                factors,
                {*enabled, "base_reranker"},
                "lora",
                built,
                base_reranker,
                lora_reranker,
                candidate_budget,
                top_k,
            )
        )
    return rows


def _subsets(items):
    items = tuple(items)
    return [
        set(subset)
        for size in range(len(items) + 1)
        for subset in itertools.combinations(items, size)
    ]


def _evaluate_configuration(
    example_id,
    question,
    system,
    seed,
    pipeline,
    sources,
    factors,
    enabled,
    reranker_level,
    built,
    base_reranker,
    lora_reranker,
    candidate_budget,
    top_k,
):
    started = time.perf_counter()
    active = [sources["raw"]]
    if "spectral" in enabled:
        active.append(sources["spectral"])
    if "bm25" in enabled:
        active.append(sources["bm25"])
    node_hits = reciprocal_rank_fusion(active)[:candidate_budget]
    if "graph_rerank" in enabled:
        graph = pipeline.evidence_graph if system.startswith("hypergraph") else pipeline.artifacts.graph
        node_hits = graph_rerank(node_hits, graph)
    node_ids = [hit.object_id for hit in node_hits]
    if system.startswith("hypergraph"):
        fact_text = {
            fact.hyperedge_id: pipeline.text_by_id[fact.hyperedge_id]
            for fact in pipeline.corpus.evidence_hyperedges
        }
        fact_ids = [item for item in node_ids if item in fact_text]
    else:
        fact_text = pipeline.fact_text_by_id
        fact_ids = _graph_facts(
            pipeline, node_ids, expand="entity_expansion" in enabled
        )
    fact_ids = list(dict.fromkeys(fact_ids))[:candidate_budget]
    if reranker_level != "none" and fact_ids:
        reranker = base_reranker if reranker_level == "base" else lora_reranker
        order = reranker.rank(question, [fact_text[item] for item in fact_ids])
        fact_ids = [fact_ids[index] for index in order]
    ranking = fact_ids[:top_k]
    row = _metric_row(example_id, system, seed, ranking, built, time.perf_counter() - started)
    row.update({factor: factor in enabled for factor in factors})
    row["reranker_level"] = reranker_level
    row["mask"] = _mask(enabled, factors)
    return row


def _graph_facts(pipeline, candidate_ids, expand):
    if expand:
        return pipeline._facts_from_candidates(candidate_ids)
    ranked = []
    for candidate_id in candidate_ids:
        if candidate_id in pipeline.artifacts.fact_by_node:
            ranked.append(pipeline.artifacts.fact_by_node[candidate_id])
        if candidate_id in pipeline.fact_text_by_id:
            ranked.append(candidate_id)
    return list(dict.fromkeys(ranked))


def _metric_row(example_id, system, seed, facts, built, seconds):
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
    path_em, path_precision, path_recall, path_f1 = _set_scores(
        set(documents[:2]), gold_documents
    )
    row = {
        "example_id": example_id,
        "system": system,
        "seed": seed,
        "seconds": seconds,
        "mrr": reciprocal_rank(facts, gold),
        "ndcg_at_10": ndcg_at_k(facts, gold, 10),
        "passage_mrr": reciprocal_rank(documents, gold_documents),
        "path_em": path_em,
        "path_precision": path_precision,
        "path_recall": path_recall,
        "path_f1": path_f1,
    }
    for k in KS:
        row[f"recall_at_{k}"] = recall_at_k(facts, gold, k)
        row[f"hit_at_{k}"] = hit_at_k(facts, gold, k)
        row[f"complete_at_{k}"] = complete_at_k(facts, gold, k)
        row[f"passage_recall_at_{k}"] = recall_at_k(documents, gold_documents, k)
        row[f"passage_hit_at_{k}"] = hit_at_k(documents, gold_documents, k)
        row[f"passage_complete_at_{k}"] = complete_at_k(documents, gold_documents, k)
    return row


def _set_scores(predicted, gold):
    precision = len(predicted & gold) / len(predicted) if predicted else float(not gold)
    recall = len(predicted & gold) / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(predicted == gold), precision, recall, f1


def _mask(enabled, factors):
    return sum(1 << index for index, factor in enumerate(factors) if factor in enabled)


def _aggregate_conditions(records):
    output = {}
    grouped = defaultdict(list)
    for row in records:
        factors = HYPERGRAPH_FACTORS if row["system"].startswith("hypergraph") else GRAPH_FACTORS
        full_mask = (1 << len(factors)) - 1
        label = None
        if row["mask"] == 0 and row["reranker_level"] == "none":
            label = "raw_minimal"
        elif row["mask"] == full_mask and row["reranker_level"] == "base":
            label = "full_base"
        elif row["mask"] == full_mask and row["reranker_level"] == "lora":
            label = "full_lora"
        elif (
            row["mask"] == full_mask ^ (1 << factors.index("base_reranker"))
            and row["reranker_level"] == "none"
        ):
            label = "full_no_neural_reranker"
        if label:
            grouped[(row["system"], label)].append(row)
    for (system, label), rows in sorted(grouped.items()):
        key = f"{system}:{label}"
        output[key] = {"system": system, "condition": label, "rows": len(rows)}
        for metric in ATTRIBUTION_METRICS:
            values = _seed_means(rows, metric)
            output[key][f"{metric}_mean"] = mean(values)
            output[key][f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return output


def _seed_means(rows, metric):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row[metric])
    return [mean(values) for values in by_seed.values()]


def _shapley_report(records):
    by_observation = _base_value_tables(records)
    values = defaultdict(lambda: defaultdict(list))
    for (system, _, _), table in by_observation.items():
        factors = HYPERGRAPH_FACTORS if system.startswith("hypergraph") else GRAPH_FACTORS
        for metric in ATTRIBUTION_METRICS:
            score_table = {mask: row[metric] for mask, row in table.items()}
            contributions = _exact_shapley(score_table, factors)
            for factor, value in contributions.items():
                values[(system, factor)][metric].append(value)
    return _effect_summary(values)


def _exact_shapley(values, factors):
    count = len(factors)
    output = {}
    for index, factor in enumerate(factors):
        bit = 1 << index
        contribution = 0.0
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            contribution += weight * (values[mask | bit] - values[mask])
        output[factor] = contribution
    return output


def _interaction_report(records):
    by_observation = _base_value_tables(records)
    values = defaultdict(lambda: defaultdict(list))
    for (system, _, _), table in by_observation.items():
        factors = HYPERGRAPH_FACTORS if system.startswith("hypergraph") else GRAPH_FACTORS
        for left_index, left in enumerate(factors):
            for right_index in range(left_index + 1, len(factors)):
                right = factors[right_index]
                left_bit, right_bit = 1 << left_index, 1 << right_index
                eligible = [
                    mask
                    for mask in range(1 << len(factors))
                    if not mask & left_bit and not mask & right_bit
                ]
                for metric in ATTRIBUTION_METRICS:
                    differences = [
                        table[mask | left_bit | right_bit][metric]
                        - table[mask | left_bit][metric]
                        - table[mask | right_bit][metric]
                        + table[mask][metric]
                        for mask in eligible
                    ]
                    values[(system, f"{left}×{right}")][metric].append(mean(differences))
    return _effect_summary(values)


def _base_value_tables(records):
    grouped = defaultdict(dict)
    for row in records:
        if row["reranker_level"] in {"none", "base"}:
            grouped[(row["system"], row["seed"], row["example_id"])][row["mask"]] = row
    return grouped


def _lora_report(records):
    grouped = defaultdict(dict)
    for row in records:
        grouped[(row["system"], row["seed"], row["example_id"], row["mask"])][
            row["reranker_level"]
        ] = row
    values = defaultdict(lambda: defaultdict(list))
    for (system, _, _, _), levels in grouped.items():
        if "base" not in levels or "lora" not in levels:
            continue
        for metric in ATTRIBUTION_METRICS:
            values[(system, "reranker_lora")][metric].append(
                levels["lora"][metric] - levels["base"][metric]
            )
    return _effect_summary(values)


def _effect_summary(values):
    rng = np.random.default_rng(20260722)
    output = {}
    for (system, factor), metrics in sorted(values.items()):
        item = {"system": system, "factor": factor}
        for metric, observations in metrics.items():
            array = np.asarray(observations, dtype=np.float64)
            observed = float(array.mean())
            samples = 10000
            bootstrap_indices = rng.integers(0, len(array), size=(samples, len(array)))
            bootstrap = np.sort(array[bootstrap_indices].mean(axis=1))
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(array)))
            permutations = (signs * array).mean(axis=1)
            item[metric] = {
                "mean": observed,
                "ci95_low": float(bootstrap[int(samples * 0.025)]),
                "ci95_high": float(bootstrap[int(samples * 0.975)]),
                "p_two_sided": (
                    1 + int(np.count_nonzero(np.abs(permutations) >= abs(observed)))
                )
                / (samples + 1),
                "observations": len(observations),
            }
        output[f"{system}:{factor}"] = item
    return output


def _render(summary, shapley, interactions, lora_effects, manifest):
    lines = [
        "# Exact factorial attribution",
        "",
        "Raw dense retrieval is always present. Binary factors are attributed with exact Shapley values; LoRA is the contextual increment over the base reranker.",
        "",
        "## Anchor configurations",
        "",
        "| System | Condition | Fact R@20 | MRR | Passage R@5 | Path F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in summary.values():
        lines.append(
            f"| {item['system']} | {item['condition']} | "
            f"{item['recall_at_20_mean']:.4f} | {item['mrr_mean']:.4f} | "
            f"{item['passage_recall_at_5_mean']:.4f} | {item['path_f1_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Exact Shapley attribution",
            "",
            "| System | Factor | Fact R@20 | 95% CI | p | MRR | Path F1 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in shapley.values():
        recall = item["recall_at_20"]
        lines.append(
            f"| {item['system']} | {item['factor']} | {recall['mean']:+.4f} | "
            f"[{recall['ci95_low']:+.4f}, {recall['ci95_high']:+.4f}] | "
            f"{recall['p_two_sided']:.4f} | {item['mrr']['mean']:+.4f} | "
            f"{item['path_f1']['mean']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## LoRA increment over base reranker",
            "",
            "| System | Fact R@20 | 95% CI | p | MRR | Path F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in lora_effects.values():
        recall = item["recall_at_20"]
        lines.append(
            f"| {item['system']} | {recall['mean']:+.4f} | "
            f"[{recall['ci95_low']:+.4f}, {recall['ci95_high']:+.4f}] | "
            f"{recall['p_two_sided']:.4f} | {item['mrr']['mean']:+.4f} | "
            f"{item['path_f1']['mean']:+.4f} |"
        )
    ranked_interactions = sorted(
        interactions.values(),
        key=lambda item: abs(item["recall_at_20"]["mean"]),
        reverse=True,
    )
    lines.extend(
        [
            "",
            "## Largest pairwise interactions on Fact Recall@20",
            "",
            "| System | Pair | Mean second difference |",
            "|---|---|---:|",
        ]
    )
    for item in ranked_interactions[:15]:
        lines.append(
            f"| {item['system']} | {item['factor']} | "
            f"{item['recall_at_20']['mean']:+.4f} |"
        )
    lines.extend(["", "## Manifest", "", "```json", json.dumps(manifest, indent=2), "```", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    typer.run(main)

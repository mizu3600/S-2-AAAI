from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.evaluation.retrieval_metrics import (
    complete_at_k, hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank,
)
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline


@dataclass(frozen=True)
class StageABRecord:
    example_id: str
    mode: str
    profile: str
    seed: int
    stage_a: bool
    stage_b: bool
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    recall_at_30: float
    recall_at_40: float
    hit_at_10: float
    complete_at_20: float
    complete_at_40: float
    mrr: float
    ndcg_at_10: float


def run_stage_ab_condition(
    examples, text_encoder, reranker, mode: str, profile: str, seed: int,
    stage_a: bool, stage_b: bool, checkpoint: dict | None, output_dir: str | Path,
) -> list[StageABRecord]:
    records = []
    for example in examples:
        built = build_example_corpus(example)
        if mode == "hypergraph":
            pipeline = QMSHEPipeline(
                built.corpus, text_encoder=text_encoder, reranker=reranker,
                seed=seed, enable_remote_reranker=False,
            )
            if stage_a:
                pipeline.load_stage_a_checkpoint(checkpoint)
        else:
            pipeline = QMSGEGraphPipeline(
                built.corpus, text_encoder=text_encoder, reranker=reranker,
                profile=GraphProfile(profile), seed=seed, enable_remote_reranker=False,
            )
            if stage_a:
                pipeline.load_stage_a_checkpoint(checkpoint)
        pipeline.generator.client = None
        result = pipeline.query(
            example.question, top_k=40, return_debug=False, candidate_count=60
        )
        facts = (
            result.retrieved_hyperedges if mode == "hypergraph" else result.retrieved_facts
        )
        gold = built.gold_fact_ids
        records.append(StageABRecord(
            example_id=example.example_id, mode=mode, profile=profile, seed=seed,
            stage_a=stage_a, stage_b=stage_b,
            recall_at_5=recall_at_k(facts, gold, 5),
            recall_at_10=recall_at_k(facts, gold, 10),
            recall_at_20=recall_at_k(facts, gold, 20),
            recall_at_30=recall_at_k(facts, gold, 30),
            recall_at_40=recall_at_k(facts, gold, 40),
            hit_at_10=hit_at_k(facts, gold, 10),
            complete_at_20=complete_at_k(facts, gold, 20),
            complete_at_40=complete_at_k(facts, gold, 40),
            mrr=reciprocal_rank(facts, gold), ndcg_at_10=ndcg_at_k(facts, gold, 10),
        ))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.json").write_text(
        json.dumps([asdict(item) for item in records], indent=2), encoding="utf-8"
    )
    return records


def aggregate_stage_ab(records: list[StageABRecord]) -> dict:
    groups = {}
    for record in records:
        key = (record.mode, record.profile, record.stage_a, record.stage_b)
        groups.setdefault(key, {}).setdefault(record.seed, []).append(record)
    metrics = (
        "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_30", "recall_at_40",
        "hit_at_10", "complete_at_20", "complete_at_40", "mrr", "ndcg_at_10",
    )
    output = {}
    for key, seed_groups in groups.items():
        name = f"{key[0]}:{key[1]}:A{int(key[2])}B{int(key[3])}"
        output[name] = {"seed_count": len(seed_groups)}
        for metric in metrics:
            values = [mean(getattr(item, metric) for item in rows) for rows in seed_groups.values()]
            output[name][f"{metric}_mean"] = mean(values)
            output[name][f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return output


def stage_ab_effects(summary: dict) -> dict:
    systems = {(name.split(":")[0], name.split(":")[1]) for name in summary}
    output = {}
    for mode, profile in systems:
        cells = {
            cell: summary[f"{mode}:{profile}:{cell}"] for cell in ("A0B0", "A1B0", "A0B1", "A1B1")
        }
        output[f"{mode}:{profile}"] = {}
        for metric in ("recall_at_10", "recall_at_20", "complete_at_20", "mrr"):
            base = cells["A0B0"][f"{metric}_mean"]
            a = cells["A1B0"][f"{metric}_mean"]
            b = cells["A0B1"][f"{metric}_mean"]
            joint = cells["A1B1"][f"{metric}_mean"]
            output[f"{mode}:{profile}"][metric] = {
                "stage_a_only": a - base, "stage_b_only": b - base,
                "joint": joint - base, "interaction": joint - a - b + base,
            }
    return output


def render_stage_ab(summary, effects, examples, seeds):
    rows = [
        "# Stage A / Stage B factorial ablation", "",
        f"Held-out examples: {examples}; seeds: {', '.join(map(str, seeds))}", "",
        "The tuned reranker is fixed in every cell. Mean±SD is across seeds.", "",
        "| System | Cell | R@10 | R@20 | R@40 | Accuracy/Hit@10 | Complete@20 | MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = ("A0B0", "A1B0", "A0B1", "A1B1")
    for name in sorted(summary, key=lambda item: (item.rsplit(":", 1)[0], order.index(item.rsplit(":", 1)[1]))):
        item = summary[name]
        system, cell = name.rsplit(":", 1)
        rows.append(
            f"| {system} | {cell} | {item['recall_at_10_mean']:.4f}±{item['recall_at_10_std']:.4f} | "
            f"{item['recall_at_20_mean']:.4f}±{item['recall_at_20_std']:.4f} | "
            f"{item['recall_at_40_mean']:.4f}±{item['recall_at_40_std']:.4f} | "
            f"{item['hit_at_10_mean']:.4f}±{item['hit_at_10_std']:.4f} | "
            f"{item['complete_at_20_mean']:.4f}±{item['complete_at_20_std']:.4f} | "
            f"{item['mrr_mean']:.4f}±{item['mrr_std']:.4f} |"
        )
    rows.extend([
        "", "## Effects on Recall@20", "",
        "| System | Stage A only | Stage B only | Joint | Interaction |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, metrics in effects.items():
        item = metrics["recall_at_20"]
        rows.append(
            f"| {name} | {item['stage_a_only']:+.4f} | {item['stage_b_only']:+.4f} | "
            f"{item['joint']:+.4f} | {item['interaction']:+.4f} |"
        )
    return "\n".join(rows) + "\n"

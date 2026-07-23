from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from statistics import stdev
from time import perf_counter

from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.benchmarks.schemas import BenchmarkSuite
from qmshe.evaluation.experiment import LocalBenchmarkEncoder
from qmshe.evaluation.retrieval_metrics import (
    complete_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline


@dataclass(frozen=True)
class DualModeRecord:
    example_id: str
    mode: str
    profile: str
    hop_count: int
    query_type: str
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    recall_at_30: float
    recall_at_40: float
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    precision_at_30: float
    precision_at_40: float
    hit_at_5: float
    hit_at_10: float
    hit_at_20: float
    hit_at_30: float
    hit_at_40: float
    complete_at_5: float
    complete_at_10: float
    complete_at_20: float
    complete_at_30: float
    complete_at_40: float
    mrr: float
    ndcg_at_10: float
    bridge_recall_at_20: float
    latency_ms: float
    seed: int = 42
    embedding_variant: str = "base"
    reranker_variant: str = "none"


class DualModeExperimentRunner:
    """Fair comparison: one corpus, encoder, top-k and generator policy across all modes."""

    def __init__(
        self, encoder_dimension: int = 128, encoder=None, reranker=None, seed: int = 42,
        embedding_variant: str = "base", reranker_variant: str = "none",
    ):
        self.encoder = encoder or LocalBenchmarkEncoder(encoder_dimension)
        self.reranker = reranker
        self.seed = seed
        self.embedding_variant = embedding_variant
        self.reranker_variant = reranker_variant

    def run(self, suite: BenchmarkSuite, output_dir: str | Path) -> list[DualModeRecord]:
        records: list[DualModeRecord] = []
        for example in suite.examples:
            built = build_example_corpus(example)
            hypergraph_pipeline = QMSHEPipeline(
                built.corpus, text_encoder=self.encoder, reranker=self.reranker,
                seed=self.seed, enable_remote_reranker=False,
            )
            pipelines = [
                ("vanilla_rag", "dense_bge_compatible", hypergraph_pipeline),
                ("hypergraph", "evidence_hypergraph", hypergraph_pipeline),
                ("graph", GraphProfile.ENTITY_RELATION.value, QMSGEGraphPipeline(
                    built.corpus, text_encoder=self.encoder, profile=GraphProfile.ENTITY_RELATION,
                    reranker=self.reranker, seed=self.seed, enable_remote_reranker=False,
                )),
                ("graph", GraphProfile.REIFIED_FACT.value, QMSGEGraphPipeline(
                    built.corpus, text_encoder=self.encoder, profile=GraphProfile.REIFIED_FACT,
                    reranker=self.reranker, seed=self.seed, enable_remote_reranker=False,
                )),
            ]
            for mode, profile, pipeline in pipelines:
                pipeline.generator.client = None
                started = perf_counter()
                if mode == "vanilla_rag":
                    query_method = getattr(self.encoder, "encode_queries", self.encoder.encode)
                    query_vector = query_method([example.question])[0]
                    hits = pipeline.raw_index.search(query_vector, 60, "vanilla-dense")
                    facts = [hit.object_id for hit in hits if hit.object_id.startswith("fact_")]
                    if self.reranker is not None:
                        order = self.reranker.rank(
                            example.question, [pipeline.text_by_id[item] for item in facts]
                        )
                        facts = [facts[index] for index in order]
                    facts = facts[:40]
                    entities = [hit.object_id for hit in hits if hit.object_id.startswith("ent_")][:20]
                else:
                    result = pipeline.query(
                        example.question, top_k=40, return_debug=False, candidate_count=60
                    )
                    if mode == "hypergraph":
                        facts = result.retrieved_hyperedges
                        entities = result.retrieved_entities
                    else:
                        facts = result.retrieved_facts
                        entities = [item for item in result.retrieved_nodes if item.startswith("ent_")]
                latency = (perf_counter() - started) * 1000
                records.append(DualModeRecord(
                    example_id=example.example_id, mode=mode, profile=profile,
                    hop_count=example.hop_count, query_type=example.query_type,
                    recall_at_5=recall_at_k(facts, built.gold_fact_ids, 5),
                    recall_at_10=recall_at_k(facts, built.gold_fact_ids, 10),
                    recall_at_20=recall_at_k(facts, built.gold_fact_ids, 20),
                    recall_at_30=recall_at_k(facts, built.gold_fact_ids, 30),
                    recall_at_40=recall_at_k(facts, built.gold_fact_ids, 40),
                    precision_at_5=precision_at_k(facts, built.gold_fact_ids, 5),
                    precision_at_10=precision_at_k(facts, built.gold_fact_ids, 10),
                    precision_at_20=precision_at_k(facts, built.gold_fact_ids, 20),
                    precision_at_30=precision_at_k(facts, built.gold_fact_ids, 30),
                    precision_at_40=precision_at_k(facts, built.gold_fact_ids, 40),
                    hit_at_5=hit_at_k(facts, built.gold_fact_ids, 5),
                    hit_at_10=hit_at_k(facts, built.gold_fact_ids, 10),
                    hit_at_20=hit_at_k(facts, built.gold_fact_ids, 20),
                    hit_at_30=hit_at_k(facts, built.gold_fact_ids, 30),
                    hit_at_40=hit_at_k(facts, built.gold_fact_ids, 40),
                    complete_at_5=complete_at_k(facts, built.gold_fact_ids, 5),
                    complete_at_10=complete_at_k(facts, built.gold_fact_ids, 10),
                    complete_at_20=complete_at_k(facts, built.gold_fact_ids, 20),
                    complete_at_30=complete_at_k(facts, built.gold_fact_ids, 30),
                    complete_at_40=complete_at_k(facts, built.gold_fact_ids, 40),
                    mrr=reciprocal_rank(facts, built.gold_fact_ids),
                    ndcg_at_10=ndcg_at_k(facts, built.gold_fact_ids, 10),
                    bridge_recall_at_20=recall_at_k(entities, built.bridge_entity_ids, 20),
                    latency_ms=latency,
                    seed=self.seed, embedding_variant=self.embedding_variant,
                    reranker_variant=self.reranker_variant,
                ))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "records.json").write_text(
            json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8"
        )
        summary = _summarize(records)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "report.md").write_text(_render_report(suite, summary), encoding="utf-8")
        return records


def _summarize(records: list[DualModeRecord]) -> dict:
    groups: dict[str, list[DualModeRecord]] = {}
    for record in records:
        groups.setdefault(f"{record.mode}:{record.profile}", []).append(record)
    metrics = (
        "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_30", "recall_at_40",
        "precision_at_5", "precision_at_10", "precision_at_20", "precision_at_30",
        "precision_at_40", "hit_at_5", "hit_at_10", "hit_at_20", "hit_at_30",
        "hit_at_40", "complete_at_5", "complete_at_10", "complete_at_20",
        "complete_at_30", "complete_at_40", "mrr", "ndcg_at_10",
        "bridge_recall_at_20", "latency_ms",
    )
    output = {}
    for name, items in groups.items():
        output[name] = {}
        for metric in metrics:
            values = [getattr(record, metric) for record in items]
            output[name][metric] = mean(values)
            output[name][f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return output


def _render_report(suite: BenchmarkSuite, summary: dict) -> str:
    rows = [
        f"# {suite.name} Graph/Hypergraph fair comparison", "",
        f"Examples: {len(suite.examples)}", "",
        "| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary.items():
        rows.append(
            f"| {name} | {item['recall_at_5']:.4f} | {item['recall_at_10']:.4f} | "
            f"{item['recall_at_20']:.4f} | {item['recall_at_30']:.4f} | "
            f"{item['recall_at_40']:.4f} | {item['hit_at_10']:.4f} | "
            f"{item['complete_at_20']:.4f} | {item['mrr']:.4f} |"
        )
    return "\n".join(rows) + "\n"

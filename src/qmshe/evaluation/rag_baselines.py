from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.evaluation.retrieval_metrics import (
    complete_at_k,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from qmshe.pipeline import verbalize_fact
from qmshe.retrieval.ann_retriever import ExactVectorIndex
from qmshe.retrieval.seed_retriever import BM25Retriever


@dataclass(frozen=True)
class RAGBaselineRecord:
    example_id: str
    method: str
    seed: int
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


@dataclass(frozen=True)
class PreparedRAGExample:
    example_id: str
    question: str
    fact_ids: list[str]
    fact_texts: list[str]
    document_vectors: np.ndarray
    bm25_ranking: list[str]
    gold_fact_ids: set[str]


def prepare_rag_examples(examples, encoder, candidate_count: int = 60):
    prepared = []
    for example in examples:
        built = build_example_corpus(example)
        names = {
            entity.entity_id: entity.canonical_name for entity in built.corpus.entities
        }
        fact_ids = [fact.hyperedge_id for fact in built.corpus.evidence_hyperedges]
        fact_texts = [
            verbalize_fact(fact, names) for fact in built.corpus.evidence_hyperedges
        ]
        vectors = encoder.encode_documents(fact_texts)
        bm25 = BM25Retriever(fact_ids, fact_texts)
        prepared.append(PreparedRAGExample(
            example_id=example.example_id,
            question=example.question,
            fact_ids=fact_ids,
            fact_texts=fact_texts,
            document_vectors=vectors,
            bm25_ranking=[
                hit.object_id for hit in bm25.search(example.question, candidate_count)
            ],
            gold_fact_ids=built.gold_fact_ids,
        ))
    return prepared


def dense_ranking(item: PreparedRAGExample, query_vector, candidate_count: int = 60):
    return [
        hit.object_id
        for hit in ExactVectorIndex(item.fact_ids, item.document_vectors).search(
            np.asarray(query_vector), candidate_count, "dense"
        )
    ]


def reciprocal_rank_fusion_ids(*rankings: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, object_id in enumerate(ranking, 1):
            scores[object_id] = scores.get(object_id, 0.0) + 1.0 / (k + rank)
    return [item[0] for item in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def rerank(item: PreparedRAGExample, ranking: list[str], reranker) -> list[str]:
    text_by_id = dict(zip(item.fact_ids, item.fact_texts, strict=True))
    order = reranker.rank(item.question, [text_by_id[object_id] for object_id in ranking])
    return [ranking[index] for index in order]


def make_record(item: PreparedRAGExample, ranking: list[str], method: str, seed: int):
    gold = item.gold_fact_ids
    return RAGBaselineRecord(
        example_id=item.example_id,
        method=method,
        seed=seed,
        recall_at_5=recall_at_k(ranking, gold, 5),
        recall_at_10=recall_at_k(ranking, gold, 10),
        recall_at_20=recall_at_k(ranking, gold, 20),
        recall_at_30=recall_at_k(ranking, gold, 30),
        recall_at_40=recall_at_k(ranking, gold, 40),
        hit_at_10=hit_at_k(ranking, gold, 10),
        complete_at_20=complete_at_k(ranking, gold, 20),
        complete_at_40=complete_at_k(ranking, gold, 40),
        mrr=reciprocal_rank(ranking, gold),
        ndcg_at_10=ndcg_at_k(ranking, gold, 10),
    )


def aggregate_rag_baselines(records: list[RAGBaselineRecord]) -> dict:
    metrics = (
        "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_30",
        "recall_at_40", "hit_at_10", "complete_at_20", "complete_at_40",
        "mrr", "ndcg_at_10",
    )
    grouped = {}
    for record in records:
        grouped.setdefault(record.method, {}).setdefault(record.seed, []).append(record)
    output = {}
    for method, seed_groups in grouped.items():
        output[method] = {"seed_count": len(seed_groups)}
        for metric in metrics:
            values = [
                mean(getattr(item, metric) for item in items)
                for items in seed_groups.values()
            ]
            output[method][f"{metric}_mean"] = mean(values)
            output[method][f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return output


def write_rag_baseline_report(records, output_dir: str | Path, example_count: int):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate_rag_baselines(records)
    (output / "records.json").write_text(
        json.dumps([asdict(item) for item in records], indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    rows = [
        "# Fact-only non-graph RAG baselines", "",
        f"Held-out examples: {example_count}", "",
        "All methods use the same fact texts and a candidate budget of 60.", "",
        "| Method | R@10 | R@20 | R@40 | Accuracy/Hit@10 | Complete@20 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, item in summary.items():
        rows.append(
            f"| {method} | {item['recall_at_10_mean']:.4f}±{item['recall_at_10_std']:.4f} | "
            f"{item['recall_at_20_mean']:.4f}±{item['recall_at_20_std']:.4f} | "
            f"{item['recall_at_40_mean']:.4f}±{item['recall_at_40_std']:.4f} | "
            f"{item['hit_at_10_mean']:.4f}±{item['hit_at_10_std']:.4f} | "
            f"{item['complete_at_20_mean']:.4f}±{item['complete_at_20_std']:.4f} | "
            f"{item['mrr_mean']:.4f}±{item['mrr_std']:.4f} |"
        )
    (output / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary

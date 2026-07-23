import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.benchmarks.schemas import BenchmarkSuite
from qmshe.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from qmshe.pipeline import QMSHEPipeline
from qmshe.providers import DeterministicEmbedder


class LocalBenchmarkEncoder:
    def __init__(self, dimension: int = 128):
        self.encoder = DeterministicEmbedder(dimension)

    def encode(self, texts):
        return self.encoder.embed(texts)


@dataclass(frozen=True)
class ExperimentRecord:
    dataset: str
    example_id: str
    method: str
    hop_count: int
    query_type: str
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float
    bridge_recall_at_20: float
    latency_ms: float


class BenchmarkExperimentRunner:
    def __init__(
        self, methods: list[str] | None = None, encoder_dimension: int = 128,
        track_mlflow: bool = False,
    ):
        self.methods = methods or [
            "bm25", "dense", "bm25+dense", "node2vec", "laplacian_eigenmaps",
            "semantic+lap_pe", "semantic+ppr", "gcn", "graphsage", "hypergraph_conv", "qmshe",
        ]
        self.encoder = LocalBenchmarkEncoder(encoder_dimension)
        self.track_mlflow = track_mlflow

    def run(self, suite: BenchmarkSuite, output_dir: str | Path) -> list[ExperimentRecord]:
        records: list[ExperimentRecord] = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for example in suite.examples:
            built = build_example_corpus(example)
            pipeline = QMSHEPipeline(built.corpus, text_encoder=self.encoder)
            pipeline.generator.client = None
            query_vector = self.encoder.encode([example.question])[0]
            for method in self.methods:
                started = perf_counter()
                ranked, entities = self._retrieve(method, pipeline, example.question, query_vector)
                latency_ms = (perf_counter() - started) * 1000
                records.append(ExperimentRecord(
                    dataset=suite.name, example_id=example.example_id, method=method,
                    hop_count=example.hop_count, query_type=example.query_type,
                    recall_at_10=recall_at_k(ranked, built.gold_fact_ids, 10),
                    recall_at_20=recall_at_k(ranked, built.gold_fact_ids, 20),
                    mrr=reciprocal_rank(ranked, built.gold_fact_ids),
                    ndcg_at_10=ndcg_at_k(ranked, built.gold_fact_ids, 10),
                    bridge_recall_at_20=recall_at_k(entities, built.bridge_entity_ids, 20),
                    latency_ms=latency_ms,
                ))
        (output_dir / "records.json").write_text(
            json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8"
        )
        summary = summarize_records(records)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        grouped = {
            "hop_count": summarize_records_by(records, "hop_count"),
            "query_type": summarize_records_by(records, "query_type"),
        }
        (output_dir / "grouped_summary.json").write_text(
            json.dumps(grouped, indent=2), encoding="utf-8"
        )
        (output_dir / "report.md").write_text(
            render_report(suite, summary, grouped), encoding="utf-8"
        )
        if self.track_mlflow:
            _log_mlflow(suite, summary, output_dir)
        return records

    def _retrieve(self, method, pipeline, question, query_vector):
        if method == "bm25":
            hits = pipeline.bm25.search(question, 20)
            return [hit.object_id for hit in hits], []
        if method == "dense":
            hits = pipeline.raw_index.search(query_vector, 60, "dense")
            return ([hit.object_id for hit in hits if hit.object_id.startswith("fact_")][:20],
                    [hit.object_id for hit in hits if hit.object_id.startswith("ent_")][:20])
        if method == "bm25+dense":
            from qmshe.retrieval.seed_retriever import reciprocal_rank_fusion
            hits = reciprocal_rank_fusion([
                pipeline.bm25.search(question, 40), pipeline.raw_index.search(query_vector, 60, "dense")
            ])
            return ([hit.object_id for hit in hits if hit.object_id.startswith("fact_")][:20],
                    [hit.object_id for hit in hits if hit.object_id.startswith("ent_")][:20])
        if method == "node2vec":
            from qmshe.evaluation.baseline_models import node2vec_embedding
            node_ids, vectors = node2vec_embedding(
                pipeline.evidence_graph, dimensions=32, walk_length=12, walks_per_node=4
            )
            by_id = {node_id: vectors[index] for index, node_id in enumerate(node_ids)}
            ordered = np.stack([by_id[object_id] for object_id in pipeline.object_ids])
            return _seed_pool_rank(pipeline, query_vector, ordered)
        if method == "laplacian_eigenmaps":
            from qmshe.evaluation.baselines import laplacian_eigenmaps
            vectors = laplacian_eigenmaps(pipeline.laplacian_scipy, dimensions=16)
            return _seed_pool_rank(pipeline, query_vector, vectors)
        if method == "semantic+lap_pe":
            from qmshe.evaluation.baselines import laplacian_eigenmaps
            lap_pe = laplacian_eigenmaps(pipeline.laplacian_scipy, dimensions=16)
            vectors = np.concatenate([pipeline.raw_features.numpy(), lap_pe], axis=-1)
            raw_scores = pipeline.raw_index.search(query_vector, 16)
            seed_indices = [pipeline.object_ids.index(hit.object_id) for hit in raw_scores]
            structural_query = vectors[seed_indices, -lap_pe.shape[1] :].mean(axis=0)
            combined_query = np.concatenate([query_vector, structural_query])
            return _rank_vectors(pipeline.object_ids, vectors, combined_query)
        if method == "semantic+ppr":
            from qmshe.evaluation.baselines import ppr_scores
            seeds = [hit.object_id for hit in pipeline.raw_index.search(query_vector, 8)]
            scores = ppr_scores(pipeline.evidence_graph, seeds)
            ranked = sorted(scores, key=scores.get, reverse=True)
            return ([item for item in ranked if item.startswith("fact_")][:20],
                    [item for item in ranked if item.startswith("ent_")][:20])
        propagation = None
        if method in {"gcn", "graphsage"}:
            propagation = np.eye(pipeline.laplacian_scipy.shape[0]) - pipeline.laplacian_scipy.toarray()
            row_sum = propagation.sum(axis=1, keepdims=True)
            propagation = propagation / np.maximum(row_sum, 1e-12)
        if method == "gcn":
            vectors = propagation @ pipeline.raw_features.numpy()
            return _seed_pool_rank(pipeline, query_vector, vectors)
        if method == "graphsage":
            raw = pipeline.raw_features.numpy()
            vectors = np.concatenate([raw, propagation @ raw], axis=-1)
            return _seed_pool_rank(pipeline, query_vector, vectors)
        if method == "hypergraph_conv":
            vectors = pipeline.node_bands["low"].numpy()
            return _seed_pool_rank(pipeline, query_vector, vectors)
        if method == "qmshe":
            result = pipeline.query(question, top_k=20, return_debug=False)
            return result.retrieved_hyperedges, result.retrieved_entities
        raise ValueError(f"unknown method: {method}")


def _seed_pool_rank(pipeline, query_vector: np.ndarray, vectors: np.ndarray):
    seeds = pipeline.raw_index.search(query_vector, min(16, len(pipeline.object_ids)))
    seed_indices = [pipeline.object_ids.index(hit.object_id) for hit in seeds]
    weights = np.asarray([max(hit.score, 0) for hit in seeds], dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-12)
    structural_query = np.einsum("n,nd->d", weights, vectors[seed_indices])
    return _rank_vectors(pipeline.object_ids, vectors, structural_query)


def _rank_vectors(object_ids: list[str], vectors: np.ndarray, query: np.ndarray):
    norms = np.linalg.norm(vectors, axis=1)
    query_norm = max(float(np.linalg.norm(query)), 1e-12)
    scores = (vectors @ query) / np.maximum(norms * query_norm, 1e-12)
    ranked = [object_ids[index] for index in np.argsort(-scores)]
    return ([item for item in ranked if item.startswith("fact_")][:20],
            [item for item in ranked if item.startswith("ent_")][:20])


def summarize_records(records: list[ExperimentRecord]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.method].append(record)
    return {
        method: {
            metric: mean(getattr(record, metric) for record in items)
            for metric in ("recall_at_10", "recall_at_20", "mrr", "ndcg_at_10", "bridge_recall_at_20", "latency_ms")
        }
        for method, items in grouped.items()
    }


def summarize_records_by(records: list[ExperimentRecord], field: str) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, field))].append(record)
    return {
        group: summarize_records(items)
        for group, items in grouped.items()
    }


def render_report(suite: BenchmarkSuite, summary: dict, grouped: dict | None = None) -> str:
    lines = [f"# {suite.name} experiment", "", f"Examples: {len(suite.examples)}", "",
             "| Method | Recall@10 | Recall@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, metrics in summary.items():
        lines.append(
            f"| {method} | {metrics['recall_at_10']:.4f} | {metrics['recall_at_20']:.4f} | "
            f"{metrics['mrr']:.4f} | {metrics['ndcg_at_10']:.4f} | "
            f"{metrics['bridge_recall_at_20']:.4f} | {metrics['latency_ms']:.2f} |"
        )
    for field, groups in (grouped or {}).items():
        lines.extend(["", f"## By {field.replace('_', ' ')}", ""])
        for group, methods in groups.items():
            lines.extend([
                f"### {group}", "",
                "| Method | Recall@20 | MRR | Bridge@20 |",
                "|---|---:|---:|---:|",
            ])
            for method, metrics in methods.items():
                lines.append(
                    f"| {method} | {metrics['recall_at_20']:.4f} | "
                    f"{metrics['mrr']:.4f} | {metrics['bridge_recall_at_20']:.4f} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def _log_mlflow(suite: BenchmarkSuite, summary: dict, output_dir: Path) -> None:
    import mlflow

    from qmshe.settings import get_settings

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("qmshe-public-benchmarks-v2")
    with mlflow.start_run(run_name=f"{suite.name}-{suite.split}"):
        mlflow.log_params({"dataset": suite.name, "split": suite.split, "examples": len(suite.examples)})
        for method, metrics in summary.items():
            for metric, value in metrics.items():
                safe_method = method.replace("+", "_plus_")
                mlflow.log_metric(f"{safe_method}.{metric}", value)
        mlflow.log_artifacts(str(output_dir), artifact_path="report")

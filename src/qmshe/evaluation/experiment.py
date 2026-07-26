from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.benchmarks.schemas import BenchmarkSuite
from qmshe.evaluation.internal_baselines import (
    INTERNAL_BASELINES,
    prepare_internal_baseline,
    rank_internal_baseline,
)
from qmshe.evaluation.metrics import (
    answer_scores,
    extract_citation_ids,
    ranking_scores,
    set_scores,
)
from qmshe.evaluation.report import write_report
from qmshe.pipeline import QMSHERAGPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.retrieval.context_builder import build_context


HOTPOT_SHARED_GENERATION_PROTOCOL = "hotpotqa_shared_generation_v1"
RANKING_METRIC_NAMES = tuple(
    [
        f"{name}_at_{k}"
        for k in (1, 3, 5, 10, 20)
        for name in ("recall", "precision", "hit", "complete")
    ]
    + ["mrr"]
    + [f"ndcg_at_{k}" for k in (1, 3, 5, 10, 20)]
)
ANSWER_METRIC_NAMES = ("answer_em", "answer_precision", "answer_recall", "answer_f1")
SET_METRIC_NAMES = ("em", "precision", "recall", "f1")


@dataclass(frozen=True)
class EvaluationConfig:
    output_k: int = 20
    candidate_k: int = 40
    context_k: int = 12

    def __post_init__(self) -> None:
        if self.output_k <= 0 or self.context_k <= 0:
            raise ValueError("output_k and context_k must be positive")
        if self.candidate_k < self.output_k:
            raise ValueError("candidate_k must be at least output_k")


class LocalBenchmarkEncoder:
    def __init__(self, dimension: int = 128):
        self.encoder = DeterministicEmbedder(dimension)

    def encode(self, texts):
        return self.encoder.embed(texts)


class BenchmarkExperimentRunner:
    """Run controlled internal baselines over a common HotpotQA fact corpus."""

    def __init__(
        self,
        methods: tuple[str, ...] = (
            "bm25",
            "dense",
            "bm25_dense_rrf",
            "reified_fact_hybrid",
        ),
        encoder_dimension: int = 128,
        config: EvaluationConfig | None = None,
    ):
        self.methods = methods
        self.encoder = LocalBenchmarkEncoder(encoder_dimension)
        self.config = config or EvaluationConfig()

    def run(
        self,
        suite: BenchmarkSuite,
        output_dir: str | Path,
        seed: int = 42,
    ) -> list[dict]:
        records: list[dict] = []
        for example in suite.examples:
            built = build_example_corpus(example)
            shared_index_started = perf_counter()
            pipeline = QMSHERAGPipeline(
                built.corpus,
                text_encoder=self.encoder,
                seed=seed,
                enable_remote_reranker=False,
            )
            pipeline.generator.client = None
            shared_index_ms = (perf_counter() - shared_index_started) * 1000
            for method in self.methods:
                preparation_started = perf_counter()
                prepared = prepare_internal_baseline(pipeline, method, seed)
                method_preparation_ms = (perf_counter() - preparation_started) * 1000
                records.append(
                    self._run_method(
                        example,
                        built,
                        pipeline,
                        method,
                        seed,
                        prepared,
                        shared_index_ms,
                        method_preparation_ms,
                    )
                )
        metadata = {
            "dataset": suite.name,
            "split": suite.split,
            "source": suite.source,
            "source_sha256": _source_sha256(suite.source),
            "examples": len(suite.examples),
            "seed": seed,
            "systems": list(self.methods),
            "protocol": "controlled_hotpotqa_internal_v2",
            "output_k": self.config.output_k,
            "candidate_k": self.config.candidate_k,
            "context_k": self.config.context_k,
            "generation_protocol": HOTPOT_SHARED_GENERATION_PROTOCOL,
        }
        write_report(records, output_dir, metadata)
        return records

    def _run_method(
        self,
        example,
        built,
        pipeline,
        method: str,
        seed: int,
        prepared: dict,
        shared_index_ms: float,
        method_preparation_ms: float,
    ) -> dict:
        retrieval_started = perf_counter()
        if method == "reified_fact_hybrid":
            result = pipeline.query(
                example.question,
                top_k=self.config.output_k,
                candidate_count=self.config.candidate_k,
                return_debug=False,
                generate_answer=False,
            )
            ranked_facts = result.retrieved_facts
            origin = "qmshe_reified_fact_hybrid"
        elif method in INTERNAL_BASELINES:
            ranked_facts = rank_internal_baseline(
                pipeline,
                example.question,
                method,
                top_k=self.config.output_k,
                candidate_k=self.config.candidate_k,
                seed=seed,
                prepared=prepared,
            )
            origin = f"internal:{method}"
        else:
            raise ValueError(f"unknown benchmark method: {method}")
        retrieval_ms = (perf_counter() - retrieval_started) * 1000

        ranked_facts = list(dict.fromkeys(ranked_facts))[: self.config.output_k]
        passage_ranking = _passages_from_facts(ranked_facts, built.fact_to_passage)
        context_fact_ids = ranked_facts[: self.config.context_k]
        context, _ = build_context(
            pipeline.corpus, context_fact_ids, limit=self.config.context_k
        )
        generation_started = perf_counter()
        answer = pipeline.generator.generate(example.question, context)
        generation_ms = (perf_counter() - generation_started) * 1000

        valid_fact_ids = {fact.hyperedge_id for fact in built.corpus.evidence_hyperedges}
        generated_fact_citations = [
            item for item in extract_citation_ids(answer) if item in valid_fact_ids
        ]
        generated_passage_citations = _passages_from_facts(
            generated_fact_citations, built.fact_to_passage
        )
        gold_passages = _gold_passages(built)
        answer_metric = answer_scores(answer, example.answer)
        fact_citation_metric = set_scores(generated_fact_citations, built.gold_fact_ids)
        joint = _joint_scores(answer_metric, fact_citation_metric)

        record = {
            "dataset": example.dataset,
            "split": example.split,
            "example_id": example.example_id,
            "system": method,
            "seed": seed,
            "query_type": example.query_type,
            "hop_count": example.hop_count,
            "status": "success",
            "result_missing": False,
            "result_failed": False,
            "ranking_origin": origin,
            "fact_ranking_available": True,
            "passage_ranking_available": True,
            "answer_metric_available": True,
            "generated_fact_citation_available": True,
            "mapping_coverage": 1.0,
            "answer": answer,
            "retrieval_evidence_fact_ids": context_fact_ids,
            "generated_fact_citations": generated_fact_citations,
            "generated_passage_citations": generated_passage_citations,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "query_total_ms": retrieval_ms + generation_ms,
            "latency_ms": retrieval_ms + generation_ms,
            "shared_index_ms": shared_index_ms,
            "method_preparation_ms": method_preparation_ms,
            **joint,
        }
        record.update(_prefixed_ranking_scores("fact", ranked_facts, built.gold_fact_ids))
        record.update(_prefixed_ranking_scores("passage", passage_ranking, gold_passages))
        record.update(answer_metric)
        record.update(
            _prefixed_set_scores(
                "retrieval_evidence_fact", context_fact_ids, built.gold_fact_ids
            )
        )
        record.update(
            _prefixed_set_scores(
                "retrieval_evidence_passage",
                _passages_from_facts(context_fact_ids, built.fact_to_passage),
                gold_passages,
            )
        )
        record.update(
            {
                f"generated_fact_citation_{key}": value
                for key, value in fact_citation_metric.items()
            }
        )
        record.update(
            _prefixed_set_scores(
                "generated_passage_citation",
                generated_passage_citations,
                gold_passages,
            )
        )
        return record


def score_external_result(
    example,
    built,
    result,
    seed: int = 42,
    config: EvaluationConfig | None = None,
) -> dict:
    """Score native external traces without inventing sentence-level rankings."""
    config = config or EvaluationConfig()
    valid_fact_ids = {fact.hyperedge_id for fact in built.corpus.evidence_hyperedges}
    valid_passage_ids = set(built.fact_to_passage.values())
    gold_passages = _gold_passages(built)

    passage_ranking = [
        item
        for item in dict.fromkeys(result.document_ranking)
        if item in valid_passage_ids
    ][: config.output_k]
    raw_fact_ranking = list(dict.fromkeys(str(item) for item in result.fact_ranking))
    fact_ranking = [item for item in raw_fact_ranking if item in valid_fact_ids][
        : config.output_k
    ]
    fact_ranking_available = result.fact_ranking_declared and (
        len(fact_ranking) == len(raw_fact_ranking)
    )
    if result.status in {"missing", "failed", "unscorable"}:
        passage_ranking = []
        fact_ranking = []
        fact_ranking_available = False

    citation_ids = result.citations or extract_citation_ids(result.answer)
    generated_fact_citations: list[str] = []
    generated_passage_citations: list[str] = []
    fact_citation_available = False
    passage_citation_available = False
    if result.citation_level == "document":
        generated_passage_citations = [
            item for item in dict.fromkeys(citation_ids) if item in valid_passage_ids
        ]
        passage_citation_available = result.citation_mapping_complete
    elif result.citation_level == "fact":
        generated_fact_citations = [
            item for item in dict.fromkeys(citation_ids) if item in valid_fact_ids
        ]
        fact_citation_available = bool(citation_ids) and (
            len(generated_fact_citations) == len(set(citation_ids))
        )
        generated_passage_citations = _passages_from_facts(
            generated_fact_citations, built.fact_to_passage
        )
        passage_citation_available = fact_citation_available

    answer_metric_available = (
        result.generation_protocol == HOTPOT_SHARED_GENERATION_PROTOCOL
    )
    answer_metric = (
        answer_scores(result.answer, example.answer)
        if answer_metric_available
        else _unavailable_answer_scores()
    )
    fact_citation_metric = (
        set_scores(generated_fact_citations, built.gold_fact_ids)
        if fact_citation_available
        else None
    )
    joint = (
        _joint_scores(answer_metric, fact_citation_metric)
        if answer_metric_available and fact_citation_metric is not None
        else _unavailable_joint_scores()
    )
    external_total_ms = (
        result.total_seconds * 1000
        if result.total_seconds is not None
        else (
            result.retrieval_seconds * 1000
            if result.retrieval_seconds is not None
            else None
        )
    )

    record = {
        "dataset": example.dataset,
        "split": example.split,
        "example_id": example.example_id,
        "system": result.system,
        "seed": seed,
        "query_type": example.query_type,
        "hop_count": example.hop_count,
        "status": result.status,
        "result_missing": result.status == "missing",
        "result_failed": result.status in {"failed", "unscorable"},
        "error": result.error,
        "ranking_origin": result.ranking_origin,
        "fact_ranking_available": fact_ranking_available,
        "passage_ranking_available": result.document_ranking_declared,
        "answer_metric_available": answer_metric_available,
        "generated_fact_citation_available": fact_citation_available,
        "generated_passage_citation_available": passage_citation_available,
        "mapping_coverage": result.mapping_coverage,
        "unmapped_ranking_ids": result.unmapped_ranking_ids,
        "answer": result.answer,
        "generated_fact_citations": generated_fact_citations,
        "generated_passage_citations": generated_passage_citations,
        "native_index_ms": (
            result.indexing_seconds * 1000
            if result.indexing_seconds is not None
            else None
        ),
        "native_retrieval_ms": (
            result.retrieval_seconds * 1000
            if result.retrieval_seconds is not None
            else None
        ),
        "native_total_ms": (
            result.total_seconds * 1000 if result.total_seconds is not None else None
        ),
        "latency_ms": external_total_ms,
        "citation_level": result.citation_level,
        "generation_protocol": result.generation_protocol,
        **joint,
    }
    record.update(_prefixed_ranking_scores("passage", passage_ranking, gold_passages))
    record.update(
        _prefixed_ranking_scores(
            "fact",
            fact_ranking,
            built.gold_fact_ids,
            available=fact_ranking_available,
        )
    )
    record.update(answer_metric)
    record.update(
        _prefixed_set_scores(
            "retrieval_evidence_passage",
            passage_ranking[: config.context_k],
            gold_passages,
        )
    )
    record.update(
        _prefixed_set_scores(
            "retrieval_evidence_fact",
            fact_ranking[: config.context_k],
            built.gold_fact_ids,
            available=fact_ranking_available,
        )
    )
    record.update(
        _prefixed_set_scores(
            "generated_passage_citation",
            generated_passage_citations,
            gold_passages,
            available=passage_citation_available,
        )
    )
    record.update(
        _prefixed_set_scores(
            "generated_fact_citation",
            generated_fact_citations,
            built.gold_fact_ids,
            available=fact_citation_available,
        )
    )
    return record


def _gold_passages(built) -> set[str]:
    return {
        built.fact_to_passage[item]
        for item in built.gold_fact_ids
        if item in built.fact_to_passage
    }


def _source_sha256(source: str) -> str | None:
    path = Path(source)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _passages_from_facts(
    fact_ids: list[str], fact_to_passage: dict[str, str]
) -> list[str]:
    return list(
        dict.fromkeys(
            fact_to_passage[item] for item in fact_ids if item in fact_to_passage
        )
    )


def _prefixed_ranking_scores(
    prefix: str,
    ranking: list[str],
    gold: set[str],
    available: bool = True,
) -> dict[str, float | None]:
    if not available:
        return {f"{prefix}_{name}": None for name in RANKING_METRIC_NAMES}
    return {
        f"{prefix}_{key}": value for key, value in ranking_scores(ranking, gold).items()
    }


def _prefixed_set_scores(
    prefix: str,
    predicted: list[str],
    gold: set[str],
    available: bool = True,
) -> dict[str, float | None]:
    if not available:
        return {f"{prefix}_{name}": None for name in SET_METRIC_NAMES}
    return {
        f"{prefix}_{key}": value for key, value in set_scores(predicted, gold).items()
    }


def _unavailable_answer_scores() -> dict[str, None]:
    return {name: None for name in ANSWER_METRIC_NAMES}


def _joint_scores(
    answer_metric: dict[str, float | None],
    citation_metric: dict[str, float] | None,
) -> dict[str, float | None]:
    if citation_metric is None or any(
        answer_metric.get(name) is None
        for name in ("answer_em", "answer_precision", "answer_recall")
    ):
        return _unavailable_joint_scores()
    joint_precision = float(answer_metric["answer_precision"]) * citation_metric["precision"]
    joint_recall = float(answer_metric["answer_recall"]) * citation_metric["recall"]
    joint_f1 = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall
        else 0.0
    )
    return {
        "joint_precision": joint_precision,
        "joint_recall": joint_recall,
        "joint_f1": joint_f1,
        "joint_em": float(answer_metric["answer_em"]) * citation_metric["em"],
    }


def _unavailable_joint_scores() -> dict[str, None]:
    return {
        "joint_precision": None,
        "joint_recall": None,
        "joint_f1": None,
        "joint_em": None,
    }

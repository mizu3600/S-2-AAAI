from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from s2rag.benchmarks.corpus_builder import (
    build_example_corpus,
    build_sentence_fixture_corpus,
)
from s2rag.benchmarks.schemas import BenchmarkSuite
from s2rag.embedding.text_encoder import LocalBGEEncoder
from s2rag.extraction import OPEN_DOMAIN_EXTRACTION_PROTOCOL
from s2rag.extraction.entity_extractor import ENTITY_SYSTEM_PROMPT
from s2rag.extraction.fact_extractor import FACT_SYSTEM_PROMPT
from s2rag.evaluation.internal_baselines import (
    ALL_EXPERIMENT_METHODS,
    BENCHMARK_METHODS,
    prepare_internal_baseline,
    retrieve_internal_candidates,
)
from s2rag.evaluation.metrics import (
    answer_scores,
    extract_citation_ids,
    ranking_scores,
    set_scores,
)
from s2rag.evaluation.official_metrics import (
    official_metric_spec,
    score_official_metrics,
)
from s2rag.evaluation.report import write_report
from s2rag.generation.generator import (
    SHARED_DEEPSEEK_GENERATION_PROTOCOL,
    EvidenceGenerator,
)
from s2rag.pipeline import S2RAGPipeline
from s2rag.providers import DeepSeekClient, ProviderError
from s2rag.settings import get_settings
from s2rag.retrieval.candidates import (
    ScoredFact,
    aggregate_passages,
    rerank_scored_facts,
)
from s2rag.retrieval.context_builder import build_context
from s2rag.retrieval.local_reranker import LocalBGEReranker


HOTPOT_SHARED_GENERATION_PROTOCOL = SHARED_DEEPSEEK_GENERATION_PROTOCOL
UNIFIED_RETRIEVAL_PROTOCOL = "unified_shared_models_v1"
TRAINING_FREE_MODEL_ID = "analytic_graph_bands_v1"
RANKING_METRIC_NAMES = tuple(
    [
        f"{name}_at_{k}"
        for k in (1, 3, 5, 10, 20)
        for name in (
            "recall",
            "precision",
            "returned_precision",
            "hit",
            "complete",
        )
    ]
    + ["mrr_at_20", "mrr"]
    + [f"ndcg_at_{k}" for k in (1, 3, 5, 10, 20)]
)
ANSWER_METRIC_NAMES = ("answer_em", "answer_precision", "answer_recall", "answer_f1")
SET_METRIC_NAMES = ("em", "precision", "recall", "f1")


def _canonical_record_order(
    records: list[dict],
    suite: BenchmarkSuite,
    methods: tuple[str, ...],
) -> list[dict]:
    example_order = {
        example.example_id: index for index, example in enumerate(suite.examples)
    }
    method_order = {method: index for index, method in enumerate(methods)}
    fallback_example = len(example_order)
    fallback_method = len(method_order)
    return sorted(
        records,
        key=lambda record: (
            example_order.get(record.get("example_id"), fallback_example),
            method_order.get(record.get("system"), fallback_method),
        ),
    )


@dataclass(frozen=True)
class EvaluationConfig:
    output_k: int = 20
    first_stage_per_channel_k: int = 40
    rerank_input_k: int = 40
    context_k: int = 12

    def __post_init__(self) -> None:
        if (
            min(
                self.output_k,
                self.first_stage_per_channel_k,
                self.rerank_input_k,
                self.context_k,
            )
            <= 0
        ):
            raise ValueError("all retrieval budgets must be positive")
        if self.rerank_input_k < self.output_k:
            raise ValueError("rerank_input_k must be at least output_k")

    @property
    def candidate_k(self) -> int:
        return self.rerank_input_k


class BenchmarkExperimentRunner:
    """Run one evaluator-owned protocol with shared models and budgets."""

    def __init__(
        self,
        methods: tuple[str, ...] = BENCHMARK_METHODS,
        encoder=None,
        reranker=None,
        use_local_reranker: bool = True,
        generator=None,
        extraction_client=None,
        use_llm_extraction: bool = True,
        config: EvaluationConfig | None = None,
        generate_for_methods: tuple[str, ...] | None = None,
    ):
        unknown = set(methods) - set(ALL_EXPERIMENT_METHODS)
        if unknown:
            raise ValueError(f"unknown benchmark methods: {sorted(unknown)}")
        self.methods = methods
        self.generate_for_methods = set(
            methods if generate_for_methods is None else generate_for_methods
        )
        if not self.generate_for_methods <= set(methods):
            raise ValueError("generate_for_methods must be a subset of methods")
        self.encoder = encoder or LocalBGEEncoder()
        self.reranker = reranker or (LocalBGEReranker() if use_local_reranker else None)
        self.generator = generator or EvidenceGenerator()
        self.use_llm_extraction = use_llm_extraction
        self.extraction_client = (
            (extraction_client or DeepSeekClient()) if use_llm_extraction else None
        )
        self.config = config or EvaluationConfig()
        if self.use_llm_extraction and self.reranker is None:
            raise ValueError("production benchmark runs require the shared local BGE reranker")
        self._validate_shared_llm()

    def run(
        self,
        suite: BenchmarkSuite,
        output_dir: str | Path,
        seed: int = 42,
    ) -> list[dict]:
        self._embedding_model_manifest = _embedding_manifest(self.encoder)
        self._reranker_model_manifest = _reranker_manifest(self.reranker)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        partial = output / "records.partial.jsonl"
        records = _load_resume_records(
            partial,
            output / "records.json",
            methods=set(self.methods),
            seed=seed,
        )
        completed = _completed_example_ids(records, set(self.methods))
        _atomic_write_jsonl(partial, records)
        pending_examples = [
            example
            for example in suite.examples
            if example.example_id not in completed
        ]
        with partial.open("a", encoding="utf-8") as checkpoint:
            committed_since_sync = 0
            with ThreadPoolExecutor(
                max_workers=min(
                    get_settings().benchmark_example_workers,
                    max(len(pending_examples), 1),
                )
            ) as example_pool:
                futures = [
                    example_pool.submit(self._run_example, example, seed)
                    for example in pending_examples
                ]
                for future in as_completed(futures):
                    example_records = future.result()
                    records.extend(example_records)
                    for record in example_records:
                        checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    committed_since_sync += 1
                    if committed_since_sync >= get_settings().checkpoint_every:
                        os.fsync(checkpoint.fileno())
                        committed_since_sync = 0
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
        records = _canonical_record_order(records, suite, self.methods)
        _atomic_write_jsonl(partial, records)
        metadata = self._metadata(suite, seed)
        write_report(records, output_dir, metadata)
        partial.unlink(missing_ok=True)
        return records

    def _run_example(self, example, seed: int) -> list[dict]:
        example_records: list[dict] = []
        corpus_build_started = perf_counter()
        try:
            built = (
                build_example_corpus(example, self.extraction_client)
                if self.use_llm_extraction
                else build_sentence_fixture_corpus(example)
            )
        except Exception as exc:
            corpus_build_ms = (perf_counter() - corpus_build_started) * 1000
            return self._extraction_failure_records(
                example,
                None,
                seed,
                corpus_build_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

        corpus_build_ms = (perf_counter() - corpus_build_started) * 1000
        if not built.corpus.entities or not built.corpus.evidence_hyperedges:
            return self._extraction_failure_records(
                example,
                built,
                seed,
                corpus_build_ms,
                error="LLM extraction returned no usable facts",
            )

        shared_index_started = perf_counter()
        pipeline = S2RAGPipeline(
            built.corpus,
            text_encoder=self.encoder,
            use_local_reranker=False,
            generator=self.generator,
        )
        shared_index_ms = (perf_counter() - shared_index_started) * 1000
        expected_candidates = min(
            self.config.rerank_input_k,
            len(built.corpus.evidence_hyperedges),
        )
        score_inputs = []
        for method in self.methods:
            preparation_started = perf_counter()
            prepared = prepare_internal_baseline(pipeline, method, seed)
            method_preparation_ms = (perf_counter() - preparation_started) * 1000
            retrieval_started = perf_counter()
            candidates, channel_candidate_counts = self._retrieve_candidates(
                pipeline,
                example.question,
                method,
                seed,
                prepared,
            )
            first_stage_ms = (perf_counter() - retrieval_started) * 1000
            if len(candidates) != expected_candidates:
                raise RuntimeError(
                    f"{method} returned {len(candidates)} candidates; "
                    f"expected {expected_candidates}"
                )
            rerank_ms = 0.0
            reranker_enabled = (
                self.reranker is not None
                and method != "s2rag_no_bge_reranker"
            )
            if reranker_enabled:
                rerank_started = perf_counter()
                candidates = rerank_scored_facts(
                    example.question,
                    candidates,
                    pipeline.fact_text_by_id,
                    self.reranker,
                )
                rerank_ms = (perf_counter() - rerank_started) * 1000
            score_inputs.append(
                (
                    example,
                    built,
                    pipeline,
                    method,
                    candidates,
                    channel_candidate_counts,
                    seed,
                    shared_index_ms,
                    method_preparation_ms,
                    first_stage_ms,
                    rerank_ms,
                    corpus_build_ms,
                    reranker_enabled,
                )
            )
        with ThreadPoolExecutor(
            max_workers=min(
                get_settings().generation_workers,
                max(len(score_inputs), 1),
            )
        ) as generation_pool:
            example_records.extend(
                generation_pool.map(
                    lambda values: self._score_method(*values),
                    score_inputs,
                )
            )
        return example_records

    def _retrieve_candidates(
        self,
        pipeline,
        question: str,
        method: str,
        seed: int,
        prepared: dict,
    ) -> tuple[list[ScoredFact], dict[str, int]]:
        s2rag_variants = {
            "reified_fact_hybrid": (
                ("graph_full", "graph_multi", "dense", "bm25"),
                True,
            ),
            "s2rag_no_bm25": (
                ("graph_full", "graph_multi", "dense"),
                True,
            ),
            "s2rag_no_dense": (
                ("graph_full", "graph_multi", "bm25"),
                True,
            ),
            "s2rag_no_spectral": (("dense", "bm25"), True),
            "s2rag_no_graph_rerank": (
                ("graph_full", "graph_multi", "dense", "bm25"),
                False,
            ),
            "s2rag_no_bge_reranker": (
                ("graph_full", "graph_multi", "dense", "bm25"),
                True,
            ),
        }
        if method in s2rag_variants:
            channels, apply_graph_rerank = s2rag_variants[method]
            result = pipeline.retrieve_fact_candidates(
                question,
                per_channel_k=self.config.first_stage_per_channel_k,
                candidate_count=self.config.rerank_input_k,
                channels=channels,
                apply_graph_rerank=apply_graph_rerank,
            )
            return result.facts, result.channel_candidate_counts
        facts = retrieve_internal_candidates(
            pipeline,
            question,
            method,
            candidate_k=self.config.rerank_input_k,
            seed=seed,
            prepared=prepared,
        )
        return facts, {method: len(facts)}

    def _score_method(
        self,
        example,
        built,
        pipeline,
        method: str,
        candidates: list[ScoredFact],
        channel_candidate_counts: dict[str, int],
        seed: int,
        shared_index_ms: float,
        method_preparation_ms: float,
        first_stage_ms: float,
        rerank_ms: float,
        corpus_build_ms: float,
        reranker_enabled: bool,
    ) -> dict:
        passage_candidates = aggregate_passages(candidates, built.fact_to_passage)
        ranked_facts = [fact.fact_id for fact in candidates[: self.config.output_k]]
        ranked_fact_evidence = _evidence_units_from_facts(
            ranked_facts,
            pipeline.corpus,
        )
        passage_ranking = passage_candidates[: self.config.output_k]
        context_fact_ids = ranked_facts[: self.config.context_k]
        context_fact_evidence = _evidence_units_from_facts(
            context_fact_ids,
            pipeline.corpus,
        )
        context_passage_ids = _passages_from_facts(
            context_fact_ids,
            built.fact_to_passage,
        )
        context, _ = build_context(
            pipeline.corpus,
            context_fact_ids,
            limit=self.config.context_k,
        )
        should_generate = method in self.generate_for_methods
        generation_started = perf_counter()
        generation_error = None
        answer = ""
        if should_generate:
            try:
                answer = self.generator.generate(example.question, context)
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
        generation_ms = (perf_counter() - generation_started) * 1000

        citations_requested = (
            should_generate
            and getattr(self.generator, "citation_capability", "fact") != "none"
        )
        valid_fact_ids = set(built.fact_to_passage)
        received_citations = extract_citation_ids(answer) if citations_requested else []
        generated_fact_citations = [
            citation for citation in received_citations if citation in valid_fact_ids
        ]
        invalid_citations = [
            citation for citation in received_citations if citation not in valid_fact_ids
        ]
        generated_fact_citation_evidence = _evidence_units_from_facts(
            generated_fact_citations,
            pipeline.corpus,
        )
        gold_passages = _gold_passages(built)
        answer_metric = (
            answer_scores(
                answer,
                example.answer,
                profile=example.metadata.get("metric_profile", "hotpotqa_official"),
            )
            if should_generate
            else _unavailable_answer_scores()
        )
        citation_metric = (
            set_scores(
                generated_fact_citation_evidence,
                built.gold_chunk_ids,
            )
            if citations_requested
            else None
        )
        joint = _joint_scores(answer_metric, citation_metric)
        official_metrics = score_official_metrics(
            dataset=example.dataset,
            answer_metric=answer_metric,
            predicted_sentence_ids=context_fact_evidence,
            gold_sentence_ids=built.gold_chunk_ids,
            predicted_passage_ids=context_passage_ids,
            gold_passage_ids=gold_passages,
            answer_available=should_generate,
            sentence_support_available=True,
            passage_support_available=True,
        )
        generator_manifest = self.generator.manifest()

        record = {
            "dataset": example.dataset,
            "split": example.split,
            "example_id": example.example_id,
            "system": method,
            "seed": seed,
            "query_type": example.query_type,
            "hop_count": example.hop_count,
            "status": "failed" if generation_error else "success",
            "result_missing": False,
            "result_failed": bool(generation_error),
            "generation_failed": bool(generation_error),
            "error": generation_error,
            "retrieval_protocol": UNIFIED_RETRIEVAL_PROTOCOL,
            "ranking_origin": (
                "s2rag_reified_fact_hybrid"
                if method == "reified_fact_hybrid"
                else f"internal:{method}"
            ),
            "graph_model_type": "training_free",
            "graph_model_id": TRAINING_FREE_MODEL_ID,
            "fact_ranking_available": True,
            "passage_ranking_available": True,
            "answer_metric_available": should_generate,
            "generated_fact_citation_available": citations_requested,
            "generated_passage_citation_available": citations_requested,
            "mapping_coverage": 1.0,
            "extraction_coverage": built.extraction_coverage,
            "answer": answer,
            "retrieval_evidence_fact_ids": context_fact_ids,
            "retrieval_evidence_sentence_ids": context_fact_evidence,
            "retrieval_evidence_passage_ids": context_passage_ids,
            "generated_fact_citations": generated_fact_citations,
            "generated_fact_citation_sentence_ids": generated_fact_citation_evidence,
            "generated_passage_citations": _passages_from_facts(
                generated_fact_citations, built.fact_to_passage
            ),
            "citation_status": _citation_status(
                "failed" if generation_error else "success",
                len(received_citations),
                len(generated_fact_citations),
                invalid_citations,
                bool(built.gold_chunk_ids),
                False,
            )
            if citations_requested
            else "not_requested",
            "received_citation_count": len(received_citations),
            "valid_citation_count": len(generated_fact_citations),
            "citation_mapping_coverage": (
                len(generated_fact_citations) / len(received_citations)
                if received_citations
                else 0.0
            ),
            "unmapped_citation_ids": invalid_citations,
            "retrieved_candidate_count": len(candidates),
            "canonical_candidate_count_before_cutoff": len(candidates),
            "per_channel_candidate_counts": channel_candidate_counts,
            "reranker_enabled": reranker_enabled,
            "reranker_model_sha256": (
                self._reranker_model_manifest.get("model_sha256")
                if self._reranker_model_manifest is not None
                else None
            ),
            "rerank_input_count": len(candidates) if reranker_enabled else 0,
            "fact_candidate_count_before_cutoff": len(candidates),
            "passage_candidate_count_before_cutoff": len(passage_candidates),
            "fact_output_count": len(ranked_facts),
            "fact_evidence_output_count": len(ranked_fact_evidence),
            "passage_output_count": len(passage_ranking),
            "passage_aggregation": "max_fact_score_v1",
            "first_stage_per_channel_k": self.config.first_stage_per_channel_k,
            "first_stage_ms": first_stage_ms,
            "shared_rerank_ms": rerank_ms,
            "retrieval_ms": first_stage_ms + rerank_ms,
            "generation_ms": generation_ms,
            "query_total_ms": first_stage_ms + rerank_ms + generation_ms,
            "latency_ms": first_stage_ms + rerank_ms + generation_ms,
            "shared_index_ms": shared_index_ms,
            "method_preparation_ms": method_preparation_ms,
            "corpus_build_ms": corpus_build_ms,
            "total_preparation_ms": (corpus_build_ms + shared_index_ms + method_preparation_ms),
            "generation_protocol": (
                generator_manifest["generation_protocol"] if should_generate else None
            ),
            "generation_trace": {
                **generator_manifest,
                "context_budget": self.config.context_k,
            },
            **joint,
            **official_metrics,
        }
        record.update(
            _prefixed_ranking_scores(
                "fact",
                ranked_fact_evidence,
                built.gold_chunk_ids,
            )
        )
        record.update(_prefixed_ranking_scores("passage", passage_ranking, gold_passages))
        record.update(answer_metric)
        record.update(
            _prefixed_set_scores(
                "retrieval_evidence_fact",
                context_fact_evidence,
                built.gold_chunk_ids,
            )
        )
        record.update(
                _prefixed_set_scores(
                    "retrieval_evidence_passage",
                    context_passage_ids,
                    gold_passages,
                )
        )
        record.update(
            (
                {f"generated_fact_citation_{key}": value for key, value in citation_metric.items()}
                if citation_metric is not None
                else {f"generated_fact_citation_{name}": None for name in SET_METRIC_NAMES}
            )
        )
        record.update(
            _prefixed_set_scores(
                "generated_passage_citation",
                record["generated_passage_citations"],
                gold_passages,
                available=citations_requested,
            )
        )
        return record

    def _metadata(self, suite: BenchmarkSuite, seed: int) -> dict:
        generator_manifest = self.generator.manifest()
        example_ids = sorted(example.example_id for example in suite.examples)
        official_spec = official_metric_spec(suite.name)
        return {
            "dataset": suite.name,
            "split": suite.split,
            "source": suite.source,
            "source_sha256": _source_sha256(suite.source),
            "expected_examples": len(example_ids),
            "expected_example_ids_sha256": _ids_sha256(example_ids),
            "seed": seed,
            "systems": list(self.methods),
            "protocol": UNIFIED_RETRIEVAL_PROTOCOL,
            "retrieval_protocol": UNIFIED_RETRIEVAL_PROTOCOL,
            "end_to_end_llm_extraction": self.use_llm_extraction,
            "output_k": self.config.output_k,
            "first_stage_per_channel_k": self.config.first_stage_per_channel_k,
            "rerank_input_k": self.config.rerank_input_k,
            "context_k": self.config.context_k,
            "passage_aggregation": "max_fact_score_v1",
            "graph_model_type": "training_free",
            "graph_model_id": TRAINING_FREE_MODEL_ID,
            "extraction_protocol": (
                OPEN_DOMAIN_EXTRACTION_PROTOCOL
                if self.use_llm_extraction
                else "test_sentence_fixture"
            ),
            "extraction_model": _extraction_manifest(self.extraction_client),
            "shared_embedding_model": self._embedding_model_manifest,
            "shared_reranker_model": self._reranker_model_manifest,
            "metric_definitions": {
                "fact_unit": (
                    "retrieved LLM facts projected to canonical HotpotQA "
                    "supporting-sentence chunk IDs"
                ),
                "precision_at_k": "strict: relevant top-k items divided by k",
                "returned_precision_at_k": (
                    "relevant returned top-k items divided by the number returned"
                ),
                "mrr_at_20": "reciprocal rank of the first relevant item within top 20",
                "mrr": "deprecated alias of mrr_at_20",
                "empty_gold_evidence": "N/A and excluded from metric aggregation",
                "answer": official_spec["answer"],
                "support": official_spec["support"],
                "evidence": official_spec["evidence"],
                "joint": official_spec["joint"],
            },
            "official_metrics": official_spec,
            **generator_manifest,
        }

    def _validate_shared_llm(self) -> None:
        extraction_model = _extraction_manifest(self.extraction_client).get("model_id")
        generation_model = self.generator.manifest().get("model_id")
        if extraction_model and generation_model and extraction_model != generation_model:
            raise ValueError("extraction and answer generation must use the same model")

    def _extraction_failure_records(
        self,
        example,
        built,
        seed,
        corpus_build_ms,
        *,
        error: str,
    ):
        records = []
        gold_fact_ids = (
            built.gold_fact_ids
            if built is not None and built.gold_fact_ids
            else {
                f"official_sentence:{item.passage_id}:{item.sentence_index}"
                for item in example.supporting_facts
            }
        )
        gold_passage_ids = {item.passage_id for item in example.supporting_facts}
        extraction_coverage = built.extraction_coverage if built is not None else 0.0
        for method in self.methods:
            answer_metric = answer_scores(
                "",
                example.answer,
                profile=example.metadata.get("metric_profile", "hotpotqa_official"),
            )
            citation_metric = set_scores([], gold_fact_ids)
            joint = _joint_scores(answer_metric, citation_metric)
            official_metrics = score_official_metrics(
                dataset=example.dataset,
                answer_metric=answer_metric,
                predicted_sentence_ids=[],
                gold_sentence_ids=gold_fact_ids,
                predicted_passage_ids=[],
                gold_passage_ids=gold_passage_ids,
                answer_available=True,
                sentence_support_available=True,
                passage_support_available=True,
            )
            record = {
                "dataset": example.dataset,
                "split": example.split,
                "example_id": example.example_id,
                "system": method,
                "seed": seed,
                "query_type": example.query_type,
                "hop_count": example.hop_count,
                "status": "failed",
                "result_missing": False,
                "result_failed": True,
                "generation_failed": True,
                "error": error,
                "retrieval_protocol": UNIFIED_RETRIEVAL_PROTOCOL,
                "mapping_coverage": 1.0,
                "extraction_coverage": extraction_coverage,
                "corpus_build_ms": corpus_build_ms,
                "latency_ms": None,
                "first_stage_ms": None,
                "shared_rerank_ms": None,
                "retrieval_ms": None,
                "generation_ms": None,
                "query_total_ms": None,
                "shared_index_ms": None,
                "method_preparation_ms": None,
                "total_preparation_ms": None,
                "fact_ranking_available": True,
                "passage_ranking_available": True,
                "answer_metric_available": True,
                "generated_fact_citation_available": bool(gold_fact_ids),
                "generated_passage_citation_available": bool(gold_passage_ids),
                "answer": "",
                "retrieval_evidence_fact_ids": [],
                "retrieval_evidence_sentence_ids": [],
                "retrieval_evidence_passage_ids": [],
                "generated_fact_citations": [],
                "generated_passage_citations": [],
                **joint,
                **official_metrics,
            }
            record.update(_prefixed_ranking_scores("fact", [], gold_fact_ids))
            record.update(_prefixed_ranking_scores("passage", [], gold_passage_ids))
            record.update(answer_metric)
            record.update(
                _prefixed_set_scores(
                    "retrieval_evidence_fact",
                    [],
                    gold_fact_ids,
                )
            )
            record.update(
                _prefixed_set_scores(
                    "retrieval_evidence_passage",
                    [],
                    gold_passage_ids,
                )
            )
            record.update(
                _prefixed_set_scores(
                    "generated_fact_citation",
                    [],
                    gold_fact_ids,
                )
            )
            record.update(
                _prefixed_set_scores(
                    "generated_passage_citation",
                    [],
                    gold_passage_ids,
                )
            )
            records.append(record)
        return records


def score_external_result(
    example,
    built,
    result,
    seed: int = 42,
    config: EvaluationConfig | None = None,
) -> dict:
    """Score external traces under an evaluator-owned capability contract."""
    config = config or EvaluationConfig()
    capability = result.capability
    valid_fact_ids = set(built.fact_to_passage)
    valid_passage_ids = set(built.fact_to_passage.values())
    gold_passages = _gold_passages(built)
    failed = result.status in {"missing", "failed", "unscorable"}

    raw_passage_ranking = list(dict.fromkeys(result.document_ranking))
    passage_ranking = raw_passage_ranking[: config.output_k]
    raw_fact_ranking = list(dict.fromkeys(str(item) for item in result.fact_ranking))
    fact_ranking = _evidence_units_from_facts(
        raw_fact_ranking[: config.output_k],
        built.corpus,
        preserve_unknown=True,
    )
    if failed:
        passage_ranking = []
        fact_ranking = []

    passage_available = capability.supports_passage_ranking and result.shared_model_protocol_matched
    fact_available = capability.supports_fact_ranking and result.shared_model_protocol_matched
    answer_available = (
        capability.supports_answer_generation
        and result.generation_protocol_matched
        and result.shared_model_protocol_matched
    )
    no_gold_fact_evidence = not built.gold_chunk_ids
    no_gold_passage_evidence = not gold_passages
    fact_citation_available = (
        capability.citation_capability in {"fact", "both"}
        and result.shared_model_protocol_matched
        and not no_gold_fact_evidence
    )
    passage_citation_available = (
        capability.citation_capability in {"passage", "both"}
        and result.shared_model_protocol_matched
        and not no_gold_passage_evidence
    )

    citation_ids = [] if failed else list(dict.fromkeys(result.citations))
    generated_fact_citations = (
        [item for item in citation_ids if item in valid_fact_ids] if fact_citation_available else []
    )
    generated_fact_citation_evidence = _evidence_units_from_facts(
        generated_fact_citations,
        built.corpus,
    )
    generated_passage_citations = (
        [item for item in citation_ids if item in valid_passage_ids]
        if passage_citation_available
        else []
    )
    valid_citation_count = len(
        generated_fact_citations if fact_citation_available else generated_passage_citations
    )
    received_citation_count = 0 if failed else result.received_citation_count
    unmapped_citations = [] if failed else result.unmapped_citation_ids

    scored_answer = (
        "" if failed else _strip_known_citations(result.answer, result.citations, example)
    )
    answer_metric = (
        answer_scores(
            scored_answer,
            example.answer,
            profile=example.metadata.get("metric_profile", "hotpotqa_official"),
        )
        if answer_available
        else _unavailable_answer_scores()
    )
    fact_citation_metric = (
        set_scores(generated_fact_citation_evidence, built.gold_chunk_ids)
        if fact_citation_available
        else None
    )
    joint = (
        _joint_scores(answer_metric, fact_citation_metric)
        if answer_available and fact_citation_metric is not None
        else _unavailable_joint_scores()
    )
    official_metrics = score_official_metrics(
        dataset=example.dataset,
        answer_metric=answer_metric,
        predicted_sentence_ids=fact_ranking[: config.context_k],
        gold_sentence_ids=built.gold_chunk_ids,
        predicted_passage_ids=passage_ranking[: config.context_k],
        gold_passage_ids=gold_passages,
        answer_available=answer_available,
        sentence_support_available=fact_available,
        passage_support_available=passage_available,
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
        "result_failed": failed,
        "protocol_mismatch": (
            capability.supports_answer_generation and not result.generation_protocol_matched
        ),
        "shared_model_protocol_mismatch": (not result.shared_model_protocol_matched),
        "shared_model_trace": result.shared_model_trace,
        "error": result.error,
        "retrieval_protocol": UNIFIED_RETRIEVAL_PROTOCOL,
        "ranking_origin": result.ranking_origin,
        "fact_ranking_available": fact_available,
        "passage_ranking_available": passage_available,
        "answer_metric_available": answer_available,
        "generated_fact_citation_available": fact_citation_available,
        "generated_passage_citation_available": passage_citation_available,
        "mapping_coverage": result.mapping_coverage,
        "unmapped_ranking_ids": result.unmapped_ranking_ids,
        "answer": "" if failed else result.answer,
        "retrieval_evidence_sentence_ids": fact_ranking[: config.context_k],
        "retrieval_evidence_passage_ids": passage_ranking[: config.context_k],
        "generated_fact_citations": generated_fact_citations,
        "generated_fact_citation_sentence_ids": generated_fact_citation_evidence,
        "generated_passage_citations": generated_passage_citations,
        "citation_capability": capability.citation_capability,
        "citation_source": result.citation_source,
        "citation_status": (
            "unsupported"
            if capability.citation_capability == "none"
            else _citation_status(
                result.status,
                received_citation_count,
                valid_citation_count,
                unmapped_citations,
                (
                    not no_gold_fact_evidence
                    if capability.citation_capability == "fact"
                    else not no_gold_passage_evidence
                ),
                result.citation_parse_failed,
            )
        ),
        "received_citation_count": received_citation_count,
        "valid_citation_count": valid_citation_count,
        "citation_mapping_coverage": (
            valid_citation_count / received_citation_count if received_citation_count else 0.0
        ),
        "unmapped_citation_ids": unmapped_citations,
        "no_gold_evidence": (no_gold_fact_evidence and no_gold_passage_evidence),
        "no_gold_fact_evidence": no_gold_fact_evidence,
        "no_gold_passage_evidence": no_gold_passage_evidence,
        "native_index_ms": (
            result.indexing_seconds * 1000 if result.indexing_seconds is not None else None
        ),
        "native_retrieval_ms": (
            result.retrieval_seconds * 1000 if result.retrieval_seconds is not None else None
        ),
        "native_total_ms": (
            result.total_seconds * 1000 if result.total_seconds is not None else None
        ),
        "latency_ms": (result.total_seconds * 1000 if result.total_seconds is not None else None),
        "citation_level": capability.citation_capability,
        "generation_protocol": result.generation_protocol,
        "generation_trace": result.generation_trace,
        **joint,
        **official_metrics,
    }
    record.update(
        _prefixed_ranking_scores(
            "passage",
            passage_ranking,
            gold_passages,
            available=passage_available,
        )
    )
    record.update(
        _prefixed_ranking_scores(
            "fact",
            fact_ranking,
            built.gold_chunk_ids,
            available=fact_available,
        )
    )
    record.update(answer_metric)
    record.update(
        _prefixed_set_scores(
            "retrieval_evidence_passage",
            passage_ranking[: config.context_k],
            gold_passages,
            available=passage_available,
        )
    )
    record.update(
        _prefixed_set_scores(
            "retrieval_evidence_fact",
            fact_ranking[: config.context_k],
            built.gold_chunk_ids,
            available=fact_available,
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
            generated_fact_citation_evidence,
            built.gold_chunk_ids,
            available=fact_citation_available,
        )
    )
    return record


def _citation_status(
    status: str,
    received_count: int,
    valid_count: int,
    unmapped: list[str],
    has_gold: bool,
    parse_failed: bool,
) -> str:
    if not has_gold:
        return "no_gold_evidence"
    if status == "missing":
        return "result_missing"
    if status in {"failed", "unscorable"}:
        return "citation_mapping_failed" if unmapped else "result_failed"
    if parse_failed:
        return "citation_parse_failed"
    if received_count == 0:
        return "citation_empty"
    if unmapped and valid_count == 0:
        return "citation_invalid_id"
    if unmapped:
        return "citation_mapping_failed"
    return "success"


def _gold_passages(built) -> set[str]:
    return set(built.gold_passage_ids)


def _source_sha256(source: str) -> str | None:
    path = Path(source)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _embedding_manifest(encoder) -> dict:
    model_path = getattr(encoder, "model_path", None)
    return {
        "implementation": type(encoder).__name__,
        "model_path": (str(model_path.resolve()) if model_path is not None else None),
        "model_sha256": _path_sha256(model_path),
        "device": getattr(encoder, "device", None),
        "batch_size": getattr(encoder, "batch_size", None),
        "normalize_embeddings": True,
    }


def _extraction_manifest(client) -> dict:
    if client is None:
        return {"model_id": None, "mode": "test_sentence_fixture"}
    config = client.generation_config() if hasattr(client, "generation_config") else {}
    return {
        "implementation": type(client).__name__,
        "entity_prompt_sha256": hashlib.sha256(ENTITY_SYSTEM_PROMPT.encode()).hexdigest(),
        "fact_prompt_sha256": hashlib.sha256(FACT_SYSTEM_PROMPT.encode()).hexdigest(),
        **config,
    }


def _reranker_manifest(reranker) -> dict | None:
    if reranker is None:
        return None
    model_path = getattr(reranker, "model_path", None)
    return {
        "implementation": type(reranker).__name__,
        "model_path": (str(model_path.resolve()) if model_path is not None else None),
        "model_sha256": _path_sha256(model_path),
        "device": getattr(reranker, "device", None),
        "batch_size": getattr(reranker, "batch_size", None),
        "max_length": getattr(reranker, "max_length", None),
    }


def expected_shared_model_trace() -> dict:
    """Evaluator-owned model contract required for strict external imports."""
    embedding = _embedding_manifest(LocalBGEEncoder())
    reranker = _reranker_manifest(LocalBGEReranker()) or {}
    if embedding.get("model_sha256") is None or reranker.get("model_sha256") is None:
        raise ProviderError(
            "strict external comparison requires both local BGE model directories "
            "so their weight hashes can be recorded"
        )
    extraction_client = DeepSeekClient()
    try:
        extraction = _extraction_manifest(extraction_client)
    finally:
        extraction_client.close()
    return {
        "protocol": UNIFIED_RETRIEVAL_PROTOCOL,
        "embedding": {
            key: embedding.get(key)
            for key in (
                "model_sha256",
                "batch_size",
                "normalize_embeddings",
            )
        },
        "reranker": {
            key: reranker.get(key)
            for key in (
                "model_sha256",
                "batch_size",
                "max_length",
            )
        },
        "extraction": {
            key: extraction.get(key)
            for key in (
                "model_id",
                "temperature",
                "max_tokens",
                "entity_prompt_sha256",
                "fact_prompt_sha256",
            )
        },
    }


def _path_sha256(path) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for file_path in files:
        relative = file_path.name if path.is_file() else str(file_path.relative_to(path))
        digest.update(relative.encode())
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(example_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(example_ids).encode()).hexdigest()


def _passages_from_facts(
    fact_ids: list[str],
    fact_to_passage: dict[str, str],
) -> list[str]:
    return list(
        dict.fromkeys(fact_to_passage[item] for item in fact_ids if item in fact_to_passage)
    )


def _evidence_units_from_facts(
    fact_ids: list[str],
    corpus,
    *,
    preserve_unknown: bool = False,
) -> list[str]:
    evidence_by_fact = {
        fact.hyperedge_id: fact.evidence_chunk_ids for fact in corpus.evidence_hyperedges
    }
    output = []
    for fact_id in fact_ids:
        evidence_ids = evidence_by_fact.get(fact_id)
        if evidence_ids is None:
            if preserve_unknown:
                output.append(f"__unmapped_fact__:{fact_id}")
            continue
        output.extend(evidence_ids)
    return list(dict.fromkeys(output))


def _strip_known_citations(answer: str, citations: list[str], example=None) -> str:
    labels = set(citations)
    if example is not None:
        labels.update(
            label for passage in example.passages for label in (passage.passage_id, passage.title)
        )
    for citation in sorted(labels, key=len, reverse=True):
        answer = answer.replace(f"[{citation}]", " ")
    return answer


def _prefixed_ranking_scores(
    prefix: str,
    ranking: list[str],
    gold: set[str],
    available: bool = True,
) -> dict[str, float | None]:
    if not available:
        return {f"{prefix}_{name}": None for name in RANKING_METRIC_NAMES}
    return {f"{prefix}_{key}": value for key, value in ranking_scores(ranking, gold).items()}


def _prefixed_set_scores(
    prefix: str,
    predicted: list[str],
    gold: set[str],
    available: bool = True,
) -> dict[str, float | None]:
    if not available:
        return {f"{prefix}_{name}": None for name in SET_METRIC_NAMES}
    return {f"{prefix}_{key}": value for key, value in set_scores(predicted, gold).items()}


def _unavailable_answer_scores() -> dict[str, None]:
    return {name: None for name in ANSWER_METRIC_NAMES}


def _joint_scores(
    answer_metric: dict[str, float | None],
    citation_metric: dict[str, float | None] | None,
) -> dict[str, float | None]:
    if (
        citation_metric is None
        or any(citation_metric.get(name) is None for name in ("em", "precision", "recall"))
        or any(
            answer_metric.get(name) is None
            for name in ("answer_em", "answer_precision", "answer_recall")
        )
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


def _load_resume_records(
    partial: Path,
    completed_output: Path,
    *,
    methods: set[str],
    seed: int,
) -> list[dict]:
    source = partial if partial.exists() else completed_output
    if not source.exists():
        return []
    try:
        if source.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            rows = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    eligible = [row for row in rows if row.get("system") in methods and row.get("seed") == seed]
    completed = _completed_example_ids(eligible, methods)
    return [row for row in eligible if row.get("example_id") in completed]


def _completed_example_ids(records: list[dict], methods: set[str]) -> set[str]:
    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(str(row.get("example_id")), []).append(row)
    return {
        example_id
        for example_id, rows in grouped.items()
        if {str(row.get("system")) for row in rows} == methods
        and all(row.get("status") == "success" for row in rows)
    }


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)

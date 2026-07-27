import hashlib
import threading
import time
from dataclasses import replace

import numpy as np

from s2rag.benchmarks.schemas import (
    BenchmarkExample,
    BenchmarkSuite,
    Passage,
    SupportingFact,
)
from s2rag.evaluation.experiment import (
    HOTPOT_SHARED_GENERATION_PROTOCOL,
    UNIFIED_RETRIEVAL_PROTOCOL,
    BenchmarkExperimentRunner,
)
from s2rag.evaluation.experiment import score_external_result
from s2rag.evaluation.external_adapters import (
    ADAPTERS,
    SystemCapability,
    expected_shared_generation_trace,
    load_external_results,
)
from s2rag.evaluation.internal_baselines import BENCHMARK_METHODS, INTERNAL_BASELINES
from s2rag.benchmarks.corpus_builder import (
    build_native_passage_evaluation_corpus,
    build_sentence_fixture_corpus as build_example_corpus,
)


class LocalEncoder:
    def encode(self, texts):
        matrix = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.casefold().split():
                index = (
                    int.from_bytes(
                        hashlib.blake2b(token.encode(), digest_size=8).digest(),
                        "little",
                    )
                    % matrix.shape[1]
                )
                matrix[row, index] += 1
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class FakeGenerator:
    protocol = HOTPOT_SHARED_GENERATION_PROTOCOL

    def generate(self, question, context):
        evidence_ids = [
            line.removeprefix("[Evidence ").removesuffix("]")
            for line in context.splitlines()
            if line.startswith("[Evidence ")
        ]
        return "request caching " + " ".join(f"[{item}]" for item in evidence_ids)

    def manifest(self):
        return {
            "generation_protocol": self.protocol,
            "prompt_sha256": "test-prompt",
            "model_id": "fake-deepseek",
            "temperature": 0.0,
            "max_tokens": 128,
            "retry_policy": "bounded_exponential_backoff",
            "max_attempts": 1,
        }


class FakeReranker:
    def score(self, question, documents):
        return [float(len(document)) for document in documents]


class FailingReranker:
    def score(self, question, documents):
        raise AssertionError("BGE reranker must be bypassed for this ablation")


class FailingGenerator(FakeGenerator):
    def generate(self, question, context):
        raise RuntimeError("generation unavailable")


class ConcurrencyTrackingGenerator(FakeGenerator):
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def generate(self, question, context):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return super().generate(question, context)
        finally:
            with self.lock:
                self.active -= 1


class DifferentModelExtractionClient:
    def generation_config(self):
        return {"model_id": "different-model"}


def test_benchmark_runner_scores_multiple_methods(tmp_path):
    example = BenchmarkExample(
        example_id="ex_1",
        question="What improves response latency?",
        answer="request caching",
        passages=[
            Passage(
                passage_id="p1",
                title="Mechanism",
                sentences=["Request caching reduces repeated database reads."],
            ),
            Passage(
                passage_id="p2",
                title="Result",
                sentences=["Request caching improves response latency."],
            ),
        ],
        supporting_facts=[
            SupportingFact(passage_id="p1", sentence_index=0),
            SupportingFact(passage_id="p2", sentence_index=0),
        ],
        dataset="toy",
        split="test",
    )
    suite = BenchmarkSuite(name="toy", split="test", examples=[example], source="memory://toy")

    records = BenchmarkExperimentRunner(
        encoder=LocalEncoder(),
        use_local_reranker=False,
        generator=FakeGenerator(),
        use_llm_extraction=False,
    ).run(suite, tmp_path, seed=42)

    assert {record["system"] for record in records} == {
        "bm25",
        "dense",
        "reified_fact_hybrid",
    }
    assert (tmp_path / "records.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert all("fact_recall_at_5" in record for record in records)


def test_shared_reranker_receives_the_same_candidate_budget_for_every_method(tmp_path):
    example = BenchmarkExample(
        example_id="fair",
        question="What improves response latency?",
        answer="request caching",
        passages=[
            Passage(
                passage_id=f"p{index}",
                title=f"Passage {index}",
                sentences=[f"Request caching fact number {index}."],
            )
            for index in range(5)
        ],
        supporting_facts=[SupportingFact(passage_id="p0", sentence_index=0)],
        dataset="toy",
        split="test",
    )
    suite = BenchmarkSuite(
        name="toy",
        split="test",
        examples=[example],
        source="memory://toy",
    )
    records = BenchmarkExperimentRunner(
        encoder=LocalEncoder(),
        reranker=FakeReranker(),
        generator=FakeGenerator(),
        use_llm_extraction=False,
    ).run(suite, tmp_path, seed=42)

    assert len(records) == len(BENCHMARK_METHODS)
    assert {record["retrieval_protocol"] for record in records} == {UNIFIED_RETRIEVAL_PROTOCOL}
    assert {record["rerank_input_count"] for record in records} == {5}
    assert {record["retrieved_candidate_count"] for record in records} == {5}
    assert {record["system"] for record in records} == set(BENCHMARK_METHODS)


def test_no_bge_reranker_ablation_bypasses_shared_reranker(tmp_path):
    example = BenchmarkExample(
        example_id="no-bge",
        question="Who won?",
        answer="Ada",
        passages=[
            Passage(
                passage_id="p1",
                title="Result",
                sentences=["Ada won the final."],
            )
        ],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="toy",
        split="test",
    )
    suite = BenchmarkSuite(
        name="toy",
        split="test",
        examples=[example],
        source="memory://toy",
    )

    records = BenchmarkExperimentRunner(
        methods=("s2rag_no_bge_reranker",),
        generate_for_methods=(),
        encoder=LocalEncoder(),
        reranker=FailingReranker(),
        generator=FakeGenerator(),
        use_llm_extraction=False,
    ).run(suite, tmp_path, seed=42)

    assert len(records) == 1
    assert records[0]["system"] == "s2rag_no_bge_reranker"
    assert records[0]["reranker_enabled"] is False
    assert records[0]["rerank_input_count"] == 0
    assert records[0]["shared_rerank_ms"] == 0.0
    assert records[0]["answer_metric_available"] is False


def test_benchmark_examples_run_concurrently(tmp_path):
    examples = [
        BenchmarkExample(
            example_id=f"parallel-{index}",
            question="Who won?",
            answer="Ada",
            passages=[
                Passage(
                    passage_id=f"p{index}",
                    title="Result",
                    sentences=["Ada won."],
                )
            ],
            supporting_facts=[
                SupportingFact(passage_id=f"p{index}", sentence_index=0)
            ],
            dataset="toy",
            split="test",
        )
        for index in range(2)
    ]
    suite = BenchmarkSuite(
        name="toy",
        split="test",
        examples=examples,
        source="memory://toy",
    )
    generator = ConcurrencyTrackingGenerator()

    records = BenchmarkExperimentRunner(
        methods=("bm25",),
        encoder=LocalEncoder(),
        use_local_reranker=False,
        generator=generator,
        use_llm_extraction=False,
    ).run(suite, tmp_path, seed=42)

    assert len(records) == 2
    assert generator.max_active == 2


def test_generation_failure_scores_zero_instead_of_dropping_the_question(tmp_path):
    example = BenchmarkExample(
        example_id="generation-failed",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )

    records = BenchmarkExperimentRunner(
        encoder=LocalEncoder(),
        reranker=FakeReranker(),
        generator=FailingGenerator(),
        use_llm_extraction=False,
    ).run(suite, tmp_path, seed=42)

    assert len(records) == len(BENCHMARK_METHODS)
    assert all(record["status"] == "failed" for record in records)
    assert all(record["answer_f1"] == 0.0 for record in records)
    assert all(record["generated_fact_citation_f1"] == 0.0 for record in records)
    assert all(record["joint_f1"] == 0.0 for record in records)


def test_extraction_and_generation_model_must_match():
    with __import__("pytest").raises(ValueError, match="same model"):
        BenchmarkExperimentRunner(
            encoder=LocalEncoder(),
            reranker=FakeReranker(),
            generator=FakeGenerator(),
            extraction_client=DifferentModelExtractionClient(),
            use_llm_extraction=True,
        )


def test_all_internal_baselines_share_the_benchmark_contract(tmp_path):
    example = BenchmarkExample(
        example_id="ex_2",
        question="What improves response latency?",
        answer="request caching",
        passages=[
            Passage(
                passage_id="p1",
                title="Mechanism",
                sentences=["Request caching reduces repeated database reads."],
            ),
            Passage(
                passage_id="p2",
                title="Result",
                sentences=["Request caching improves response latency."],
            ),
        ],
        supporting_facts=[
            SupportingFact(passage_id="p1", sentence_index=0),
            SupportingFact(passage_id="p2", sentence_index=0),
        ],
        dataset="toy",
        split="test",
    )
    suite = BenchmarkSuite(name="toy", split="test", examples=[example], source="memory://toy")

    records = BenchmarkExperimentRunner(
        methods=INTERNAL_BASELINES,
        encoder=LocalEncoder(),
        use_local_reranker=False,
        generator=FakeGenerator(),
        use_llm_extraction=False,
    ).run(suite, tmp_path, seed=42)

    assert {record["system"] for record in records} == set(INTERNAL_BASELINES)
    assert all(record["ranking_origin"] for record in records)


def test_baseline_registry_contains_only_the_selected_methods():
    assert INTERNAL_BASELINES == ("bm25", "dense", "reified_fact_hybrid")
    assert BENCHMARK_METHODS == ("bm25", "dense", "reified_fact_hybrid")


def test_each_external_adapter_normalizes_title_rankings(tmp_path):
    example = BenchmarkExample(
        example_id="ex_3",
        question="What improves response latency?",
        answer="request caching",
        passages=[
            Passage(passage_id="p1", title="Mechanism", sentences=["Caching reduces reads."]),
            Passage(passage_id="p2", title="Result", sentences=["Latency improves."]),
        ],
        supporting_facts=[SupportingFact(passage_id="p2", sentence_index=0)],
        dataset="toy",
        split="test",
    )
    suite = BenchmarkSuite(name="toy", split="test", examples=[example], source="memory://toy")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        '[{"example_id":"ex_3","document_ranking":["Result","Mechanism"],'
        '"citations":["Result"],"answer":"request caching"}]',
        encoding="utf-8",
    )

    for name in ADAPTERS:
        normalized = load_external_results(name, result_path, suite)
        assert normalized[0].document_ranking == ["p2", "p1"]
        score = score_external_result(example, build_example_corpus(example), normalized[0])
        assert score["system"] == f"external:{name}"


def test_external_passage_ranking_is_not_expanded_into_fact_ranking(tmp_path):
    example = BenchmarkExample(
        example_id="ex_passage",
        question="Who won?",
        answer="Ada",
        passages=[
            Passage(passage_id="p1", title="Distractor", sentences=["No.", "Still no."]),
            Passage(passage_id="p2", title="Result", sentences=["Setup.", "Ada won."]),
        ],
        supporting_facts=[SupportingFact(passage_id="p2", sentence_index=1)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "external.json"
    result_path.write_text(
        '[{"example_id":"ex_passage","document_ranking":["Result"],'
        '"citations":["Result"],"answer":"Ada"}]',
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, build_example_corpus(example), result)

    assert score["passage_recall_at_1"] == 1.0
    assert score["fact_recall_at_1"] is None
    assert score["generated_passage_citation_f1"] == 1.0
    assert score["generated_fact_citation_f1"] is None
    assert score["answer_f1"] is None
    assert score["joint_f1"] is None


def test_native_passage_evaluation_uses_canonical_source_passages():
    example = BenchmarkExample(
        example_id="native-passage-view",
        question="Who won?",
        answer="Ada",
        passages=[
            Passage(passage_id="p1", title="Distractor", sentences=["No."]),
            Passage(passage_id="p2", title="Result", sentences=["Ada won."]),
        ],
        supporting_facts=[SupportingFact(passage_id="p2", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )

    built = build_native_passage_evaluation_corpus(example)

    assert set(built.fact_to_passage.values()) == {"p1", "p2"}
    assert built.gold_passage_ids == {"p2"}


def test_missing_external_results_stay_in_the_denominator(tmp_path):
    examples = [
        BenchmarkExample(
            example_id=example_id,
            question="Who won?",
            answer="Ada",
            passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
            supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
            dataset="hotpotqa",
            split="validation",
        )
        for example_id in ("present", "missing")
    ]
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=examples,
        source="memory://hotpot",
    )
    result_path = tmp_path / "partial.json"
    result_path.write_text(
        '[{"example_id":"present","document_ranking":["Result"]}]',
        encoding="utf-8",
    )

    results = load_external_results("lightrag", result_path, suite)
    missing = next(item for item in results if item.example_id == "missing")
    score = score_external_result(examples[1], build_example_corpus(examples[1]), missing)

    assert len(results) == 2
    assert missing.status == "missing"
    assert score["passage_recall_at_1"] == 0.0
    assert score["generated_passage_citation_f1"] == 0.0
    assert score["generated_passage_citation_available"] is True
    assert score["result_missing"] is True


def test_each_native_ranking_field_uses_the_hotpot_source_map(tmp_path):
    fields = {
        "graphrag": "community_ranking",
        "lightrag": "entity_ranking",
        "pathrag": "path_context_ranking",
        "hypergraphrag": "hyperedge_ranking",
        "hipporag2": "ppr_ranking",
        "cograg": "dual_hypergraph_ranking",
        "hgrag": "diffusion_ranking",
        "hyperrag": "hypergraph_ranking",
    }
    example = BenchmarkExample(
        example_id="mapped",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )

    for baseline, field in fields.items():
        result_path = tmp_path / f"{baseline}.json"
        result_path.write_text(
            __import__("json").dumps(
                [
                    {
                        "example_id": "mapped",
                        field: [{"id": "native-node-1"}],
                        "source_id_map": {"native-node-1": "Result"},
                    }
                ]
            ),
            encoding="utf-8",
        )
        result = load_external_results(baseline, result_path, suite)[0]
        assert result.document_ranking == ["p1"]
        assert result.mapping_coverage == 1.0
        assert result.status == "success"


def test_unmapped_native_ids_are_unscorable_instead_of_silently_dropped(tmp_path):
    example = BenchmarkExample(
        example_id="unmapped",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "unmapped.json"
    result_path.write_text(
        '[{"example_id":"unmapped","community_ranking":["unknown-community"]}]',
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, build_example_corpus(example), result)

    assert result.status == "unscorable"
    assert result.mapping_coverage == 0.0
    assert result.unmapped_ranking_ids == ["unknown-community"]
    assert score["passage_recall_at_1"] == 0.0
    assert score["result_failed"] is True


def test_partially_mapped_rankings_keep_unknown_items_as_irrelevant(tmp_path):
    example = BenchmarkExample(
        example_id="partially-mapped",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "partially-mapped.json"
    result_path.write_text(
        '[{"example_id":"partially-mapped","community_ranking":["unknown-community","p1"]}]',
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, build_example_corpus(example), result)

    assert result.status == "success"
    assert result.mapping_coverage == 0.5
    assert score["passage_recall_at_1"] == 0.0
    assert score["passage_recall_at_3"] == 1.0


def test_shared_hotpot_generation_respects_passage_citation_capability(tmp_path):
    example = BenchmarkExample(
        example_id="shared",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    built = build_example_corpus(example)
    fact_id = next(iter(built.gold_fact_ids))
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "shared.json"
    result_path.write_text(
        __import__("json").dumps(
            [
                {
                    "example_id": "shared",
                    "document_ranking": ["p1"],
                    "fact_ranking": [fact_id],
                    "answer": "Ada",
                    "citations": ["p1"],
                    "generation_protocol": HOTPOT_SHARED_GENERATION_PROTOCOL,
                    "generation_trace": expected_shared_generation_trace(),
                }
            ]
        ),
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, built, result)

    assert score["answer_f1"] == 1.0
    assert score["generated_passage_citation_f1"] == 1.0
    assert score["generated_fact_citation_f1"] is None
    assert score["joint_f1"] is None

    fact_capability = SystemCapability(
        supports_passage_ranking=True,
        supports_fact_ranking=True,
        supports_answer_generation=True,
        citation_capability="fact",
        generation_protocol=HOTPOT_SHARED_GENERATION_PROTOCOL,
    )
    fact_result = replace(
        result,
        capability=fact_capability,
        citations=[fact_id],
        citation_level="fact",
        received_citation_count=1,
        unmapped_citation_ids=[],
    )
    fact_score = score_external_result(example, built, fact_result)
    assert fact_score["generated_fact_citation_f1"] == 1.0
    assert fact_score["joint_f1"] == 1.0


def test_explicit_empty_citations_do_not_fall_back_to_answer(tmp_path):
    example = BenchmarkExample(
        example_id="empty-citation",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "empty-citation.json"
    result_path.write_text(
        __import__("json").dumps(
            [
                {
                    "example_id": example.example_id,
                    "document_ranking": ["p1"],
                    "answer": "Ada [p1]",
                    "citations": [],
                    "capability": {"citation_capability": "none"},
                }
            ]
        ),
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, build_example_corpus(example), result)

    assert result.capability.citation_capability == "passage"
    assert result.citations == []
    assert score["generated_passage_citation_f1"] == 0.0
    assert score["citation_status"] == "citation_empty"


def test_passage_title_citation_is_removed_before_answer_scoring(tmp_path):
    example = BenchmarkExample(
        example_id="title-citation",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "title-citation.json"
    result_path.write_text(
        __import__("json").dumps(
            [
                {
                    "example_id": example.example_id,
                    "document_ranking": ["p1"],
                    "answer": "Ada [Result]",
                    "generation_protocol": HOTPOT_SHARED_GENERATION_PROTOCOL,
                    "generation_trace": expected_shared_generation_trace(),
                }
            ]
        ),
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, build_example_corpus(example), result)

    assert score["answer_f1"] == 1.0
    assert score["generated_passage_citation_f1"] == 1.0


def test_invalid_citation_ids_score_zero_and_keep_mapping_audit(tmp_path):
    example = BenchmarkExample(
        example_id="invalid-citation",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "invalid-citation.json"
    result_path.write_text(
        '[{"example_id":"invalid-citation","document_ranking":["p1"],'
        '"citations":["fact_not_exist"]}]',
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, build_example_corpus(example), result)

    assert score["generated_passage_citation_f1"] == 0.0
    assert score["citation_status"] == "citation_invalid_id"
    assert score["received_citation_count"] == 1
    assert score["valid_citation_count"] == 0
    assert score["citation_mapping_coverage"] == 0.0


def test_generation_protocol_mismatch_excludes_the_entire_system(tmp_path):
    examples = [
        BenchmarkExample(
            example_id=example_id,
            question="Who won?",
            answer="Ada",
            passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
            supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
            dataset="hotpotqa",
            split="validation",
        )
        for example_id in ("matched", "mismatched")
    ]
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=examples,
        source="memory://hotpot",
    )
    expected_trace = expected_shared_generation_trace()
    mismatched_trace = {**expected_trace, "max_tokens": expected_trace["max_tokens"] + 1}
    result_path = tmp_path / "protocol-mismatch.json"
    result_path.write_text(
        __import__("json").dumps(
            [
                {
                    "example_id": "matched",
                    "document_ranking": ["p1"],
                    "answer": "Ada",
                    "generation_protocol": HOTPOT_SHARED_GENERATION_PROTOCOL,
                    "generation_trace": expected_trace,
                },
                {
                    "example_id": "mismatched",
                    "document_ranking": ["p1"],
                    "answer": "Ada",
                    "generation_protocol": HOTPOT_SHARED_GENERATION_PROTOCOL,
                    "generation_trace": mismatched_trace,
                },
            ]
        ),
        encoding="utf-8",
    )

    results = load_external_results("graphrag", result_path, suite)
    scores = [
        score_external_result(example, build_example_corpus(example), result)
        for example, result in zip(examples, results, strict=True)
    ]

    assert all(not result.generation_protocol_matched for result in results)
    assert all(score["protocol_mismatch"] for score in scores)
    assert all(score["answer_f1"] is None for score in scores)


def test_shared_model_manifest_gates_external_retrieval_metrics(tmp_path):
    example = BenchmarkExample(
        example_id="model-contract",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    expected_model_trace = {"protocol": UNIFIED_RETRIEVAL_PROTOCOL}
    result_path = tmp_path / "model-contract.json"
    result_path.write_text(
        '[{"example_id":"model-contract","document_ranking":["p1"]}]',
        encoding="utf-8",
    )

    mismatched = load_external_results(
        "graphrag",
        result_path,
        suite,
        expected_shared_model_trace=expected_model_trace,
    )[0]
    score = score_external_result(
        example,
        build_example_corpus(example),
        mismatched,
    )

    assert mismatched.shared_model_protocol_matched is False
    assert score["shared_model_protocol_mismatch"] is True
    assert score["passage_recall_at_5"] is None


def test_system_manifest_keeps_failed_questions_in_shared_generation(tmp_path):
    examples = [
        BenchmarkExample(
            example_id=example_id,
            question="Who won?",
            answer="Ada",
            passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
            supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
            dataset="hotpotqa",
            split="validation",
        )
        for example_id in ("success", "failed")
    ]
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=examples,
        source="memory://hotpot",
    )
    result_path = tmp_path / "system-manifest.json"
    result_path.write_text(
        __import__("json").dumps(
            {
                "manifest": {
                    "expected_examples": 2,
                    "expected_example_ids_sha256": hashlib.sha256(
                        "failed\nsuccess".encode()
                    ).hexdigest(),
                    "generation_protocol": HOTPOT_SHARED_GENERATION_PROTOCOL,
                    "generation_trace": expected_shared_generation_trace(),
                },
                "records": [
                    {
                        "example_id": "success",
                        "document_ranking": ["p1"],
                        "answer": "Ada",
                    },
                    {
                        "example_id": "failed",
                        "document_ranking": [],
                        "status": "failed",
                        "error": "timeout",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    results = load_external_results("graphrag", result_path, suite)
    failed_score = score_external_result(
        examples[1],
        build_example_corpus(examples[1]),
        results[1],
    )

    assert all(result.generation_protocol_matched for result in results)
    assert failed_score["answer_f1"] == 0.0
    assert failed_score["passage_recall_at_1"] == 0.0
    assert failed_score["generated_passage_citation_f1"] == 0.0
    assert failed_score["result_failed"] is True


def test_unparseable_answer_citation_is_audited(tmp_path):
    example = BenchmarkExample(
        example_id="parse-failed",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "parse-failed.json"
    result_path.write_text(
        '[{"example_id":"parse-failed","document_ranking":["p1"],"answer":"Ada [unknown source]"}]',
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, build_example_corpus(example), result)

    assert score["generated_passage_citation_f1"] == 0.0
    assert score["citation_status"] == "citation_parse_failed"

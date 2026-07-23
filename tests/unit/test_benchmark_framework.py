import json

from qmshe.benchmark_framework.adapters import load_internal_text_unit_traces
from qmshe.benchmark_framework.dataset import induce_fact_ranking, map_text_units_to_facts
from qmshe.benchmark_framework.metrics import evaluate_trace
from qmshe.benchmark_framework.report import aggregate
from qmshe.benchmark_framework.schemas import (
    CanonicalDocument,
    CanonicalExample,
    CanonicalFact,
    StandardTrace,
)


def _example():
    return CanonicalExample(
        example_id="e1",
        question="Where was Ada born?",
        answer="London",
        documents=(
            CanonicalDocument("d1", "Ada", "Ada was born in London."),
            CanonicalDocument("d2", "Noise", "Unrelated evidence."),
        ),
        facts=(
            CanonicalFact("f1", "d1", "born: Ada, London", "Ada was born in London."),
            CanonicalFact("f2", "d2", "states: unrelated", "Unrelated evidence."),
        ),
        gold_fact_ids=frozenset({"f1"}),
    )


def test_canonical_mapping_supports_document_and_text_unit_systems():
    example = _example()
    assert induce_fact_ranking(example, ["d1", "d2"]) == ["f1", "f2"]
    assert map_text_units_to_facts(example, ["Ada was born in London."]) == ["f1"]


def test_complete_metrics_include_all_k_accuracy_answer_citation_and_joint():
    trace = StandardTrace(
        system="test",
        example_id="e1",
        status="success",
        document_ranking=["d1", "d2"],
        fact_ranking=["f1", "f2"],
        induced_path=["d1"],
        answer="The London",
        citations=["d1"],
        ranking_origin="internal",
        path_origin="induced",
    )
    record = evaluate_trace(_example(), trace)
    assert record["passage_accuracy_at_1"] == 1
    assert record["fact_complete_at_40"] == 1
    assert record["answer_em"] == 1
    assert record["citation_em"] == 1
    assert record["joint_f1"] == 1


def test_aggregate_preserves_missing_usage_coverage_instead_of_zero_filling():
    trace = StandardTrace(system="test", example_id="e1", status="success")
    trace.usage.prompt_tokens = None
    summary = aggregate([evaluate_trace(_example(), trace)])["test"]
    assert "prompt_tokens_mean" not in summary
    assert summary["success_rate"] == 1


def test_native_text_unit_adapter_preserves_query_usage_and_canonical_ids(tmp_path):
    path = tmp_path / "native.jsonl"
    path.write_text(
        json.dumps(
            {
                "system": "official:pathrag",
                "example_id": "e1",
                "status": "success",
                "document_ranking": ["d1"],
                "fact_ranking": ["f1"],
                "text_unit_ranking": ["Ada was born in London."],
                "retrieval_seconds": 1.25,
                "usage": {
                    "llm_calls": 1,
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "embedding_calls": 2,
                    "embedding_tokens": 20,
                },
            }
        )
        + "\n"
    )
    trace = load_internal_text_unit_traces(path, {"e1": _example()})[0]
    assert trace.document_ranking == ["d1"]
    assert trace.fact_ranking[0] == "f1"
    assert trace.usage.prompt_tokens == 100
    assert trace.usage.token_count_mode == "query_measured_index_unavailable"
    assert trace.timing.retrieval_seconds == 1.25

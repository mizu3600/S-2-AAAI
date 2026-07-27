from pathlib import Path

import pytest

from scripts.run_protocol_v2_bridge import (
    MODEL_PROTOCOL,
    SYSTEM,
    _validate_base_model_directory,
    _validate_bridge_records,
    bridge_record,
    canonical_row_to_example,
)


def _canonical_row() -> dict:
    return {
        "dataset": "hotpotqa",
        "seed": 42,
        "source_example_id": "source-1",
        "example_id": "hotpotqa:seed_42:source-1",
        "question": "Where was Ada educated?",
        "answer": "London",
        "documents": [
            {
                "document_id": "doc_a",
                "title": "Ada",
                "text": "Ada studied in London. She wrote notes.",
            },
            {
                "document_id": "doc_b",
                "title": "Other",
                "text": "A distractor.",
            },
        ],
        "facts": [
            {
                "fact_id": "fact_a1",
                "document_id": "doc_a",
                "sentence": "Ada studied in London.",
                "text": "studied(Ada, London)",
            },
            {
                "fact_id": "fact_a2",
                "document_id": "doc_a",
                "sentence": "She wrote notes.",
                "text": "wrote(Ada, notes)",
            },
            {
                "fact_id": "fact_b1",
                "document_id": "doc_b",
                "sentence": "A distractor.",
                "text": "states(distractor)",
            },
        ],
        "gold_fact_ids": ["fact_a1"],
    }


def test_canonical_row_preserves_protocol_document_ids_and_gold_sentence():
    example = canonical_row_to_example(
        _canonical_row(),
        dataset="hotpotqa",
        seed=42,
    )

    assert [passage.passage_id for passage in example.passages] == ["doc_a", "doc_b"]
    assert example.passages[0].sentences == [
        "Ada studied in London.",
        "She wrote notes.",
    ]
    assert example.supporting_facts[0].passage_id == "doc_a"
    assert example.supporting_facts[0].sentence_index == 0
    assert example.gold_path == ["doc_a"]


def test_bridge_record_uses_one_training_free_system_and_seconds():
    row = bridge_record(
        {
            "example_id": "hotpotqa:seed_42:source-1",
            "status": "success",
            "passage_ranking": ["doc_a", "doc_a", "doc_b"],
            "total_preparation_ms": 1250,
            "retrieval_ms": 250,
            "graph_model_id": "analytic_graph_bands_v1",
            "extraction_coverage": 1.0,
            "error": None,
        },
        dataset="hotpotqa",
        seed=42,
        suite_sha256="abc",
        embedding_model=Path("/models/bge-m3"),
        reranker_model=Path("/models/bge-reranker-v2-m3"),
    )

    assert row["system"] == SYSTEM
    assert row["model_protocol"] == MODEL_PROTOCOL
    assert row["ranking"] == ["doc_a", "doc_b"]
    assert row["index_seconds"] == 1.25
    assert row["retrieval_seconds"] == 0.25
    assert row["usage"]["token_count_mode"].endswith("local_base_models")


def test_bridge_validation_rejects_noncanonical_ranking():
    row = _canonical_row()
    record = {
        "system": SYSTEM,
        "example_id": row["example_id"],
        "ranking": ["doc_not_in_suite"],
    }

    with pytest.raises(ValueError, match="non-canonical"):
        _validate_bridge_records([record], rows=[row], expected_count=1)


def test_canonical_row_rejects_empty_fact_text_without_shifting_gold_index():
    row = _canonical_row()
    row["facts"][0]["sentence"] = ""
    row["facts"][0]["text"] = ""

    with pytest.raises(ValueError, match="contains no usable"):
        canonical_row_to_example(row, dataset="hotpotqa", seed=42)


def test_base_model_preflight_rejects_lora_directory(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        '{"peft_type": "LORA"}',
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="PEFT/LoRA"):
        _validate_base_model_directory(tmp_path, "BGE-M3")

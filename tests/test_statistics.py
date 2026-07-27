from s2rag.evaluation.report import aggregate, render_markdown
from s2rag.evaluation.statistics import holm_adjust
from scripts.run_multi_seed_benchmark import (
    _collapse_records_across_seeds,
    _write_paired_statistics,
)


def test_unavailable_metrics_render_as_na_instead_of_zero():
    records = [
        {
            "system": "external:graphrag",
            "seed": 42,
            "passage_recall_at_5": 1.0,
            "fact_recall_at_5": None,
            "answer_f1": None,
            "generated_fact_citation_f1": None,
            "joint_f1": None,
            "mapping_coverage": 1.0,
            "result_failed": False,
        }
    ]

    summary = aggregate(records)
    report = render_markdown(
        summary,
        {"dataset": "hotpotqa", "examples": 1, "protocol": "test"},
    )

    assert summary["external:graphrag"]["fact_recall_at_5"]["mean"] is None
    retrieval_section = report.split("## Retrieval", 1)[1]
    retrieval_row = next(
        line
        for line in retrieval_section.splitlines()
        if line.startswith("| external:graphrag |")
    )
    assert "1.0000" in retrieval_row
    assert "N/A" in retrieval_row


def test_holm_adjustment_is_monotonic_in_sorted_pvalues():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})

    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}


def test_incomplete_seed_groups_are_not_silently_averaged(tmp_path):
    records = [
        {
            "retrieval_protocol": "unified_shared_models_v1",
            "system": "bm25",
            "example_id": "q1",
            "seed": 13,
            "fact_recall_at_5": 1.0,
            "passage_recall_at_5": 1.0,
            "answer_f1": 1.0,
        },
        {
            "retrieval_protocol": "unified_shared_models_v1",
            "system": "dense",
            "example_id": "q1",
            "seed": 13,
            "fact_recall_at_5": 0.0,
            "passage_recall_at_5": 0.0,
            "answer_f1": 0.0,
        },
        {
            "retrieval_protocol": "unified_shared_models_v1",
            "system": "dense",
            "example_id": "q1",
            "seed": 42,
            "fact_recall_at_5": 1.0,
            "passage_recall_at_5": 1.0,
            "answer_f1": 1.0,
        },
    ]

    collapsed = _collapse_records_across_seeds(records, expected_seeds=[13, 42])
    bm25 = next(item for item in collapsed if item["system"] == "bm25")
    assert bm25["seed_complete"] is False
    assert bm25["fact_recall_at_5"] is None

    output = tmp_path / "paired.json"
    _write_paired_statistics(records, output)
    assert all(
        not comparisons for comparisons in __import__("json").loads(output.read_text()).values()
    )

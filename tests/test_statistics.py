from qmshe.evaluation.report import aggregate, render_markdown
from qmshe.evaluation.statistics import holm_adjust


def test_unavailable_metrics_render_as_na_instead_of_zero():
    records = [{
        "system": "external:graphrag",
        "seed": 42,
        "passage_recall_at_5": 1.0,
        "fact_recall_at_5": None,
        "answer_f1": None,
        "generated_fact_citation_f1": None,
        "joint_f1": None,
        "mapping_coverage": 1.0,
        "result_failed": False,
    }]

    summary = aggregate(records)
    report = render_markdown(
        summary,
        {"dataset": "hotpotqa", "examples": 1, "protocol": "test"},
    )

    assert summary["external:graphrag"]["fact_recall_at_5"]["mean"] is None
    assert "| 1.0000 | N/A |" in report


def test_holm_adjustment_is_monotonic_in_sorted_pvalues():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})

    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}

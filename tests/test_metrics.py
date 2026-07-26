import pytest

from qmshe.evaluation.metrics import answer_scores, ranking_scores


def test_answer_f1_ignores_inline_fact_citations():
    scores = answer_scores(
        "surface passivation [fact_123] [fact_456]",
        "surface passivation",
    )

    assert scores == {
        "answer_em": 1.0,
        "answer_precision": 1.0,
        "answer_recall": 1.0,
        "answer_f1": 1.0,
    }


def test_multi_answer_scores_come_from_one_reference():
    scores = answer_scores("x y", ["x", "x y z w"])

    assert scores["answer_f1"] == pytest.approx(2 / 3)
    assert (scores["answer_precision"], scores["answer_recall"]) in {
        (0.5, 1.0),
        (1.0, 0.5),
    }


def test_empty_prediction_and_gold_are_consistent():
    scores = answer_scores("", "")

    assert scores["answer_em"] == 1.0
    assert scores["answer_f1"] == 1.0


def test_binary_retrieval_metrics_use_deduplicated_ranking():
    scores = ranking_scores(["a", "a", "b"], {"a", "b"}, ks=(1, 2))

    assert scores["recall_at_1"] == 0.5
    assert scores["recall_at_2"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["ndcg_at_2"] == 1.0

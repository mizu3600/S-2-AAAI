import pytest

from s2rag.evaluation.metrics import answer_scores, ranking_scores, set_scores


def test_answer_f1_ignores_inline_fact_citations():
    scores = answer_scores(
        "request caching [fact_123] [fact_456]",
        "request caching",
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
    assert scores["mrr_at_2"] == 1.0
    assert scores["ndcg_at_2"] == 1.0


def test_hotpot_yes_no_mismatch_has_zero_token_overlap_score():
    scores = answer_scores("yes maybe", "yes")

    assert scores["answer_em"] == 0.0
    assert scores["answer_precision"] == 0.0
    assert scores["answer_recall"] == 0.0
    assert scores["answer_f1"] == 0.0


def test_only_canonical_fact_citations_are_removed_from_answers():
    scores = answer_scores("published [1998]", "published 1998")

    assert scores["answer_f1"] == 1.0


def test_empty_gold_evidence_is_unavailable():
    assert all(value is None for value in set_scores([], set()).values())
    assert all(value is None for value in ranking_scores([], set(), ks=(1, 2)).values())


def test_strict_and_returned_precision_are_distinct():
    scores = ranking_scores(["a", "b"], {"a", "b"}, ks=(20,))

    assert scores["precision_at_20"] == 0.1
    assert scores["returned_precision_at_20"] == 1.0

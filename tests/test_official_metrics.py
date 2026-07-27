from s2rag.evaluation.metrics import answer_scores
from s2rag.evaluation.official_metrics import score_official_metrics


def test_hotpot_official_metrics_include_answer_support_and_joint():
    answer = answer_scores("Ada", "Ada", profile="hotpotqa_official")

    scores = score_official_metrics(
        dataset="hotpotqa",
        answer_metric=answer,
        predicted_sentence_ids=["s1", "wrong"],
        gold_sentence_ids={"s1", "s2"},
        predicted_passage_ids=["p1"],
        gold_passage_ids={"p1", "p2"},
        answer_available=True,
        sentence_support_available=True,
        passage_support_available=True,
    )

    assert scores["official_answer_f1"] == 1.0
    assert scores["official_support_f1"] == 0.5
    assert scores["official_joint_f1"] == 0.5


def test_musique_official_support_is_paragraph_level():
    answer = answer_scores("London", ["London", "Greater London"], profile="musique_official")

    scores = score_official_metrics(
        dataset="musique",
        answer_metric=answer,
        predicted_sentence_ids=[],
        gold_sentence_ids=set(),
        predicted_passage_ids=["p1", "p2"],
        gold_passage_ids={"p1", "p2"},
        answer_available=True,
        sentence_support_available=False,
        passage_support_available=True,
    )

    assert scores["official_answer_em"] == 1.0
    assert scores["official_support_em"] == 1.0
    assert scores["official_joint_f1"] is None


def test_2wiki_does_not_claim_official_evidence_or_joint_scores():
    answer = answer_scores("yes maybe", "yes", profile="2wikimultihopqa_official")

    scores = score_official_metrics(
        dataset="2wikimultihopqa",
        answer_metric=answer,
        predicted_sentence_ids=["s1"],
        gold_sentence_ids={"s1"},
        predicted_passage_ids=["p1"],
        gold_passage_ids={"p1"},
        answer_available=True,
        sentence_support_available=True,
        passage_support_available=True,
    )

    assert scores["official_answer_f1"] == 0.0
    assert scores["official_support_f1"] == 1.0
    assert scores["official_evidence_f1"] is None
    assert scores["official_joint_f1"] is None
    assert scores["official_metric_status"] == "official_partial"


def test_ultradomain_has_no_official_metric_namespace():
    answer = answer_scores("Ada", "Ada", profile="unified_squad_style")

    scores = score_official_metrics(
        dataset="ultradomain",
        answer_metric=answer,
        predicted_sentence_ids=[],
        gold_sentence_ids=set(),
        predicted_passage_ids=[],
        gold_passage_ids=set(),
        answer_available=True,
        sentence_support_available=False,
        passage_support_available=False,
    )

    assert scores["official_answer_f1"] is None
    assert scores["official_support_f1"] is None
    assert scores["official_metric_status"] == "no_official_scorer"

from qmshe.evaluation.retrieval_metrics import complete_at_k, hit_at_k, precision_at_k, recall_at_k


def test_multi_evidence_accuracy_metrics():
    ranked = ["noise", "gold-a", "noise-2", "gold-b"]
    relevant = {"gold-a", "gold-b"}
    assert hit_at_k(ranked, relevant, 2) == 1.0
    assert complete_at_k(ranked, relevant, 2) == 0.0
    assert complete_at_k(ranked, relevant, 4) == 1.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert precision_at_k(ranked, relevant, 2) == 0.5


def test_complete_accuracy_requires_gold_evidence():
    assert hit_at_k(["x"], set(), 1) == 0.0
    assert complete_at_k(["x"], set(), 1) == 0.0

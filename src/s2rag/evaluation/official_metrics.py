from __future__ import annotations

from s2rag.evaluation.metrics import set_scores


OFFICIAL_METRIC_FIELDS = (
    "official_answer_em",
    "official_answer_precision",
    "official_answer_recall",
    "official_answer_f1",
    "official_support_em",
    "official_support_precision",
    "official_support_recall",
    "official_support_f1",
    "official_evidence_em",
    "official_evidence_precision",
    "official_evidence_recall",
    "official_evidence_f1",
    "official_joint_em",
    "official_joint_precision",
    "official_joint_recall",
    "official_joint_f1",
)


OFFICIAL_METRIC_SPECS = {
    "hotpotqa": {
        "status": "official_compatible",
        "evaluator": "HotpotQA hotpot_evaluate_v1.py",
        "answer": "official EM/F1",
        "support": "sentence-level supporting-fact EM/F1",
        "evidence": "not defined separately",
        "joint": "answer x supporting-fact precision/recall",
        "corpus_protocol": "distractor_per_question_candidate_passages",
    },
    "musique": {
        "status": "official_compatible",
        "evaluator": "MuSiQue evaluate_v1.0.py",
        "answer": "official alias-aware EM/F1",
        "support": "paragraph-index support F1",
        "evidence": "not defined",
        "joint": "not defined",
        "corpus_protocol": "per_question_candidate_paragraphs",
    },
    "2wikimultihopqa": {
        "status": "official_partial",
        "evaluator": "2WikiMultiHopQA 2wikimultihop_evaluate_v1.1.py",
        "answer": "official alias-aware EM/F1",
        "support": "sentence-level supporting-fact EM/F1",
        "evidence": "N/A unless official relation triples are emitted",
        "joint": "N/A unless official relation triples are emitted",
        "corpus_protocol": "per_question_candidate_passages",
    },
    "ultradomain": {
        "status": "no_official_scorer",
        "evaluator": "none",
        "answer": "unified EM/token F1 only",
        "support": "N/A: no gold evidence",
        "evidence": "N/A: no gold evidence",
        "joint": "N/A",
        "corpus_protocol": "per_example_long_document",
    },
}


def official_metric_spec(dataset: str) -> dict:
    return OFFICIAL_METRIC_SPECS.get(
        dataset,
        {
            "status": "no_official_scorer",
            "evaluator": "none",
            "answer": "unified metrics only",
            "support": "N/A",
            "evidence": "N/A",
            "joint": "N/A",
            "corpus_protocol": "unspecified",
        },
    )


def score_official_metrics(
    *,
    dataset: str,
    answer_metric: dict[str, float | None],
    predicted_sentence_ids: list[str],
    gold_sentence_ids: set[str],
    predicted_passage_ids: list[str],
    gold_passage_ids: set[str],
    answer_available: bool,
    sentence_support_available: bool,
    passage_support_available: bool,
) -> dict[str, float | str | None]:
    spec = official_metric_spec(dataset)
    result: dict[str, float | str | None] = {
        field: None for field in OFFICIAL_METRIC_FIELDS
    }
    result.update(
        {
            "official_metric_status": spec["status"],
            "official_evaluator": spec["evaluator"],
            "official_corpus_protocol": spec["corpus_protocol"],
        }
    )
    if spec["status"] == "no_official_scorer":
        return result

    if answer_available:
        for name in ("em", "precision", "recall", "f1"):
            result[f"official_answer_{name}"] = answer_metric[f"answer_{name}"]

    support_metric = None
    if dataset in {"hotpotqa", "2wikimultihopqa"} and sentence_support_available:
        support_metric = set_scores(predicted_sentence_ids, gold_sentence_ids)
    elif dataset == "musique" and passage_support_available:
        support_metric = set_scores(predicted_passage_ids, gold_passage_ids)
    if support_metric is not None:
        for name in ("em", "precision", "recall", "f1"):
            result[f"official_support_{name}"] = support_metric[name]

    if dataset == "hotpotqa" and answer_available and support_metric is not None:
        joint = _joint_scores(answer_metric, support_metric)
        for name in ("em", "precision", "recall", "f1"):
            result[f"official_joint_{name}"] = joint[name]
    return result


def _joint_scores(
    answer_metric: dict[str, float | None],
    support_metric: dict[str, float | None],
) -> dict[str, float | None]:
    answer_precision = answer_metric["answer_precision"]
    answer_recall = answer_metric["answer_recall"]
    support_precision = support_metric["precision"]
    support_recall = support_metric["recall"]
    if None in (
        answer_metric["answer_em"],
        answer_precision,
        answer_recall,
        support_metric["em"],
        support_precision,
        support_recall,
    ):
        return {"em": None, "precision": None, "recall": None, "f1": None}
    precision = float(answer_precision) * float(support_precision)
    recall = float(answer_recall) * float(support_recall)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "em": float(answer_metric["answer_em"]) * float(support_metric["em"]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

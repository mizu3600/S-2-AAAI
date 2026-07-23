from __future__ import annotations

import re
import string
from collections import Counter

from qmshe.benchmark_framework.schemas import CanonicalExample, StandardTrace


KS = (1, 2, 5, 10, 20, 30, 40)


def _normalize_answer(text: str) -> str:
    text = text.casefold()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _answer_scores(prediction: str, gold: str) -> tuple[float, float, float, float]:
    predicted, expected = _normalize_answer(prediction), _normalize_answer(gold)
    exact = float(predicted == expected)
    if predicted in {"yes", "no", "noanswer"} or expected in {"yes", "no", "noanswer"}:
        return exact, exact, exact, exact
    predicted_tokens, expected_tokens = predicted.split(), expected.split()
    shared = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    precision = shared / len(predicted_tokens) if predicted_tokens else 0.0
    recall = shared / len(expected_tokens) if expected_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return exact, precision, recall, f1


def _set_scores(predicted: set[str], gold: set[str]) -> tuple[float, float, float, float]:
    precision = len(predicted & gold) / len(predicted) if predicted else float(not gold)
    recall = len(predicted & gold) / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(predicted == gold), precision, recall, f1


def _ranking_metrics(prefix: str, ranking: list[str], gold: set[str]) -> dict[str, float]:
    output = {}
    for k in KS:
        found = set(ranking[:k]) & gold
        output[f"{prefix}_recall_at_{k}"] = len(found) / len(gold) if gold else 0.0
        output[f"{prefix}_hit_at_{k}"] = float(bool(found))
        output[f"{prefix}_complete_at_{k}"] = float(bool(gold) and gold <= set(ranking[:k]))
        output[f"{prefix}_accuracy_at_{k}"] = output[f"{prefix}_hit_at_{k}"]
    output[f"{prefix}_mrr"] = next(
        (1 / rank for rank, item in enumerate(ranking, 1) if item in gold), 0.0
    )
    return output


def evaluate_trace(example: CanonicalExample, trace: StandardTrace) -> dict:
    gold_documents, gold_facts = set(example.gold_document_ids), set(example.gold_fact_ids)
    record = {
        "example_id": example.example_id,
        "system": trace.system,
        "status": trace.status,
        "success": float(trace.status == "success"),
        "error": trace.error,
        "ranking_origin": trace.ranking_origin,
        "path_origin": trace.path_origin,
        "citation_level": trace.citation_level,
        **_ranking_metrics("passage", trace.document_ranking, gold_documents),
        **_ranking_metrics("fact", trace.fact_ranking, gold_facts),
    }
    path = trace.native_paths[0] if trace.native_paths else trace.induced_path
    path_em, path_p, path_r, path_f1 = _set_scores(set(path), gold_documents)
    answer_em, answer_p, answer_r, answer_f1 = _answer_scores(trace.answer, example.answer)
    citation_gold = gold_documents if trace.citation_level == "document" else gold_facts
    citation_em, citation_p, citation_r, citation_f1 = _set_scores(
        set(trace.citations), citation_gold
    )
    # Hotpot-style joint precision and recall combine answer and evidence components.
    joint_precision = answer_p * citation_p
    joint_recall = answer_r * citation_r
    joint_f1 = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall
        else 0.0
    )
    usage = trace.usage.__dict__
    token_values = [usage["prompt_tokens"], usage["completion_tokens"], usage["embedding_tokens"]]
    total_model_tokens = (
        sum(value for value in token_values if value is not None)
        if any(value is not None for value in token_values)
        else None
    )
    record.update(
        {
            "path_em": path_em,
            "path_precision": path_p,
            "path_recall": path_r,
            "path_f1": path_f1,
            "answer_em": answer_em,
            "answer_precision": answer_p,
            "answer_recall": answer_r,
            "answer_f1": answer_f1,
            "citation_em": citation_em,
            "citation_precision": citation_p,
            "citation_recall": citation_r,
            "citation_f1": citation_f1,
            "joint_em": answer_em * citation_em,
            "joint_precision": joint_precision,
            "joint_recall": joint_recall,
            "joint_f1": joint_f1,
            **usage,
            "total_model_tokens": total_model_tokens,
            **trace.timing.__dict__,
            "metadata": trace.metadata,
        }
    )
    return record

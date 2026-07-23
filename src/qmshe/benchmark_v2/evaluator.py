from __future__ import annotations

import re
import string
from collections import Counter

from qmshe.benchmark_v2.schemas import CandidateView, RankingPrediction


def normalize_answer(text: str) -> str:
    text = text.casefold()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction: str, gold: str) -> tuple[float, float, float, float]:
    predicted, expected = normalize_answer(prediction), normalize_answer(gold)
    em = float(predicted == expected)
    if predicted in {"yes", "no", "noanswer"} or expected in {"yes", "no", "noanswer"}:
        f1 = em
    else:
        predicted_tokens, expected_tokens = predicted.split(), expected.split()
        common = Counter(predicted_tokens) & Counter(expected_tokens)
        shared = sum(common.values())
        if not predicted_tokens or not expected_tokens or not shared:
            f1 = 0.0
        else:
            precision = shared / len(predicted_tokens)
            recall = shared / len(expected_tokens)
            f1 = 2 * precision * recall / (precision + recall)
    precision = 1.0 if em else f1
    recall = 1.0 if em else f1
    return em, precision, recall, f1


def set_scores(predicted: set[str], gold: set[str]) -> tuple[float, float, float, float]:
    precision = len(predicted & gold) / len(predicted) if predicted else float(not gold)
    recall = len(predicted & gold) / len(gold) if gold else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(predicted == gold), precision, recall, f1


def evaluate_prediction(view: CandidateView, prediction: RankingPrediction, ks=(1, 2, 5, 10, 20, 30, 40)) -> dict[str, object]:
    fact_ranking = list(prediction.fact_ranking)
    passage_ranking = list(prediction.passage_ranking)
    gold_facts = set(view.gold_fact_ids)
    gold_passages = set(view.gold_passage_ids)
    record: dict[str, object] = {
        "example_id": view.example.example_id,
        "suite": "nary_stress" if view.example.example_id.startswith("nary_") else "hotpotqa_dev",
        "system": prediction.system,
        "candidate_count": view.candidate_count,
        "query_type": view.example.query_type,
        "level": view.example.level,
        "ranking_origin": prediction.diagnostics.get("ranking_origin", "unknown"),
    }
    for prefix, ranking, gold in (("fact", fact_ranking, gold_facts), ("passage", passage_ranking, gold_passages)):
        for k in ks:
            found = set(ranking[:k]) & gold
            record[f"{prefix}_recall_at_{k}"] = len(found) / len(gold) if gold else 0.0
            record[f"{prefix}_hit_at_{k}"] = float(bool(found))
            record[f"{prefix}_complete_at_{k}"] = float(bool(gold) and gold <= set(ranking[:k]))
        record[f"{prefix}_mrr"] = next((1 / rank for rank, item in enumerate(ranking, 1) if item in gold), 0.0)
    path_em, path_p, path_r, path_f1 = set_scores(set(prediction.path), gold_passages)
    answer_em, answer_p, answer_r, answer_f1 = answer_scores(prediction.answer, view.example.answer)
    citation_em, citation_p, citation_r, citation_f1 = set_scores(set(prediction.citations), gold_facts)
    valid_citations = set(prediction.citations) <= set(prediction.fact_ranking)
    record.update({
        "path_em": path_em, "path_precision": path_p, "path_recall": path_r, "path_f1": path_f1,
        "answer_em": answer_em, "answer_precision": answer_p, "answer_recall": answer_r, "answer_f1": answer_f1,
        "citation_em": citation_em, "citation_precision": citation_p, "citation_recall": citation_r,
        "citation_f1": citation_f1, "citation_valid": float(valid_citations),
        "joint_em": answer_em * citation_em,
        "joint_f1": answer_f1 * citation_f1,
        "diagnostics": prediction.diagnostics,
    })
    return record

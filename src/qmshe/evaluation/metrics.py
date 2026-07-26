from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub(" ", text)


def extract_citation_ids(text: str) -> list[str]:
    return list(dict.fromkeys(match.strip() for match in _CITATION_RE.findall(text)))


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", strip_citations(text)).casefold()
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(prediction: str, gold: str | list[str]) -> dict[str, float]:
    candidates = [gold] if isinstance(gold, str) else gold
    if not candidates:
        candidates = [""]
    scores = [_answer_pair(prediction, item) for item in candidates]
    best = max(scores, key=lambda score: (score[3], score[0], score[2], score[1]))
    return {
        name: best[index]
        for index, name in enumerate(("answer_em", "answer_precision", "answer_recall", "answer_f1"))
    }


def set_scores(predicted: list[str] | set[str], gold: set[str]) -> dict[str, float]:
    predicted_set = set(predicted)
    precision = len(predicted_set & gold) / len(predicted_set) if predicted_set else float(not gold)
    recall = len(predicted_set & gold) / len(gold) if gold else float(not predicted_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "em": float(predicted_set == gold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def ranking_scores(ranking: list[str], gold: set[str], ks: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict[str, float]:
    ranking = list(dict.fromkeys(ranking))
    result: dict[str, float] = {}
    for k in ks:
        found = set(ranking[:k]) & gold
        result[f"recall_at_{k}"] = len(found) / len(gold) if gold else 0.0
        result[f"precision_at_{k}"] = len(found) / max(k, 1)
        result[f"hit_at_{k}"] = float(bool(found))
        result[f"complete_at_{k}"] = float(bool(gold) and gold <= set(ranking[:k]))
    result["mrr"] = next(
        (1.0 / rank for rank, item in enumerate(ranking, 1) if item in gold), 0.0
    )
    for k in ks:
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(gold), k) + 1)
        )
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, item in enumerate(ranking[:k], 1)
            if item in gold
        )
        result[f"ndcg_at_{k}"] = dcg / ideal if ideal else 0.0
    return result


def citation_scores(citations: list[str], gold: set[str]) -> dict[str, float]:
    scores = set_scores(citations, gold)
    return {f"citation_{key}": value for key, value in scores.items()}


def _answer_pair(prediction: str, gold: str) -> tuple[float, float, float, float]:
    predicted, expected = normalize_answer(prediction), normalize_answer(gold)
    em = float(predicted == expected)
    predicted_tokens, expected_tokens = predicted.split(), expected.split()
    if not predicted_tokens and not expected_tokens:
        return 1.0, 1.0, 1.0, 1.0
    if not predicted_tokens or not expected_tokens:
        return em, 0.0, 0.0, 0.0
    shared = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    precision = shared / len(predicted_tokens)
    recall = shared / len(expected_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return em, precision, recall, f1

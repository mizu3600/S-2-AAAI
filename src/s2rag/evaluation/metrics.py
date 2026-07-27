from __future__ import annotations

import math
import re
import string
from collections import Counter


_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_FACT_CITATION_RE = re.compile(r"\[(fact_[A-Za-z0-9_-]+)\]")
_HOTPOT_SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


def strip_citations(text: str) -> str:
    """Remove only canonical fact citations, preserving ordinary bracketed answer text."""
    return _FACT_CITATION_RE.sub(" ", text)


def extract_citation_ids(text: str, allowed_ids: set[str] | None = None) -> list[str]:
    candidates = [match.strip() for match in _BRACKET_RE.findall(text)]
    if allowed_ids is None:
        candidates = [item for item in candidates if item.startswith("fact_")]
    else:
        candidates = [item for item in candidates if item in allowed_ids]
    return list(dict.fromkeys(candidates))


def normalize_answer(text: str) -> str:
    """HotpotQA official-style English answer normalization."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(char for char in value if char not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(strip_citations(text).lower())).split())


def answer_scores(
    prediction: str,
    gold: str | list[str],
    *,
    profile: str = "hotpotqa_official",
) -> dict[str, float]:
    candidates = [gold] if isinstance(gold, str) else gold
    if not candidates:
        candidates = [""]
    scores = [_answer_pair(prediction, item, profile=profile) for item in candidates]
    best = max(scores, key=lambda score: (score[3], score[0], score[2], score[1]))
    return {
        name: best[index]
        for index, name in enumerate(
            ("answer_em", "answer_precision", "answer_recall", "answer_f1")
        )
    }


def set_scores(predicted: list[str] | set[str], gold: set[str]) -> dict[str, float | None]:
    if not gold:
        return {"em": None, "precision": None, "recall": None, "f1": None}
    predicted_set = set(predicted)
    intersection = predicted_set & gold
    precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
    recall = len(intersection) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "em": float(predicted_set == gold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def ranking_scores(
    ranking: list[str],
    gold: set[str],
    ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[str, float | None]:
    ranking = list(dict.fromkeys(ranking))
    result: dict[str, float | None] = {}
    cutoff = max(ks)
    if not gold:
        for k in ks:
            for name in (
                "recall",
                "precision",
                "returned_precision",
                "hit",
                "complete",
                "ndcg",
            ):
                result[f"{name}_at_{k}"] = None
        result[f"mrr_at_{cutoff}"] = None
        result["mrr"] = None
        return result

    for k in ks:
        returned_count = min(k, len(ranking))
        found = set(ranking[:k]) & gold
        result[f"recall_at_{k}"] = len(found) / len(gold)
        result[f"precision_at_{k}"] = len(found) / k
        result[f"returned_precision_at_{k}"] = (
            len(found) / returned_count if returned_count else 0.0
        )
        result[f"hit_at_{k}"] = float(bool(found))
        result[f"complete_at_{k}"] = float(gold <= set(ranking[:k]))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
        dcg = sum(
            1.0 / math.log2(rank + 1) for rank, item in enumerate(ranking[:k], 1) if item in gold
        )
        result[f"ndcg_at_{k}"] = dcg / ideal

    reciprocal_rank = next(
        (1.0 / rank for rank, item in enumerate(ranking[:cutoff], 1) if item in gold),
        0.0,
    )
    result[f"mrr_at_{cutoff}"] = reciprocal_rank
    result["mrr"] = reciprocal_rank  # Deprecated compatibility alias.
    return result


def citation_scores(citations: list[str], gold: set[str]) -> dict[str, float | None]:
    scores = set_scores(citations, gold)
    return {f"citation_{key}": value for key, value in scores.items()}


def _answer_pair(
    prediction: str,
    gold: str,
    *,
    profile: str,
) -> tuple[float, float, float, float]:
    predicted, expected = normalize_answer(prediction), normalize_answer(gold)
    em = float(predicted == expected)
    if (
        profile == "hotpotqa_official"
        and predicted != expected
        and (predicted in _HOTPOT_SPECIAL_ANSWERS or expected in _HOTPOT_SPECIAL_ANSWERS)
    ):
        return em, 0.0, 0.0, 0.0

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

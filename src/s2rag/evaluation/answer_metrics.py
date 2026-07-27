"""Deprecated compatibility wrappers around canonical citation scoring."""

from s2rag.evaluation.metrics import extract_citation_ids, set_scores


def _scores(answer: str, gold_ids: set[str]):
    return set_scores(extract_citation_ids(answer), gold_ids)


def citation_precision(answer: str, allowed_ids: set[str]):
    return _scores(answer, allowed_ids)["precision"]


def citation_recall(answer: str, gold_ids: set[str]):
    return _scores(answer, gold_ids)["recall"]


def citation_f1(answer: str, gold_ids: set[str]):
    return _scores(answer, gold_ids)["f1"]

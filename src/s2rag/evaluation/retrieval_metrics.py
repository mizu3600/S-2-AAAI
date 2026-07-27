"""Deprecated compatibility wrappers around the canonical metric implementation."""

from s2rag.evaluation.metrics import ranking_scores


def recall_at_k(ranked: list[str], relevant: set[str], k: int):
    return ranking_scores(ranked, relevant, ks=(k,))[f"recall_at_{k}"]


def precision_at_k(ranked: list[str], relevant: set[str], k: int):
    return ranking_scores(ranked, relevant, ks=(k,))[f"precision_at_{k}"]


def returned_precision_at_k(ranked: list[str], relevant: set[str], k: int):
    return ranking_scores(ranked, relevant, ks=(k,))[f"returned_precision_at_{k}"]


def hit_at_k(ranked: list[str], relevant: set[str], k: int):
    return ranking_scores(ranked, relevant, ks=(k,))[f"hit_at_{k}"]


def complete_at_k(ranked: list[str], relevant: set[str], k: int):
    return ranking_scores(ranked, relevant, ks=(k,))[f"complete_at_{k}"]


def reciprocal_rank(ranked: list[str], relevant: set[str]):
    cutoff = max(len(ranked), 1)
    return ranking_scores(ranked, relevant, ks=(cutoff,))[f"mrr_at_{cutoff}"]


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int):
    return ranking_scores(ranked, relevant, ks=(k,))[f"ndcg_at_{k}"]

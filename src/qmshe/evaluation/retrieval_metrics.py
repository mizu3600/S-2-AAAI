import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / max(k, 1)


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Retrieval accuracy: one if at least one relevant item occurs in top-k."""
    return float(bool(set(ranked[:k]) & relevant))


def complete_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Multi-evidence exact accuracy: one if top-k covers every relevant item."""
    return float(bool(relevant) and relevant <= set(ranked[:k]))


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    return next((1.0 / rank for rank, item in enumerate(ranked, 1) if item in relevant), 0.0)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], 1) if item in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0

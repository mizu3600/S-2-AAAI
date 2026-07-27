from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredFact:
    fact_id: str
    score: float
    rank: int
    source: str


def rank_scored_facts(
    scores: dict[str, tuple[float, int, str]],
    limit: int | None = None,
) -> list[ScoredFact]:
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1][0], item[1][1], item[0]),
    )
    if limit is not None:
        ranked = ranked[:limit]
    return [
        ScoredFact(fact_id, values[0], rank, values[2])
        for rank, (fact_id, values) in enumerate(ranked, 1)
    ]


def aggregate_passages(
    facts: list[ScoredFact],
    fact_to_passage: dict[str, str],
) -> list[str]:
    best: dict[str, tuple[float, int]] = {}
    for fact in facts:
        passage_id = fact_to_passage.get(fact.fact_id)
        if passage_id is None:
            continue
        candidate = (fact.score, fact.rank)
        previous = best.get(passage_id)
        if (
            previous is None
            or candidate[0] > previous[0]
            or (candidate[0] == previous[0] and candidate[1] < previous[1])
        ):
            best[passage_id] = candidate
    return [
        passage_id
        for passage_id, _ in sorted(
            best.items(),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
        )
    ]


def rerank_scored_facts(
    question: str,
    facts: list[ScoredFact],
    fact_text_by_id: dict[str, str],
    reranker,
) -> list[ScoredFact]:
    documents = [fact_text_by_id[fact.fact_id] for fact in facts]
    logits = reranker.score(question, documents)
    if len(logits) != len(facts):
        raise ValueError("reranker returned a score count that differs from its input")
    ranked = sorted(
        zip(facts, logits, strict=True),
        key=lambda item: (-float(item[1]), item[0].rank, item[0].fact_id),
    )
    return [
        ScoredFact(fact.fact_id, float(score), rank, "shared_bge_rerank")
        for rank, (fact, score) in enumerate(ranked, 1)
    ]

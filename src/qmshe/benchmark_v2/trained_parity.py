from __future__ import annotations

from types import MethodType

from qmshe.retrieval.ann_retriever import SearchHit


def _facts_by_entity(corpus) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for fact in corpus.evidence_hyperedges:
        for argument in fact.arguments:
            output.setdefault(argument.entity_id, set()).add(fact.hyperedge_id)
    return output


def _aggregate(candidate_ids, direct_fact, facts_by_entity) -> list[str]:
    scores: dict[str, float] = {}
    for rank, candidate_id in enumerate(candidate_ids, 1):
        fact_ids = []
        if candidate_id in direct_fact:
            fact_ids.append(direct_fact[candidate_id])
        fact_ids.extend(sorted(facts_by_entity.get(candidate_id, set())))
        if candidate_id.startswith("fact_"):
            fact_ids.append(candidate_id)
        for fact_id in fact_ids:
            scores[fact_id] = max(scores.get(fact_id, 0.0), 1.0 / rank)
    return [fact_id for fact_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _fill_and_rerank(pipeline, question: str, candidates: list[str], budget: int) -> list[str]:
    lexical = [hit.object_id for hit in pipeline.bm25.search(question, budget)]
    fact_ids = list(dict.fromkeys([*candidates, *lexical]))[:budget]
    text_lookup = getattr(pipeline, "fact_text_by_id", getattr(pipeline, "text_by_id", {}))
    if pipeline.reranker is None or not fact_ids:
        return fact_ids
    order = pipeline.reranker.rank(question, [text_lookup[fact_id] for fact_id in fact_ids])
    return [fact_ids[index] for index in order]


def install_graph_parity_hooks(pipeline, budget: int = 60) -> None:
    """Make both ordinary graph profiles use one bounded Entity→Fact contract."""
    facts_by_entity = _facts_by_entity(pipeline.corpus)
    direct_fact = dict(pipeline.artifacts.fact_by_node)

    def facts_from_candidates(_self, candidate_ids):
        return _aggregate(candidate_ids, direct_fact, facts_by_entity)

    def remote_rerank(_self, question, fact_ids):
        return _fill_and_rerank(_self, question, fact_ids, budget)

    pipeline._facts_from_candidates = MethodType(facts_from_candidates, pipeline)
    pipeline._remote_rerank = MethodType(remote_rerank, pipeline)


def install_hypergraph_parity_hooks(pipeline, budget: int = 60) -> None:
    """Convert hypergraph entity and fact hits before applying the same budget."""
    facts_by_entity = _facts_by_entity(pipeline.corpus)
    direct_fact = {fact.hyperedge_id: fact.hyperedge_id for fact in pipeline.corpus.evidence_hyperedges}

    def remote_rerank(_self, question, hits):
        candidates = _aggregate([hit.object_id for hit in hits], direct_fact, facts_by_entity)
        fact_ids = _fill_and_rerank(_self, question, candidates, budget)
        score_by_id = {hit.object_id: hit.score for hit in hits}
        return [
            SearchHit(fact_id, score_by_id.get(fact_id, 0.0), rank, "v2-parity-fact")
            for rank, fact_id in enumerate(fact_ids, 1)
        ]

    pipeline._remote_rerank = MethodType(remote_rerank, pipeline)

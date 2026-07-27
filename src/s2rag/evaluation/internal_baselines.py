from __future__ import annotations

from s2rag.retrieval.ann_retriever import ExactVectorIndex
from s2rag.retrieval.candidates import ScoredFact


INTERNAL_BASELINES = (
    "bm25",
    "dense",
    "reified_fact_hybrid",
)
S2RAG_ABLATIONS = (
    "s2rag_no_bm25",
    "s2rag_no_dense",
    "s2rag_no_spectral",
    "s2rag_no_graph_rerank",
    "s2rag_no_bge_reranker",
)
BENCHMARK_METHODS = INTERNAL_BASELINES
ALL_EXPERIMENT_METHODS = (*BENCHMARK_METHODS, *S2RAG_ABLATIONS)


def rank_internal_baseline(
    pipeline,
    question: str,
    method: str,
    top_k: int = 20,
    candidate_k: int = 40,
    seed: int = 42,
    prepared: dict | None = None,
) -> list[str]:
    return [
        fact.fact_id
        for fact in retrieve_internal_candidates(
            pipeline,
            question,
            method,
            candidate_k=top_k,
            seed=seed,
            prepared=prepared,
        )
    ]


def retrieve_internal_candidates(
    pipeline,
    question: str,
    method: str,
    candidate_k: int,
    seed: int = 42,
    prepared: dict | None = None,
) -> list[ScoredFact]:
    prepared = prepared or prepare_internal_baseline(pipeline, method, seed)
    if method == "bm25":
        hits = pipeline.bm25.search(question, candidate_k)
    elif method == "dense":
        query = pipeline.text_encoder.encode([question])[0]
        hits = prepared["fact_index"].search(query, candidate_k, "dense")
    else:
        raise ValueError(f"unknown internal baseline: {method}")

    return [ScoredFact(hit.object_id, hit.score, hit.rank, hit.source) for hit in hits]


def prepare_internal_baseline(pipeline, method: str, seed: int = 42) -> dict:
    if method == "dense":
        return {"fact_index": _fact_raw_index(pipeline)}
    if method in {
        "bm25",
        "reified_fact_hybrid",
        *S2RAG_ABLATIONS,
    }:
        return {}
    raise ValueError(f"unknown internal baseline: {method}")


def _fact_raw_index(pipeline) -> ExactVectorIndex:
    fact_ids = [fact.hyperedge_id for fact in pipeline.corpus.evidence_hyperedges]
    node_index = {item: index for index, item in enumerate(pipeline.node_ids)}
    vectors = pipeline.raw_features.numpy()[[node_index[item] for item in fact_ids]]
    return ExactVectorIndex(fact_ids, vectors)

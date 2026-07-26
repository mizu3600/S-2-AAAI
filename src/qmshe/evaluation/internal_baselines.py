from __future__ import annotations

import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from qmshe.retrieval.ann_retriever import ExactVectorIndex
from qmshe.retrieval.seed_retriever import reciprocal_rank_fusion


INTERNAL_BASELINES = (
    "bm25",
    "dense",
    "bm25_dense_rrf",
    "node2vec",
    "laplacian_eigenmaps",
    "semantic_lap_pe",
    "semantic_ppr",
    "gcn",
    "graphsage",
    "hypergraph_conv",
    "reified_fact_hybrid",
)


def rank_internal_baseline(
    pipeline,
    question: str,
    method: str,
    top_k: int = 20,
    candidate_k: int = 40,
    seed: int = 42,
    prepared: dict | None = None,
) -> list[str]:
    prepared = prepared or prepare_internal_baseline(pipeline, method, seed)
    if method == "bm25":
        return [hit.object_id for hit in pipeline.bm25.search(question, top_k)]

    query = pipeline.text_encoder.encode([question])[0]
    if method == "dense":
        return [
            hit.object_id
            for hit in prepared["fact_index"].search(query, top_k, "dense")
        ]
    if method == "bm25_dense_rrf":
        hits = reciprocal_rank_fusion([
            pipeline.bm25.search(question, candidate_k),
            prepared["fact_index"].search(query, candidate_k, "dense"),
        ])
        return [hit.object_id for hit in hits[:top_k]]
    if method == "semantic_ppr":
        seeds = [
            hit.object_id
            for hit in pipeline.raw_index.search(query, min(8, len(pipeline.node_ids)), "dense")
        ]
        personalization = {node: 0.0 for node in pipeline.artifacts.graph}
        valid = [node for node in seeds if node in personalization]
        for node in valid:
            personalization[node] = 1.0 / len(valid)
        scores = nx.pagerank(pipeline.artifacts.graph, personalization=personalization)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return pipeline._facts_from_candidates(ranked)[:top_k]
    if method == "node2vec":
        ids, vectors = prepared["ids"], prepared["vectors"]
        return _seed_pool_rank(pipeline, query, ids, vectors, top_k)
    if method == "laplacian_eigenmaps":
        vectors = prepared["vectors"]
        return _seed_pool_rank(pipeline, query, pipeline.node_ids, vectors, top_k)
    if method == "semantic_lap_pe":
        lap_pe, vectors = prepared["lap_pe"], prepared["vectors"]
        raw_indices, weights = _seed_indices_and_weights(pipeline, query)
        structural = np.einsum("n,nd->d", weights, lap_pe[raw_indices])
        combined_query = np.concatenate([query, structural])
        return _rank_vectors_to_facts(pipeline, pipeline.node_ids, vectors, combined_query, top_k)
    if method == "gcn":
        return _seed_pool_rank(
            pipeline, query, pipeline.node_ids, prepared["vectors"], top_k
        )
    if method == "graphsage":
        propagated, vectors = prepared["propagated"], prepared["vectors"]
        raw_indices, weights = _seed_indices_and_weights(pipeline, query)
        structural = np.einsum("n,nd->d", weights, propagated[raw_indices])
        return _rank_vectors_to_facts(
            pipeline, pipeline.node_ids, vectors, np.concatenate([query, structural]), top_k
        )
    if method == "hypergraph_conv":
        return [
            hit.object_id
            for hit in prepared["fact_index"].search(query, top_k, "hypergraph-conv")
        ]
    raise ValueError(f"unknown internal baseline: {method}")


def prepare_internal_baseline(pipeline, method: str, seed: int = 42) -> dict:
    if method in {"dense", "bm25_dense_rrf"}:
        return {"fact_index": _fact_raw_index(pipeline)}
    if method == "node2vec":
        ids, vectors = _node2vec_embedding(pipeline.artifacts.graph, seed=seed)
        return {"ids": ids, "vectors": vectors}
    if method == "laplacian_eigenmaps":
        vectors = _laplacian_eigenmaps(_laplacian(pipeline.artifacts.propagation))
        return {"vectors": vectors}
    if method == "semantic_lap_pe":
        lap_pe = _laplacian_eigenmaps(_laplacian(pipeline.artifacts.propagation))
        vectors = np.concatenate([pipeline.raw_features.numpy(), lap_pe], axis=-1)
        return {"lap_pe": lap_pe, "vectors": vectors}
    if method == "gcn":
        vectors = np.asarray(
            pipeline.artifacts.propagation @ pipeline.raw_features.numpy()
        )
        return {"vectors": vectors}
    if method == "graphsage":
        propagated = np.asarray(
            pipeline.artifacts.propagation @ pipeline.raw_features.numpy()
        )
        vectors = np.concatenate([pipeline.raw_features.numpy(), propagated], axis=-1)
        return {"propagated": propagated, "vectors": vectors}
    if method == "hypergraph_conv":
        return {"fact_index": _hypergraph_incidence_index(pipeline)}
    return {}


def _fact_raw_index(pipeline) -> ExactVectorIndex:
    fact_ids = [fact.hyperedge_id for fact in pipeline.corpus.evidence_hyperedges]
    node_index = {item: index for index, item in enumerate(pipeline.node_ids)}
    vectors = pipeline.raw_features.numpy()[[node_index[item] for item in fact_ids]]
    return ExactVectorIndex(fact_ids, vectors)


def _seed_indices_and_weights(pipeline, query: np.ndarray) -> tuple[list[int], np.ndarray]:
    seeds = pipeline.raw_index.search(query, min(16, len(pipeline.node_ids)), "seed")
    indices = [pipeline.node_ids.index(hit.object_id) for hit in seeds]
    values = np.asarray([max(hit.score, 0.0) for hit in seeds], dtype=np.float32)
    return indices, values / max(float(values.sum()), 1e-12)


def _seed_pool_rank(pipeline, query: np.ndarray, ids: list[str], vectors: np.ndarray, top_k: int) -> list[str]:
    id_to_index = {item: index for index, item in enumerate(ids)}
    seed_ids = [
        hit.object_id
        for hit in pipeline.raw_index.search(query, min(16, len(pipeline.node_ids)), "seed")
        if hit.object_id in id_to_index
    ]
    if not seed_ids:
        return []
    weights = np.asarray([
        max(hit.score, 0.0)
        for hit in pipeline.raw_index.search(query, min(16, len(pipeline.node_ids)), "seed")
        if hit.object_id in id_to_index
    ], dtype=np.float32)
    weights /= max(float(weights.sum()), 1e-12)
    indices = [id_to_index[item] for item in seed_ids]
    structural_query = np.einsum("n,nd->d", weights, vectors[indices])
    return _rank_vectors_to_facts(pipeline, ids, vectors, structural_query, top_k)


def _rank_vectors_to_facts(
    pipeline, ids: list[str], vectors: np.ndarray, query: np.ndarray, top_k: int
) -> list[str]:
    hits = ExactVectorIndex(ids, vectors).search(query, len(ids), "structural")
    return pipeline._facts_from_candidates([hit.object_id for hit in hits])[:top_k]


def _laplacian(propagation: sp.spmatrix) -> sp.csr_matrix:
    return (sp.eye(propagation.shape[0], format="csr") - propagation).tocsr()


def _laplacian_eigenmaps(laplacian: sp.spmatrix, dimensions: int = 16) -> np.ndarray:
    size = laplacian.shape[0]
    if size <= 2:
        return np.eye(size, dtype=np.float32)
    k = min(dimensions + 1, size - 1)
    _, vectors = sp.linalg.eigsh(
        laplacian, k=k, which="SM", v0=np.ones(size, dtype=np.float32)
    )
    return vectors[:, 1:].astype(np.float32)


def _node2vec_embedding(
    graph: nx.Graph,
    dimensions: int = 32,
    walk_length: int = 12,
    walks_per_node: int = 4,
    window: int = 4,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    index = {node: position for position, node in enumerate(nodes)}
    counts: dict[tuple[int, int], float] = {}
    for start in nodes:
        for _ in range(walks_per_node):
            walk = [start]
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(rng.choice(neighbors))
            for position, node in enumerate(walk):
                for context in walk[max(0, position - window) : position + window + 1]:
                    if context != node:
                        pair = (index[node], index[context])
                        counts[pair] = counts.get(pair, 0.0) + 1.0
    rows, columns, values = zip(
        *((row, column, value) for (row, column), value in counts.items())
    ) if counts else ([], [], [])
    matrix = sp.csr_matrix((values, (rows, columns)), shape=(len(nodes), len(nodes)))
    if len(nodes) <= 2:
        return nodes, np.eye(len(nodes), dtype=np.float32)
    dimensions = min(dimensions, len(nodes) - 1)
    return nodes, TruncatedSVD(dimensions, random_state=seed).fit_transform(matrix).astype(np.float32)


def _hypergraph_incidence_index(pipeline) -> ExactVectorIndex:
    entities = [entity.entity_id for entity in pipeline.corpus.entities]
    facts = [fact.hyperedge_id for fact in pipeline.corpus.evidence_hyperedges]
    entity_index = {item: index for index, item in enumerate(entities)}
    fact_index = {item: index for index, item in enumerate(facts)}
    rows, columns = [], []
    for fact in pipeline.corpus.evidence_hyperedges:
        for argument in fact.arguments:
            rows.append(entity_index[argument.entity_id])
            columns.append(fact_index[fact.hyperedge_id])
    incidence = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(entities), len(facts)),
    )
    node_index = {item: index for index, item in enumerate(pipeline.node_ids)}
    entity_vectors = pipeline.raw_features.numpy()[[node_index[item] for item in entities]]
    fact_vectors = incidence.T @ entity_vectors
    fact_degree = np.maximum(np.asarray(incidence.sum(axis=0)).ravel(), 1.0)
    fact_vectors = np.asarray(fact_vectors) / fact_degree[:, None]
    return ExactVectorIndex(facts, fact_vectors)

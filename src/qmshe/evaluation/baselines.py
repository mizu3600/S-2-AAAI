import networkx as nx
import numpy as np
import scipy.sparse as sp

from qmshe.retrieval.ann_retriever import ExactVectorIndex


def dense_baseline(ids: list[str], vectors: np.ndarray, query: np.ndarray, top_k: int) -> list[str]:
    return [hit.object_id for hit in ExactVectorIndex(ids, vectors).search(query, top_k)]


def ppr_scores(graph: nx.Graph, seed_ids: list[str]) -> dict[str, float]:
    personalization = {node: 0.0 for node in graph}
    for node in seed_ids:
        if node in personalization:
            personalization[node] = 1.0 / max(len(seed_ids), 1)
    return nx.pagerank(graph, personalization=personalization)


def laplacian_eigenmaps(laplacian: sp.spmatrix, dimensions: int = 8) -> np.ndarray:
    n = laplacian.shape[0]
    if n <= 2:
        return np.eye(n, dtype=np.float32)
    k = min(dimensions + 1, n - 1)
    _, vectors = sp.linalg.eigsh(laplacian, k=k, which="SM")
    return vectors[:, 1:].astype(np.float32)


BASELINE_REGISTRY = {
    "bm25": "BM25Retriever",
    "dense": dense_baseline,
    "bm25+dense": "reciprocal_rank_fusion",
    "node2vec": "optional-networkx-adapter",
    "laplacian_eigenmaps": laplacian_eigenmaps,
    "semantic+lap_pe": "concat-adapter",
    "semantic+ppr": ppr_scores,
    "graphsage": "torch-geometric-adapter",
    "gcn": "torch-geometric-adapter",
    "hypergraph_conv": "torch-geometric-adapter",
    "qmshe": "QMSHEPipeline",
}


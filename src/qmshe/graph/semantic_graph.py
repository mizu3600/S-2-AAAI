from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SemanticGraph:
    adjacency: sp.csr_matrix
    similarities: dict[tuple[int, int], float]


def build_mutual_knn_graph(
    embeddings: np.ndarray, k: int = 15, threshold: float = 0.72, max_degree: int = 40
) -> SemanticGraph:
    n = len(embeddings)
    if n == 0:
        return SemanticGraph(sp.csr_matrix((0, 0)), {})
    neighbors = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=neighbors, metric="cosine").fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    directed: dict[tuple[int, int], float] = {}
    for i, (row_d, row_i) in enumerate(zip(distances, indices, strict=True)):
        for distance, j in zip(row_d, row_i, strict=True):
            similarity = 1.0 - float(distance)
            if i != j and similarity >= threshold:
                directed[(i, int(j))] = similarity
    mutual = {
        (i, j): score
        for (i, j), score in directed.items()
        if (j, i) in directed and i < j
    }
    degree = np.zeros(n, dtype=int)
    rows, cols, data, kept = [], [], [], {}
    for (i, j), score in sorted(mutual.items(), key=lambda item: item[1], reverse=True):
        if degree[i] >= max_degree or degree[j] >= max_degree:
            continue
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([score, score])
        degree[i] += 1
        degree[j] += 1
        kept[(i, j)] = score
    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return SemanticGraph(adjacency, kept)


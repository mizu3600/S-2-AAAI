from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchHit:
    object_id: str
    score: float
    rank: int
    source: str


class ExactVectorIndex:
    """Portable cosine gold index. Qdrant/FAISS can replace it without changing callers."""

    def __init__(self, object_ids: list[str], vectors: np.ndarray):
        if len(object_ids) != len(vectors):
            raise ValueError("ids and vectors differ in length")
        self.object_ids = object_ids
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-12)

    def search(self, query: np.ndarray, top_k: int, source: str = "dense") -> list[SearchHit]:
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = self.vectors @ query
        order = np.argsort(-scores)[: min(top_k, len(scores))]
        return [
            SearchHit(self.object_ids[index], float(scores[index]), rank + 1, source)
            for rank, index in enumerate(order)
        ]


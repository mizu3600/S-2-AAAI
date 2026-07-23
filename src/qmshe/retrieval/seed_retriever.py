from rank_bm25 import BM25Okapi

from qmshe.retrieval.ann_retriever import SearchHit


class BM25Retriever:
    def __init__(self, object_ids: list[str], texts: list[str]):
        self.object_ids = object_ids
        self.corpus = [text.lower().split() for text in texts]
        self.index = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if self.index is None:
            return []
        scores = self.index.get_scores(query.lower().split())
        order = scores.argsort()[::-1][:top_k]
        return [
            SearchHit(self.object_ids[index], float(scores[index]), rank + 1, "bm25")
            for rank, index in enumerate(order)
        ]


def reciprocal_rank_fusion(result_sets: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    for results in result_sets:
        for hit in results:
            scores[hit.object_id] = scores.get(hit.object_id, 0.0) + 1.0 / (k + hit.rank)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [SearchHit(object_id, score, rank + 1, "rrf") for rank, (object_id, score) in enumerate(ranked)]


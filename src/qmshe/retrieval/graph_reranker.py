import networkx as nx

from qmshe.retrieval.ann_retriever import SearchHit


def graph_rerank(
    hits: list[SearchHit], graph: nx.Graph, path_bonus: float = 0.05, max_hops: int = 4
) -> list[SearchHit]:
    candidate_ids = {hit.object_id for hit in hits if hit.object_id in graph}
    adjusted = []
    for hit in hits:
        connected = 0
        if hit.object_id in graph:
            lengths = nx.single_source_shortest_path_length(graph, hit.object_id, cutoff=max_hops)
            connected = sum(1 for node in candidate_ids if node != hit.object_id and node in lengths)
        adjusted.append((hit, hit.score + path_bonus * connected))
    adjusted.sort(key=lambda item: item[1], reverse=True)
    return [SearchHit(hit.object_id, score, rank + 1, "graph") for rank, (hit, score) in enumerate(adjusted)]


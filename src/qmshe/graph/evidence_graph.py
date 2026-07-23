import networkx as nx

from qmshe.ingest.schemas import Corpus


def build_evidence_graph(corpus: Corpus) -> nx.Graph:
    graph = nx.Graph()
    for entity in corpus.entities:
        graph.add_node(entity.entity_id, object_type="entity", evidence=True)
    for fact in corpus.evidence_hyperedges:
        graph.add_node(
            fact.hyperedge_id,
            object_type="hyperedge",
            evidence=True,
            chunk_ids=fact.evidence_chunk_ids,
        )
        for argument in fact.arguments:
            graph.add_edge(fact.hyperedge_id, argument.entity_id, role=argument.role)
    return graph


def evidence_paths(graph: nx.Graph, candidates: list[str], cutoff: int = 4) -> list[list[str]]:
    paths: list[list[str]] = []
    for i, source in enumerate(candidates):
        for target in candidates[i + 1 :]:
            if source not in graph or target not in graph:
                continue
            try:
                path = nx.shortest_path(graph, source, target)
            except nx.NetworkXNoPath:
                continue
            if len(path) - 1 <= cutoff:
                paths.append(path)
    return paths


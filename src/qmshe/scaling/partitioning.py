from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class GraphPartition:
    partition_id: str
    node_ids: list[str]


def partition_graph(graph: nx.Graph, max_partition_size: int = 50_000) -> list[GraphPartition]:
    if graph.number_of_nodes() == 0:
        return []
    communities = list(nx.community.greedy_modularity_communities(graph)) if graph.number_of_edges() else [set(graph)]
    partitions = []
    counter = 0
    for community in communities:
        ordered = sorted(community)
        for start in range(0, len(ordered), max_partition_size):
            partitions.append(GraphPartition(f"partition_{counter:05d}", ordered[start : start + max_partition_size]))
            counter += 1
    return partitions


def build_coarse_graph(
    graph: nx.Graph, partitions: list[GraphPartition], features: dict[str, np.ndarray]
) -> tuple[nx.Graph, dict[str, np.ndarray]]:
    coarse = nx.Graph()
    node_to_partition = {node: part.partition_id for part in partitions for node in part.node_ids}
    coarse_features = {}
    for partition in partitions:
        vectors = [features[node] for node in partition.node_ids if node in features]
        coarse.add_node(partition.partition_id, size=len(partition.node_ids))
        if vectors:
            coarse_features[partition.partition_id] = np.mean(vectors, axis=0)
    for left, right in graph.edges():
        source, target = node_to_partition[left], node_to_partition[right]
        if source != target:
            weight = coarse.get_edge_data(source, target, {}).get("weight", 0) + 1
            coarse.add_edge(source, target, weight=weight)
    return coarse, coarse_features


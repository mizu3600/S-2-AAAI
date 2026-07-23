from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class GraphIncrementalPlan:
    new_nodes: list[str]
    new_edges: int
    affected_nodes: list[str]
    requires_full_rebuild: bool


def plan_graph_incremental_update(
    old_graph: nx.Graph, new_graph: nx.Graph, node_threshold: float = 0.03,
    edge_threshold: float = 0.05, hops: int = 2,
) -> GraphIncrementalPlan:
    old_nodes = set(old_graph)
    new_nodes = sorted(set(new_graph) - old_nodes)
    old_edges = {frozenset(edge) for edge in old_graph.edges}
    current_edges = {frozenset(edge) for edge in new_graph.edges}
    added_edges = current_edges - old_edges
    changed_edges = set()
    for edge in old_edges & current_edges:
        left, right = tuple(edge)
        if old_graph.get_edge_data(left, right) != new_graph.get_edge_data(left, right):
            changed_edges.add(edge)
    updated_edges = added_edges | changed_edges
    seeds = set(new_nodes)
    for edge in updated_edges:
        seeds.update(edge)
    affected = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        frontier = {neighbor for node in frontier if node in new_graph for neighbor in new_graph.neighbors(node)} - affected
        affected.update(frontier)
    node_growth = len(new_nodes) / max(len(old_nodes), 1)
    edge_growth = len(updated_edges) / max(old_graph.number_of_edges(), 1)
    return GraphIncrementalPlan(
        new_nodes=new_nodes,
        new_edges=len(updated_edges),
        affected_nodes=sorted(affected),
        requires_full_rebuild=node_growth >= node_threshold or edge_growth >= edge_threshold,
    )

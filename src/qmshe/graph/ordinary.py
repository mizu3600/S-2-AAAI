from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import networkx as nx
import numpy as np
from scipy import sparse

from qmshe.ingest.schemas import Corpus
from qmshe.graph.ordinary_schemas import OrdinaryGraphEdge, OrdinaryGraphNode


class GraphProfile(str, Enum):
    REIFIED_FACT = "reified_fact"


@dataclass(frozen=True)
class OrdinaryGraphArtifacts:
    profile: GraphProfile
    graph: nx.Graph
    node_ids: list[str]
    node_texts: list[str]
    adjacency: sparse.csr_matrix
    propagation: sparse.csr_matrix
    fact_by_node: dict[str, str]
    facts_by_entity: dict[str, set[str]]
    nodes: list[OrdinaryGraphNode]
    edges: list[OrdinaryGraphEdge]


def _fact_text(fact, names: dict[str, str]) -> str:
    arguments = ", ".join(
        f"{argument.role}={names.get(argument.entity_id, argument.entity_id)}"
        for argument in fact.arguments
    )
    qualifiers = ", ".join(
        f"{key}={value}" for key, value in fact.qualifiers.items() if value is not None
    )
    return f"{fact.predicate}: {arguments}" + (f"; {qualifiers}" if qualifiers else "")


def normalized_propagation(adjacency: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    """Return D^-1/2 (A + I) D^-1/2 without densifying the graph."""
    adjacency = sparse.csr_matrix(adjacency, dtype=np.float32)
    augmented = adjacency + sparse.eye(adjacency.shape[0], format="csr", dtype=np.float32)
    degree = np.asarray(augmented.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    nonzero = degree > 0
    inverse[nonzero] = np.power(degree[nonzero], -0.5)
    scale = sparse.diags(inverse, format="csr")
    return (scale @ augmented @ scale).tocsr()


def build_ordinary_graph(
    corpus: Corpus, profile: GraphProfile | str = GraphProfile.REIFIED_FACT
) -> OrdinaryGraphArtifacts:
    """Build an ordinary graph independently from the evidence hypergraph operator.

    Entity-relation mode keeps entity nodes and evidence-bearing binary relations. Reified-fact
    mode represents every n-ary fact as a normal node and connects its arguments with role edges.
    In both cases, source fact IDs remain attached to edges for citation recovery.
    """
    profile = GraphProfile(profile)
    names = {entity.entity_id: entity.canonical_name for entity in corpus.entities}
    descriptions = {
        entity.entity_id: f"{entity.canonical_name}. {entity.description}".strip()
        for entity in corpus.entities
    }
    graph = nx.Graph(mode="ordinary_graph", profile=profile.value)
    facts_by_entity: dict[str, set[str]] = {entity_id: set() for entity_id in names}
    fact_by_node: dict[str, str] = {}

    for entity in corpus.entities:
        graph.add_node(entity.entity_id, kind="entity", text=descriptions[entity.entity_id])

    for fact in corpus.evidence_hyperedges:
        fact_id = fact.hyperedge_id
        fact_by_node[fact_id] = fact_id
        graph.add_node(
            fact_id,
            kind="fact",
            predicate=fact.predicate,
            text=_fact_text(fact, names),
            evidence_chunk_ids=list(fact.evidence_chunk_ids),
        )
        for argument in fact.arguments:
            if argument.entity_id not in graph:
                continue
            facts_by_entity.setdefault(argument.entity_id, set()).add(fact_id)
            graph.add_edge(
                argument.entity_id,
                fact_id,
                role=argument.role,
                relation=f"{argument.role.upper()}_OF",
                fact_id=fact_id,
                weight=float(fact.confidence),
            )

    node_ids = list(graph.nodes)
    node_texts = [str(graph.nodes[node_id].get("text", node_id)) for node_id in node_ids]
    adjacency = nx.to_scipy_sparse_array(
        graph, nodelist=node_ids, weight="weight", format="csr", dtype=np.float32
    )
    adjacency = sparse.csr_matrix(adjacency)
    nodes = [
        OrdinaryGraphNode(
            node_id=node_id, node_type=graph.nodes[node_id].get("kind", "entity"),
            text=str(graph.nodes[node_id].get("text", node_id)),
            source_fact_id=fact_by_node.get(node_id),
        )
        for node_id in node_ids
    ]
    edges = []
    for index, (left, right, attributes) in enumerate(graph.edges(data=True)):
        role_pairs = attributes.get("role_pairs", [])
        source_role, target_role = role_pairs[0] if role_pairs else (attributes.get("role"), None)
        fact_ids = attributes.get("fact_ids") or [attributes.get("fact_id")]
        edges.append(OrdinaryGraphEdge(
            edge_id=f"{profile.value}:edge:{index}", source_id=left, target_id=right,
            relation=attributes.get("relation") or "+".join(attributes.get("relations", [])),
            source_role=source_role, target_role=target_role,
            evidence_fact_ids=[item for item in fact_ids if item],
            weight=float(attributes.get("weight", 1.0)),
        ))
    return OrdinaryGraphArtifacts(
        profile=profile,
        graph=graph,
        node_ids=node_ids,
        node_texts=node_texts,
        adjacency=adjacency,
        propagation=normalized_propagation(adjacency),
        fact_by_node=fact_by_node,
        facts_by_entity=facts_by_entity,
        nodes=nodes,
        edges=edges,
    )

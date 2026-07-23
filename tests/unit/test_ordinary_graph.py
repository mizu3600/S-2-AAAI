import numpy as np
import torch

from qmshe.embedding.chebyshev import scipy_to_torch_sparse
from qmshe.embedding.graph_encoder import GraphSpectralSemanticEncoder
from qmshe.graph.ordinary import GraphProfile, build_ordinary_graph, normalized_propagation
from qmshe.graph.ordinary_incremental import plan_graph_incremental_update
from qmshe.synthetic import make_synthetic_corpus


def test_graph_profiles_are_independent_and_preserve_evidence():
    corpus = make_synthetic_corpus()
    entity_graph = build_ordinary_graph(corpus, GraphProfile.ENTITY_RELATION)
    reified_graph = build_ordinary_graph(corpus, GraphProfile.REIFIED_FACT)

    assert all(not node.startswith("fact_") for node in entity_graph.node_ids)
    assert any(node.startswith("fact_") for node in reified_graph.node_ids)
    assert all("fact_ids" in data for _, _, data in entity_graph.graph.edges(data=True))
    assert all("fact_id" in data and "role" in data for _, _, data in reified_graph.graph.edges(data=True))
    assert entity_graph.adjacency.shape[0] == len(corpus.entities)
    assert reified_graph.adjacency.shape[0] == len(corpus.entities) + len(corpus.evidence_hyperedges)
    assert all(edge.evidence_fact_ids for edge in entity_graph.edges)
    assert any(node.node_type == "fact" for node in reified_graph.nodes)


def test_normalized_graph_operator_and_bands_follow_design_formula():
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    propagation = normalized_propagation(adjacency)
    assert np.allclose(propagation.toarray(), propagation.toarray().T)

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    bands = GraphSpectralSemanticEncoder.raw_bands(x, scipy_to_torch_sparse(propagation))
    assert torch.allclose(bands["low"] + bands["mid"] + bands["high"], x, atol=1e-6)
    assert not torch.allclose(bands["low"], bands["high"])


def test_graph_incremental_plan_marks_local_neighborhood():
    corpus = make_synthetic_corpus()
    old = build_ordinary_graph(corpus, GraphProfile.REIFIED_FACT).graph
    new = old.copy()
    new.add_node("ent_new")
    new.add_edge("ent_new", next(iter(old.nodes)), weight=1.0)
    plan = plan_graph_incremental_update(old, new, node_threshold=1.0, edge_threshold=1.0)
    assert plan.new_nodes == ["ent_new"]
    assert "ent_new" in plan.affected_nodes
    assert plan.new_edges == 1
    assert not plan.requires_full_rebuild

    changed = old.copy()
    left, right = next(iter(changed.edges))
    changed[left][right]["weight"] += 0.1
    changed_plan = plan_graph_incremental_update(
        old, changed, node_threshold=1.0, edge_threshold=1.0
    )
    assert changed_plan.new_edges == 1
    assert {left, right}.issubset(changed_plan.affected_nodes)

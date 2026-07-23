import numpy as np
import torch

from qmshe.embedding.query_gate import QueryBandGate
from qmshe.graph.incremental import needs_full_rebuild
from qmshe.graph.semantic_graph import build_mutual_knn_graph


def test_gate_is_probability_distribution():
    gate = QueryBandGate(8, hidden_dim=4)
    output = gate(torch.randn(5, 8))
    assert output.shape == (5, 4)
    assert torch.allclose(output.sum(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.all(output >= 0)


def test_semantic_graph_is_mutual_symmetric_and_capped():
    vectors = np.array([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    graph = build_mutual_knn_graph(vectors, k=2, threshold=0.9, max_degree=1).adjacency
    assert np.allclose(graph.toarray(), graph.T.toarray())
    assert np.max(np.asarray((graph > 0).sum(axis=1)).ravel()) <= 1


def test_incremental_rebuild_threshold():
    assert not needs_full_rebuild(1000, 20, 1000, 20)
    assert needs_full_rebuild(1000, 31, 1000, 20)


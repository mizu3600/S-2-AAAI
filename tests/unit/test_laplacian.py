import numpy as np
import scipy.sparse as sp

from qmshe.graph.laplacian import (
    build_joint_spectral_laplacian,
    build_normalized_hypergraph_laplacian,
)


def test_sparse_zhou_matches_dense_formula():
    h_dense = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
    h = sp.csr_matrix(h_dense)
    weights = np.array([0.8, 0.6])
    actual = build_normalized_hypergraph_laplacian(h, weights).toarray()
    de = np.diag(h_dense.sum(axis=0))
    dv = np.diag(h_dense @ weights)
    inv_dv = np.diag(1 / np.sqrt(np.diag(dv)))
    expected = np.eye(3) - inv_dv @ h_dense @ np.diag(weights) @ np.linalg.inv(de) @ h_dense.T @ inv_dv
    assert np.max(np.abs(actual - expected)) < 1e-5
    assert np.allclose(actual, actual.T)


def test_empty_and_isolated_node_are_finite():
    h = sp.csr_matrix(np.array([[1], [1], [0]], dtype=float))
    laplacian = build_normalized_hypergraph_laplacian(h)
    assert sp.issparse(laplacian)
    assert np.isfinite(laplacian.data).all()
    assert laplacian[2, 2] == 1


def test_semantic_edges_enter_joint_operator_without_densifying():
    h = sp.csr_matrix(np.array([[1, 0], [1, 1], [0, 1]], dtype=float))
    semantic = sp.csr_matrix(np.array([[0, 0.9], [0.9, 0]], dtype=float))
    evidence_only = build_joint_spectral_laplacian(h)
    combined = build_joint_spectral_laplacian(h, semantic, semantic_weight=0.25)
    assert sp.issparse(combined)
    assert not np.allclose(evidence_only.toarray(), combined.toarray())
    assert np.allclose(combined.toarray(), combined.T.toarray())

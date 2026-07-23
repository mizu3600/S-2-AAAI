import numpy as np
import scipy.sparse as sp


def build_normalized_hypergraph_laplacian(
    incidence: sp.spmatrix, edge_weight: np.ndarray | None = None
) -> sp.csr_matrix:
    """Zhou normalized hypergraph Laplacian, kept sparse end to end."""
    h = incidence.astype(np.float64).tocsr()
    n, m = h.shape
    weights = np.ones(m, dtype=np.float64) if edge_weight is None else np.asarray(edge_weight, dtype=np.float64)
    if weights.shape != (m,):
        raise ValueError(f"edge_weight must have shape ({m},)")
    edge_degree = np.asarray(h.sum(axis=0)).ravel()
    vertex_degree = np.asarray(h @ weights).ravel()
    inv_edge = np.divide(1.0, edge_degree, out=np.zeros_like(edge_degree), where=edge_degree > 0)
    inv_sqrt_vertex = np.divide(
        1.0, np.sqrt(vertex_degree), out=np.zeros_like(vertex_degree), where=vertex_degree > 0
    )
    dv = sp.diags(inv_sqrt_vertex)
    middle = sp.diags(weights * inv_edge)
    propagation = dv @ h @ middle @ h.T @ dv
    return (sp.eye(n, format="csr", dtype=np.float64) - propagation).tocsr()


def build_joint_bipartite_laplacian(incidence: sp.spmatrix) -> sp.csr_matrix:
    return build_joint_spectral_laplacian(incidence)


def build_joint_spectral_laplacian(
    incidence: sp.spmatrix,
    semantic_fact_adjacency: sp.spmatrix | None = None,
    semantic_weight: float = 0.25,
) -> sp.csr_matrix:
    """Joint entity-fact operator with bounded retrieval-only fact similarity edges."""
    h = incidence.astype(np.float64).tocsr()
    zero_v = sp.csr_matrix((h.shape[0], h.shape[0]))
    if semantic_fact_adjacency is None:
        semantic = sp.csr_matrix((h.shape[1], h.shape[1]))
    else:
        semantic = semantic_fact_adjacency.astype(np.float64).tocsr()
        if semantic.shape != (h.shape[1], h.shape[1]):
            raise ValueError("semantic fact adjacency shape does not match hyperedge count")
        semantic = semantic.maximum(semantic.T) * semantic_weight
    adjacency = sp.bmat([[zero_v, h], [h.T, semantic]], format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    normalized = sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)
    return (sp.eye(adjacency.shape[0], format="csr") - normalized).tocsr()


def estimate_lmax(laplacian: sp.spmatrix) -> float:
    if laplacian.shape[0] <= 1:
        return 1.0
    value = float(sp.linalg.eigsh(laplacian, k=1, which="LM", return_eigenvectors=False)[0])
    return max(value, 1e-6)

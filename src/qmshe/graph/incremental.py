import numpy as np


def approximate_new_spectral_embedding(
    raw_vector: np.ndarray,
    neighbor_band_vectors: np.ndarray,
    neighbor_weights: np.ndarray | None = None,
) -> np.ndarray:
    if len(neighbor_band_vectors) == 0:
        return np.repeat(raw_vector[None, :], 3, axis=0)
    if neighbor_weights is None:
        neighbor_weights = np.ones(len(neighbor_band_vectors), dtype=np.float32)
    weights = neighbor_weights / np.maximum(neighbor_weights.sum(), 1e-12)
    return np.einsum("n,nbd->bd", weights, neighbor_band_vectors)


def needs_full_rebuild(
    old_nodes: int, new_nodes: int, old_edges: int, new_edges: int, node_threshold: float = 0.03,
    edge_threshold: float = 0.05,
) -> bool:
    node_growth = new_nodes / max(old_nodes, 1)
    edge_growth = new_edges / max(old_edges, 1)
    return node_growth >= node_threshold or edge_growth >= edge_threshold


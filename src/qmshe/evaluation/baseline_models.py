import random

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from torch import nn


def node2vec_embedding(
    graph: nx.Graph, dimensions: int = 32, walk_length: int = 20, walks_per_node: int = 10,
    window: int = 5, seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Deterministic random-walk co-occurrence Node2Vec baseline."""
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    rows, cols, values = [], [], []
    counts: dict[tuple[int, int], float] = {}
    for start in nodes:
        for _ in range(walks_per_node):
            walk = [start]
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(rng.choice(neighbors))
            for position, node in enumerate(walk):
                left, right = max(0, position - window), min(len(walk), position + window + 1)
                for context in walk[left:right]:
                    if context != node:
                        key = (index[node], index[context])
                        counts[key] = counts.get(key, 0.0) + 1.0
    for (row, col), value in counts.items():
        rows.append(row)
        cols.append(col)
        values.append(value)
    matrix = sp.csr_matrix((values, (rows, cols)), shape=(len(nodes), len(nodes)))
    if len(nodes) <= 2:
        return nodes, np.eye(len(nodes), dtype=np.float32)
    size = min(dimensions, len(nodes) - 1)
    return nodes, TruncatedSVD(size, random_state=seed).fit_transform(matrix).astype(np.float32)


class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.sparse.mm(normalized_adjacency, self.linear(x)))


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, mean_adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.sparse.mm(mean_adjacency, x)
        return torch.relu(self.linear(torch.cat([x, neighbors], dim=-1)))


class HypergraphConvEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, propagation: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.sparse.mm(propagation, self.linear(x)))


def semantic_lappe_concat(semantic: np.ndarray, lap_pe: np.ndarray) -> np.ndarray:
    return np.concatenate([semantic, lap_pe], axis=-1)

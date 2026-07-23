import torch
from torch import nn


class QueryBandGate(nn.Module):
    def __init__(self, query_dim: int, num_bands: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_bands)
        )

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.mlp(query_embedding), dim=-1)


class QueryRelationGate(nn.Module):
    def __init__(self, query_dim: int, num_relations: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_relations)
        )

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.mlp(query_embedding), dim=-1)


def build_seed_weights(
    query_embedding: torch.Tensor,
    raw_embeddings: torch.Tensor,
    top_m: int = 64,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.nn.functional.normalize(query_embedding, dim=-1)
    nodes = torch.nn.functional.normalize(raw_embeddings, dim=-1)
    scores = nodes @ query
    count = min(top_m, len(scores))
    values, indices = torch.topk(scores, k=count)
    return indices, torch.softmax(values / temperature, dim=0)


def pool_seed_bands(
    indices: torch.Tensor, weights: torch.Tensor, band_embeddings: list[torch.Tensor]
) -> list[torch.Tensor]:
    return [(band[indices] * weights[:, None]).sum(dim=0) for band in band_embeddings]

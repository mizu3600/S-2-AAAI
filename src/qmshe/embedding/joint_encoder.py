import torch
from torch import nn

from qmshe.embedding.filter_bank import ChebyshevFilterBank
from qmshe.embedding.query_gate import QueryBandGate, build_seed_weights, pool_seed_bands


class JointSpectralSemanticEncoder(nn.Module):
    def __init__(
        self, input_dim: int, raw_dim: int = 256, band_dim: int = 128, order: int = 5,
        gate_hidden_dim: int = 256,
    ):
        super().__init__()
        self.raw_projection = nn.Linear(input_dim, raw_dim)
        self.filter_bank = ChebyshevFilterBank(input_dim, band_dim, order=order)
        self.query_gate = QueryBandGate(input_dim, num_bands=4, hidden_dim=gate_hidden_dim)
        self.query_raw_projection = nn.Linear(input_dim, raw_dim)
        self.query_band_projections = nn.ModuleList(
            [nn.Linear(band_dim, band_dim) for _ in range(3)]
        )

    @property
    def output_dim(self) -> int:
        return self.raw_projection.out_features + sum(p.out_features for p in self.query_band_projections)

    def encode_nodes(self, x: torch.Tensor, laplacian: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.raw_projection(x)
        low, mid, high = self.filter_bank(x, laplacian)
        full = torch.cat([raw, low, mid, high], dim=-1)
        return {"raw": raw, "low": low, "mid": mid, "high": high, "full": full}

    def encode_query(
        self, query: torch.Tensor, raw_node_features: torch.Tensor,
        node_bands: dict[str, torch.Tensor], top_m: int = 64, temperature: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 1:
            raise ValueError("encode_query expects one query vector")
        indices, weights = build_seed_weights(query, raw_node_features, top_m, temperature)
        pooled = pool_seed_bands(indices, weights, [node_bands["low"], node_bands["mid"], node_bands["high"]])
        gate = self.query_gate(query)
        pieces = [gate[0] * self.query_raw_projection(query)]
        pieces.extend(gate[i + 1] * projection(vector) for i, (projection, vector) in enumerate(zip(self.query_band_projections, pooled, strict=True)))
        return torch.cat(pieces), gate


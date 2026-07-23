from __future__ import annotations

import torch
from torch import nn

from qmshe.embedding.query_gate import QueryBandGate, build_seed_weights, pool_seed_bands


class GraphSpectralSemanticEncoder(nn.Module):
    """QMSxE ordinary-graph encoder using explicit raw/low/mid/high graph bands."""

    def __init__(
        self, input_dim: int, raw_dim: int = 64, band_dim: int = 32,
        gate_hidden_dim: int = 128,
    ):
        super().__init__()
        self.raw_projection = nn.Linear(input_dim, raw_dim)
        self.band_projections = nn.ModuleDict({
            name: nn.Linear(input_dim, band_dim) for name in ("low", "mid", "high")
        })
        self.query_gate = QueryBandGate(input_dim, num_bands=4, hidden_dim=gate_hidden_dim)
        self.query_raw_projection = nn.Linear(input_dim, raw_dim)
        self.query_band_projections = nn.ModuleDict({
            name: nn.Linear(band_dim, band_dim) for name in ("low", "mid", "high")
        })

    @property
    def output_dim(self) -> int:
        return self.raw_projection.out_features + 3 * next(iter(self.band_projections.values())).out_features

    @staticmethod
    def raw_bands(x: torch.Tensor, propagation: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = torch.sparse.mm(propagation, x)
        z2 = torch.sparse.mm(propagation, z1)
        return {"raw": x, "low": z2, "mid": z1 - z2, "high": x - z1}

    def encode_nodes(self, x: torch.Tensor, propagation: torch.Tensor) -> dict[str, torch.Tensor]:
        source = self.raw_bands(x, propagation)
        projected = {"raw": self.raw_projection(source["raw"])}
        projected.update({name: self.band_projections[name](source[name]) for name in ("low", "mid", "high")})
        projected["full"] = torch.cat(
            [projected[name] for name in ("raw", "low", "mid", "high")], dim=-1
        )
        return projected

    def encode_query(
        self, query: torch.Tensor, raw_node_features: torch.Tensor,
        node_bands: dict[str, torch.Tensor], top_m: int = 64, temperature: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts, gate = self.encode_query_parts(
            query, raw_node_features, node_bands, top_m=top_m, temperature=temperature
        )
        return torch.cat([
            gate[index] * parts[name]
            for index, name in enumerate(("raw", "low", "mid", "high"))
        ]), gate

    def encode_query_parts(
        self, query: torch.Tensor, raw_node_features: torch.Tensor,
        node_bands: dict[str, torch.Tensor], top_m: int = 64, temperature: float = 0.05,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        indices, weights = build_seed_weights(query, raw_node_features, top_m, temperature)
        pooled = pool_seed_bands(
            indices, weights, [node_bands[name] for name in ("low", "mid", "high")]
        )
        gate = self.query_gate(query)
        parts = {"raw": self.query_raw_projection(query)}
        parts.update({
            name: self.query_band_projections[name](pooled[index])
            for index, name in enumerate(("low", "mid", "high"))
        })
        return parts, gate

from __future__ import annotations

import torch

from s2rag.embedding.query_gate import (
    analytic_band_weights,
    build_seed_weights,
    pool_seed_bands,
)


class GraphSpectralSemanticEncoder:
    """Deterministic raw/low/mid/high graph-band encoder without learned weights."""

    def __init__(
        self,
        input_dim: int,
        raw_dim: int = 64,
        band_dim: int = 32,
        gate_hidden_dim: int = 128,
    ):
        self.input_dim = input_dim

    def eval(self) -> GraphSpectralSemanticEncoder:
        """Compatibility no-op: this analytic encoder has no train/eval state."""
        return self

    def parameters(self):
        """Expose an empty iterator for callers auditing trainable state."""
        return iter(())

    @property
    def output_dim(self) -> int:
        return self.input_dim * 4

    @staticmethod
    def raw_bands(x: torch.Tensor, propagation: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = torch.sparse.mm(propagation, x)
        z2 = torch.sparse.mm(propagation, z1)
        return {"raw": x, "low": z2, "mid": z1 - z2, "high": x - z1}

    def encode_nodes(self, x: torch.Tensor, propagation: torch.Tensor) -> dict[str, torch.Tensor]:
        bands = self.raw_bands(x, propagation)
        bands["full"] = torch.cat([bands[name] for name in ("raw", "low", "mid", "high")], dim=-1)
        return bands

    def encode_query(
        self,
        query: torch.Tensor,
        raw_node_features: torch.Tensor,
        node_bands: dict[str, torch.Tensor],
        top_m: int = 64,
        temperature: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts, gate = self.encode_query_parts(
            query, raw_node_features, node_bands, top_m=top_m, temperature=temperature
        )
        return torch.cat(
            [gate[index] * parts[name] for index, name in enumerate(("raw", "low", "mid", "high"))]
        ), gate

    def encode_query_parts(
        self,
        query: torch.Tensor,
        raw_node_features: torch.Tensor,
        node_bands: dict[str, torch.Tensor],
        top_m: int = 64,
        temperature: float = 0.05,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        indices, weights = build_seed_weights(query, raw_node_features, top_m, temperature)
        pooled = pool_seed_bands(
            indices,
            weights,
            [node_bands[name] for name in ("raw", "low", "mid", "high")],
        )
        gate = analytic_band_weights(query, pooled)
        parts = {"raw": query}
        parts.update(
            {name: pooled[index] for index, name in enumerate(("low", "mid", "high"), start=1)}
        )
        return parts, gate

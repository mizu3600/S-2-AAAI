import math

import torch
from torch import nn

from qmshe.embedding.chebyshev import chebyshev_terms, scale_laplacian


def _initial_coefficients(order: int) -> torch.Tensor:
    """Distinct stable polynomial initializers for low/mid/high graph frequencies."""
    coefficients = torch.zeros(3, order + 1)
    # Low: repeated smoothing; high: residual; mid: difference of two scales.
    coefficients[0, 0] = 0.5
    if order >= 1:
        coefficients[0, 1] = -0.5
        coefficients[2, 0] = 0.5
        coefficients[2, 1] = 0.5
    if order >= 2:
        coefficients[1, 0] = 0.5
        coefficients[1, 2] = -0.5
    else:
        coefficients[1, 0] = 1.0
    return coefficients


class ChebyshevFilterBank(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, order: int = 5, num_bands: int = 3,
        learnable_coefficients: bool = True,
    ):
        super().__init__()
        if num_bands != 3:
            raise ValueError("MVP defines exactly low, mid and high bands")
        self.order = order
        initial = _initial_coefficients(order)
        self.coefficients = nn.Parameter(initial, requires_grad=learnable_coefficients)
        self.projections = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_bands)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in self.projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor, lmax: float = 2.0) -> list[torch.Tensor]:
        scaled = scale_laplacian(laplacian, lmax)
        terms = chebyshev_terms(x, scaled, self.order)
        outputs = []
        for band, projection in enumerate(self.projections):
            filtered = sum(self.coefficients[band, k] * terms[k] for k in range(self.order + 1))
            outputs.append(projection(filtered) / math.sqrt(max(1, x.shape[-1])))
        return outputs


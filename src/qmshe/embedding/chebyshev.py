import numpy as np
import scipy.sparse as sp
import torch


def scipy_to_torch_sparse(matrix: sp.spmatrix, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=dtype)
    return torch.sparse_coo_tensor(
        indices, values, coo.shape, dtype=dtype, check_invariants=True
    ).coalesce()


def scale_laplacian(laplacian: torch.Tensor, lmax: float = 2.0) -> torch.Tensor:
    laplacian = laplacian.coalesce()
    n = laplacian.shape[0]
    identity_indices = torch.arange(n, device=laplacian.device).repeat(2, 1)
    indices = torch.cat([laplacian.indices(), identity_indices], dim=1)
    values = torch.cat(
        [2.0 * laplacian.values() / max(lmax, 1e-6), -torch.ones(n, device=laplacian.device)]
    )
    return torch.sparse_coo_tensor(
        indices, values, laplacian.shape, check_invariants=True
    ).coalesce()


def chebyshev_terms(
    x: torch.Tensor, scaled_laplacian: torch.Tensor, order: int
) -> list[torch.Tensor]:
    if order < 0:
        raise ValueError("order must be non-negative")
    terms = [x]
    if order == 0:
        return terms
    terms.append(torch.sparse.mm(scaled_laplacian, x))
    for _ in range(2, order + 1):
        terms.append(2 * torch.sparse.mm(scaled_laplacian, terms[-1]) - terms[-2])
    return terms


def apply_chebyshev(terms: list[torch.Tensor], coefficients: torch.Tensor) -> torch.Tensor:
    if len(terms) != coefficients.shape[-1]:
        raise ValueError("coefficient count must match terms")
    return sum(coefficient * term for coefficient, term in zip(coefficients, terms, strict=True))

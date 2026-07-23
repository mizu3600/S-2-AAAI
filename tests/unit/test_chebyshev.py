import numpy as np
import scipy.sparse as sp
import torch

from qmshe.embedding.chebyshev import apply_chebyshev, chebyshev_terms, scale_laplacian, scipy_to_torch_sparse


def test_chebyshev_polynomial_matches_direct_eigendecomposition():
    lap = np.array([[1, -1], [-1, 1]], dtype=np.float32)
    x = np.array([[1.0], [0.25]], dtype=np.float32)
    coefficients = np.array([0.4, -0.3, 0.2, 0.1], dtype=np.float32)
    sparse = scipy_to_torch_sparse(sp.csr_matrix(lap))
    scaled = scale_laplacian(sparse, lmax=2.0)
    actual = apply_chebyshev(
        chebyshev_terms(torch.tensor(x), scaled, order=3), torch.tensor(coefficients)
    ).numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(lap)
    scaled_values = eigenvalues - 1.0
    t0 = np.ones_like(scaled_values)
    t1 = scaled_values
    t2 = 2 * scaled_values * t1 - t0
    t3 = 2 * scaled_values * t2 - t1
    response = coefficients[0] * t0 + coefficients[1] * t1 + coefficients[2] * t2 + coefficients[3] * t3
    expected = eigenvectors @ np.diag(response) @ eigenvectors.T @ x
    assert np.max(np.abs(actual - expected)) < 1e-5


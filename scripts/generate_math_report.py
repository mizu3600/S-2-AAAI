import json
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import torch
import typer

from qmshe.embedding.chebyshev import (
    apply_chebyshev,
    chebyshev_terms,
    scale_laplacian,
    scipy_to_torch_sparse,
)
from qmshe.graph.laplacian import build_normalized_hypergraph_laplacian


def main(output_dir: Path = typer.Option(Path("reports/math"))) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float64)
    weights = np.array([0.8, 0.6])
    sparse_laplacian = build_normalized_hypergraph_laplacian(
        sparse.csr_matrix(h), weights
    ).toarray()
    edge_degree = np.diag(h.sum(axis=0))
    vertex_degree = np.diag(h @ weights)
    inverse_vertex = np.diag(1 / np.sqrt(np.diag(vertex_degree)))
    dense_laplacian = (
        np.eye(3) - inverse_vertex @ h @ np.diag(weights)
        @ np.linalg.inv(edge_degree) @ h.T @ inverse_vertex
    )
    laplacian_error = float(np.max(np.abs(sparse_laplacian - dense_laplacian)))

    laplacian = np.array([[1, -1], [-1, 1]], dtype=np.float32)
    signal = np.array([[1.0], [0.25]], dtype=np.float32)
    coefficients = np.array([0.4, -0.3, 0.2, 0.1], dtype=np.float32)
    scaled = scale_laplacian(
        scipy_to_torch_sparse(sparse.csr_matrix(laplacian)), lmax=2.0
    )
    actual = apply_chebyshev(
        chebyshev_terms(torch.tensor(signal), scaled, order=3),
        torch.tensor(coefficients),
    ).numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    scaled_values = eigenvalues - 1.0
    terms = [np.ones_like(scaled_values), scaled_values]
    terms.extend(2 * scaled_values * terms[-1] - terms[-2] for _ in range(2, 4))
    response = sum(coefficient * term for coefficient, term in zip(coefficients, terms, strict=True))
    expected = eigenvectors @ np.diag(response) @ eigenvectors.T @ signal
    chebyshev_error = float(np.max(np.abs(actual - expected)))

    lambdas = np.linspace(0, 2, 81)
    ordinary_responses = {
        "raw": np.ones_like(lambdas),
        "low": (1 - lambdas) ** 2,
        "mid": (1 - lambdas) - (1 - lambdas) ** 2,
        "high": lambdas,
    }
    _write_line_svg(output_dir / "ordinary_frequency_response.svg", lambdas, ordinary_responses)
    metrics = {
        "zhou_sparse_dense_max_abs_error": laplacian_error,
        "chebyshev_eigendecomposition_max_abs_error": chebyshev_error,
        "threshold": 1e-5,
        "zhou_pass": laplacian_error < 1e-5,
        "chebyshev_pass": chebyshev_error < 1e-5,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "spectral_validation.md").write_text(
        "\n".join([
            "# Spectral correctness report", "",
            "| Check | Maximum absolute error | Threshold | Result |",
            "|---|---:|---:|---|",
            f"| Sparse Zhou vs dense formula | {laplacian_error:.3e} | 1e-5 | {'PASS' if metrics['zhou_pass'] else 'FAIL'} |",
            f"| Chebyshev vs eigendecomposition | {chebyshev_error:.3e} | 1e-5 | {'PASS' if metrics['chebyshev_pass'] else 'FAIL'} |",
            "", "The ordinary-graph response plot is derived directly from `S=I-L`, "
            "`low=S²X`, `mid=SX-S²X`, and `high=X-SX`.", "",
            "![Ordinary graph frequency response](ordinary_frequency_response.svg)", "",
        ]), encoding="utf-8",
    )
    typer.echo(json.dumps(metrics, indent=2))


def _write_line_svg(path: Path, x: np.ndarray, series: dict[str, np.ndarray]) -> None:
    width, height, left, top, plot_width, plot_height = 760, 420, 70, 30, 650, 320
    values = np.concatenate(list(series.values()))
    y_min, y_max = min(float(values.min()), -0.3), max(float(values.max()), 2.0)
    colors = {"raw": "#666666", "low": "#2f6fed", "mid": "#ef8a17", "high": "#c33c54"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#555"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#555"/>',
        f'<text x="{left + plot_width / 2}" y="395" text-anchor="middle" font-family="sans-serif">Laplacian eigenvalue λ</text>',
        '<text x="18" y="190" transform="rotate(-90 18 190)" text-anchor="middle" font-family="sans-serif">Spectral response</text>',
    ]
    for tick in np.linspace(0, 2, 5):
        px = left + tick / 2 * plot_width
        lines.append(f'<text x="{px:.1f}" y="370" text-anchor="middle" font-family="sans-serif" font-size="12">{tick:.1f}</text>')
    for index, (name, y) in enumerate(series.items()):
        points = []
        for x_value, y_value in zip(x, y, strict=True):
            px = left + x_value / 2 * plot_width
            py = top + (y_max - float(y_value)) / (y_max - y_min) * plot_height
            points.append(f"{px:.2f},{py:.2f}")
        lines.append(
            f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2.5" points="{" ".join(points)}"/>'
        )
        lines.append(
            f'<text x="{left + 15 + index * 100}" y="20" fill="{colors[name]}" font-family="sans-serif" font-size="13">{name}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    typer.run(main)

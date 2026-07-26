from __future__ import annotations

import math
import random
from statistics import mean, stdev


def summarize(values: list[float], bootstrap_samples: int = 1000, seed: int = 42) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    bootstrap = [
        mean(rng.choice(values) for _ in values)
        for _ in range(bootstrap_samples)
    ]
    bootstrap.sort()
    low = bootstrap[max(0, int(0.025 * len(bootstrap)) - 1)]
    high = bootstrap[min(len(bootstrap) - 1, int(0.975 * len(bootstrap)))]
    return {
        "n": len(values),
        "mean": mean(values),
        "sd": stdev(values) if len(values) > 1 else None,
        "ci95_low": low,
        "ci95_high": high,
    }


def paired_randomization_pvalue(
    left: list[float], right: list[float], permutations: int = 5000, seed: int = 42
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must have equal non-zero length")
    observed = abs(mean(a - b for a, b in zip(left, right, strict=True)))
    rng = random.Random(seed)
    exceed = 0
    for _ in range(permutations):
        swapped = [
            (b - a) if rng.getrandbits(1) else (a - b)
            for a, b in zip(left, right, strict=True)
        ]
        exceed += int(abs(mean(swapped)) >= observed)
    return (exceed + 1) / (permutations + 1)


def cohens_d_paired(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    differences = [a - b for a, b in zip(left, right, strict=True)]
    deviation = stdev(differences)
    if deviation > 0:
        return mean(differences) / deviation
    return 0.0 if mean(differences) == 0 else math.nan


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Family-wise error correction that preserves the original comparison keys."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[key] = running
    return adjusted

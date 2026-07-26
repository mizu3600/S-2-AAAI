from __future__ import annotations

import json
from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.experiment import BenchmarkExperimentRunner


def main(
    manifest_path: Path = typer.Option(
        Path("data/benchmarks/configured_suites/manifest.json")
    ),
    output_dir: Path = typer.Option(Path("data/experiments/configured")),
    methods: str = typer.Option("bm25,dense,bm25_dense_rrf,reified_fact_hybrid"),
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from qmshe.evaluation.internal_baselines import INTERNAL_BASELINES

    selected = INTERNAL_BASELINES if methods == "all" else tuple(
        item.strip() for item in methods.split(",") if item.strip()
    )
    for suite_name, spec in manifest.get("suites", {}).items():
        suite = load_benchmark(
            spec["dataset"],
            spec["path"],
            split=spec.get("split", "validation"),
            limit=spec.get("examples"),
        )
        target = output_dir / suite_name
        BenchmarkExperimentRunner(methods=selected).run(
            suite, target, seed=int(spec["seed"])
        )
        typer.echo(f"{suite_name}: {len(suite.examples)} examples -> {target}")


if __name__ == "__main__":
    typer.run(main)

from __future__ import annotations

import json
from pathlib import Path

import typer

from s2rag.benchmarks import load_benchmark
from s2rag.evaluation.experiment import BenchmarkExperimentRunner


def main(
    manifest_path: Path = typer.Option(
        Path("data/benchmarks/configured_suites/manifest.json")
    ),
    output_dir: Path = typer.Option(Path("data/experiments/configured")),
    methods: str = typer.Option("bm25,dense,reified_fact_hybrid"),
    generate_methods: str = typer.Option("bm25,dense,reified_fact_hybrid"),
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from s2rag.evaluation.internal_baselines import (
        ALL_EXPERIMENT_METHODS,
        BENCHMARK_METHODS,
    )

    selected = (
        ALL_EXPERIMENT_METHODS
        if methods == "all_ablations"
        else BENCHMARK_METHODS
        if methods == "all"
        else tuple(item.strip() for item in methods.split(",") if item.strip())
    )
    generated = tuple(
        item.strip()
        for item in generate_methods.split(",")
        if item.strip() in selected
    )
    for suite_name, spec in manifest.get("suites", {}).items():
        suite = load_benchmark(
            spec["dataset"],
            spec["path"],
            split=spec.get("split", "validation"),
            limit=spec.get("examples"),
        )
        target = output_dir / suite_name
        BenchmarkExperimentRunner(
            methods=selected,
            generate_for_methods=generated,
        ).run(
            suite, target, seed=int(spec["seed"])
        )
        typer.echo(f"{suite_name}: {len(suite.examples)} examples -> {target}")


if __name__ == "__main__":
    typer.run(main)

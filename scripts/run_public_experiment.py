from pathlib import Path

import typer

from s2rag.benchmarks import load_benchmark
from s2rag.evaluation.experiment import BenchmarkExperimentRunner


def main(
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_1000.jsonl")),
    output_dir: Path = typer.Option(Path("data/experiments/public")),
    limit: int = typer.Option(0, help="0 means all 1,000 prepared examples"),
    split: str = typer.Option("validation"),
    seed: int = typer.Option(42),
    methods: str = typer.Option(
        "bm25,dense,reified_fact_hybrid",
        help="Comma-separated methods, or 'all' for every selected baseline",
    ),
    generate_methods: str = typer.Option(
        "bm25,dense,reified_fact_hybrid",
        help="Methods that receive answer generation; retrieval-only ablations are omitted",
    ),
) -> None:
    suite = load_benchmark(
        dataset,
        input_path,
        split=split,
        limit=None if limit == 0 else limit,
    )
    from s2rag.evaluation.internal_baselines import (
        ALL_EXPERIMENT_METHODS,
        BENCHMARK_METHODS,
    )

    selected = (
        ALL_EXPERIMENT_METHODS
        if methods == "all_ablations"
        else BENCHMARK_METHODS
        if methods == "all"
        else tuple(
        item.strip() for item in methods.split(",") if item.strip()
        )
    )
    generated = tuple(
        item.strip()
        for item in generate_methods.split(",")
        if item.strip() in selected
    )
    records = BenchmarkExperimentRunner(
        methods=selected,
        generate_for_methods=generated,
    ).run(
        suite, output_dir / dataset, seed=seed
    )
    typer.echo(f"wrote {len(records)} records to {output_dir / dataset}")


if __name__ == "__main__":
    typer.run(main)

from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.experiment import BenchmarkExperimentRunner


def main(
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_sample.json")),
    output_dir: Path = typer.Option(Path("data/experiments/public")),
    limit: int = typer.Option(20),
    split: str = typer.Option("validation"),
    seed: int = typer.Option(42),
    methods: str = typer.Option(
        "bm25,dense,bm25_dense_rrf,reified_fact_hybrid",
        help="Comma-separated methods, or 'all' for every internal baseline",
    ),
) -> None:
    suite = load_benchmark(dataset, input_path, split=split, limit=limit)
    from qmshe.evaluation.internal_baselines import INTERNAL_BASELINES

    selected = INTERNAL_BASELINES if methods == "all" else tuple(
        item.strip() for item in methods.split(",") if item.strip()
    )
    records = BenchmarkExperimentRunner(methods=selected).run(
        suite, output_dir / dataset, seed=seed
    )
    typer.echo(f"wrote {len(records)} records to {output_dir / dataset}")


if __name__ == "__main__":
    typer.run(main)

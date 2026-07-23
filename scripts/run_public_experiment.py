from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.experiment import BenchmarkExperimentRunner


def main(
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_sample.json")),
    output_dir: Path = typer.Option(Path("data/experiments/public")),
    limit: int = typer.Option(20),
    track_mlflow: bool = typer.Option(False),
) -> None:
    suite = load_benchmark(dataset, input_path, limit=limit)
    records = BenchmarkExperimentRunner(track_mlflow=track_mlflow).run(suite, output_dir / dataset)
    typer.echo(f"wrote {len(records)} records to {output_dir / dataset}")


if __name__ == "__main__":
    typer.run(main)

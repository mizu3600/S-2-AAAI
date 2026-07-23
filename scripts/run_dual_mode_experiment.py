from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.dual_mode import DualModeExperimentRunner


def main(
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_sample.json")),
    output_dir: Path = typer.Option(Path("data/experiments/dual_mode")),
    limit: int = typer.Option(20),
) -> None:
    suite = load_benchmark(dataset, input_path, limit=limit)
    records = DualModeExperimentRunner().run(suite, output_dir / dataset)
    typer.echo(f"wrote {len(records)} graph/hypergraph records to {output_dir / dataset}")


if __name__ == "__main__":
    typer.run(main)

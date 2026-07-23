import json
from pathlib import Path

import typer

from qmshe.evaluation.ablations import run_runtime_ablations
from qmshe.pipeline import QMSHEPipeline, load_corpus


def main(
    corpus_path: Path = typer.Option(Path("data/processed/synthetic.json")),
    output: Path = typer.Option(Path("data/experiments/ablations.json")),
) -> None:
    pipeline = QMSHEPipeline(load_corpus(corpus_path))
    pipeline.generator.client = None
    questions = [("How does PEAI improve Voc?", {"fact_1", "fact_2", "fact_3"})]
    typer.echo(json.dumps(run_runtime_ablations(pipeline, questions, output), indent=2))


if __name__ == "__main__":
    typer.run(main)

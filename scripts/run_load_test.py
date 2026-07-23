import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import typer

from qmshe.pipeline import QMSHEPipeline, load_corpus
from qmshe.scaling.load_test import run_load_test


def main(
    corpus_path: Path = typer.Option(Path("data/processed/synthetic.json")),
    requests: int = typer.Option(100), concurrency: int = typer.Option(8),
) -> None:
    pipeline = QMSHEPipeline(load_corpus(corpus_path))
    pipeline.generator.client = None
    questions = ["How does PEAI improve Voc?", "What reduces recombination?", "What affects stability?"]
    cold_questions = [f"How does PEAI improve Voc? request-{index}" for index in range(requests)]
    cold = asyncio.run(run_load_test(pipeline, cold_questions, requests, concurrency))
    warm = asyncio.run(run_load_test(pipeline, questions, requests, concurrency))
    typer.echo(json.dumps({"cold_cache": asdict(cold), "warm_cache": asdict(warm)}, indent=2))


if __name__ == "__main__":
    typer.run(main)

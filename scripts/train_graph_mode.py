import json
from pathlib import Path

import torch
import typer

from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_suite_corpus
from qmshe.evaluation.experiment import LocalBenchmarkEncoder
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline


def main(
    dataset: str = typer.Option("hotpotqa"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_sample.json")),
    output: Path = typer.Option(Path("data/models/graph_stage_a.pt")),
    profile: GraphProfile = typer.Option(GraphProfile.REIFIED_FACT),
    limit: int = typer.Option(20), epochs: int = typer.Option(10),
) -> None:
    suite = load_benchmark(dataset, input_path, split="train", limit=limit)
    built = build_suite_corpus(suite.examples)
    pipeline = QMSGEGraphPipeline(
        built.corpus, text_encoder=LocalBenchmarkEncoder(), profile=profile
    )
    pipeline.generator.client = None
    history = pipeline.train_stage_a(built.training_pairs, epochs=epochs)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": pipeline.model.state_dict(), "profile": profile.value,
        "history": history, "graph_version": built.corpus.graph_version, "dataset": dataset,
    }, output)
    output.with_suffix(".json").write_text(
        json.dumps({"profile": profile.value, "history": history}, indent=2), encoding="utf-8"
    )
    typer.echo(f"saved {profile.value} checkpoint to {output}; loss {history[0]:.4f} -> {history[-1]:.4f}")


if __name__ == "__main__":
    typer.run(main)

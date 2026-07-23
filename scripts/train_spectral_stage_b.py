import json
from dataclasses import asdict
from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.local_models import LocalBGEEncoder
from qmshe.evaluation.splits import fixed_partition
from qmshe.graph.ordinary import GraphProfile
from qmshe.training.spectral_stage_b import SpectralStageBConfig, train_spectral_stage_b


def main(
    base_model: Path = typer.Option(...), stage_a_checkpoint: Path = typer.Option(...),
    mode: str = typer.Option(...), profile: str = typer.Option("evidence_hypergraph"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output_dir: Path = typer.Option(Path("data/models/spectral_stage_b")),
    dataset: str = typer.Option("hotpotqa"), limit: int = typer.Option(500),
    epochs: int = typer.Option(3), learning_rate: float = typer.Option(2e-5),
    seed: int = typer.Option(42), device: str = typer.Option("cuda"),
    embedding_batch_size: int = typer.Option(32), max_memory_fraction: float = typer.Option(0.45),
) -> None:
    if mode == "graph":
        GraphProfile(profile)
    elif mode != "hypergraph":
        raise typer.BadParameter("mode must be graph or hypergraph")
    suite = load_benchmark(dataset, input_path, split="train", limit=limit)
    partitions = fixed_partition(suite.examples)
    document_encoder = LocalBGEEncoder(
        str(base_model), batch_size=embedding_batch_size, device=device
    )
    report = train_spectral_stage_b(
        partitions["train"], partitions["validation"], document_encoder,
        SpectralStageBConfig(
            base_model=str(base_model), stage_a_checkpoint=str(stage_a_checkpoint),
            output_dir=str(output_dir), mode=mode, profile=profile, epochs=epochs,
            learning_rate=learning_rate, seed=seed, device=device,
            max_memory_fraction=max_memory_fraction,
        ),
    )
    typer.echo(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    typer.run(main)

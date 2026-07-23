import json
from dataclasses import asdict
from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.local_models import LocalBGEEncoder
from qmshe.evaluation.splits import fixed_partition
from qmshe.graph.ordinary import GraphProfile
from qmshe.training.inductive_stage_a import InductiveStageAConfig, train_inductive_stage_a


def main(
    base_model: Path = typer.Option(...),
    mode: str = typer.Option(...),
    profile: str = typer.Option("evidence_hypergraph"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output: Path = typer.Option(Path("data/models/stage_a.pt")),
    dataset: str = typer.Option("hotpotqa"),
    limit: int = typer.Option(500),
    epochs: int = typer.Option(3),
    learning_rate: float = typer.Option(2e-4),
    seed: int = typer.Option(42),
    device: str = typer.Option("cuda"),
    embedding_batch_size: int = typer.Option(32),
    variant: str = typer.Option("full"),
) -> None:
    if mode == "graph":
        GraphProfile(profile)
    elif mode != "hypergraph":
        raise typer.BadParameter("mode must be graph or hypergraph")
    suite = load_benchmark(dataset, input_path, split="train", limit=limit)
    partitions = fixed_partition(suite.examples)
    encoder = LocalBGEEncoder(str(base_model), batch_size=embedding_batch_size, device=device)
    report = train_inductive_stage_a(
        partitions["train"],
        partitions["validation"],
        encoder,
        InductiveStageAConfig(
            mode=mode,
            profile=profile,
            output_path=str(output),
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            variant=variant,
            bridge_loss_weight=0.0 if variant == "no_bridge_loss" else 0.5,
        ),
    )
    typer.echo(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    typer.run(main)

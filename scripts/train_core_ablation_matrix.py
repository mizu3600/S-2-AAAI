from __future__ import annotations

import gc
from pathlib import Path

import torch
import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.local_models import LocalBGEEncoder
from qmshe.evaluation.splits import fixed_partition
from qmshe.training.inductive_stage_a import (
    InductiveStageAConfig,
    train_inductive_stage_a,
)


RETRAIN_VARIANTS = (
    "raw_only",
    "no_low",
    "no_mid",
    "no_high",
    "fixed_gate",
    "no_role_gate",
    "no_semantic_graph",
    "no_bridge_loss",
    "no_hard_negatives",
)


def main(
    base_model: Path = typer.Option(...),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output_root: Path = typer.Option(Path("data/models/core_ablation")),
    dataset: str = typer.Option("hotpotqa"),
    limit: int = typer.Option(500),
    epochs: int = typer.Option(3),
    seeds: str = typer.Option("13,42,73"),
    variants: str = typer.Option(",".join(RETRAIN_VARIANTS)),
    device: str = typer.Option("cuda"),
    embedding_batch_size: int = typer.Option(32),
    force: bool = typer.Option(False),
) -> None:
    selected_variants = [item.strip() for item in variants.split(",") if item.strip()]
    unknown = sorted(set(selected_variants) - set(RETRAIN_VARIANTS))
    if unknown:
        raise typer.BadParameter(f"unknown variants: {unknown}")
    suite = load_benchmark(dataset, input_path, split="train", limit=limit)
    partitions = fixed_partition(suite.examples)
    for seed in (int(item) for item in seeds.split(",") if item.strip()):
        encoder = LocalBGEEncoder(str(base_model), batch_size=embedding_batch_size, device=device)
        for variant in selected_variants:
            output = output_root / variant / f"seed_{seed}" / "stage_a.pt"
            if output.exists() and not force:
                typer.echo(f"skip existing {variant} seed={seed}")
                continue
            typer.echo(f"train {variant} seed={seed}")
            train_inductive_stage_a(
                partitions["train"],
                partitions["validation"],
                encoder,
                InductiveStageAConfig(
                    mode="hypergraph",
                    profile="evidence_hypergraph",
                    output_path=str(output),
                    epochs=epochs,
                    seed=seed,
                    device=device,
                    variant=variant,
                    bridge_loss_weight=0.0 if variant == "no_bridge_loss" else 0.5,
                ),
            )
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    typer.run(main)

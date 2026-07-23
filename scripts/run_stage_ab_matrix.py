from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch
import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.local_models import LocalBGEEncoder, LocalBGEReranker
from qmshe.evaluation.splits import fixed_partition
from qmshe.evaluation.stage_ab import (
    aggregate_stage_ab, render_stage_ab, run_stage_ab_condition, stage_ab_effects,
)


SYSTEMS = (
    ("hypergraph", "evidence_hypergraph"),
    ("graph", "entity_relation"),
    ("graph", "reified_fact"),
)


def main(
    embedding_model: Path = typer.Option(...), reranker_model: Path = typer.Option(...),
    dense_stage_b_adapters: str = typer.Option(...), reranker_adapters: str = typer.Option(...),
    stage_ab_root: Path = typer.Option(Path("data/models/stage_ab")),
    seeds: str = typer.Option("13,42,73"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output_dir: Path = typer.Option(Path("reports/stage_ab_matrix")),
    dataset: str = typer.Option("hotpotqa"), limit: int = typer.Option(500),
    embedding_batch_size: int = typer.Option(16), reranker_batch_size: int = typer.Option(8),
    device: str = typer.Option("cuda"),
) -> None:
    seed_values = [int(item) for item in seeds.split(",")]
    dense_adapters = _mapping(dense_stage_b_adapters)
    reranker_paths = _mapping(reranker_adapters)
    suite = load_benchmark(dataset, input_path, split="test", limit=limit)
    test_examples = fixed_partition(suite.examples)["test"]
    all_records = []
    for seed in seed_values:
        reranker = LocalBGEReranker(
            str(reranker_model), reranker_paths[seed], batch_size=reranker_batch_size,
            device=device,
        )
        for encoder_kind, adapter in (("base", None), ("dense", dense_adapters[seed])):
            encoder = LocalBGEEncoder(
                str(embedding_model), adapter, batch_size=embedding_batch_size, device=device
            )
            cells = ((False, False), (True, False)) if encoder_kind == "base" else ((False, True),)
            for stage_a, stage_b in cells:
                for mode, profile in SYSTEMS:
                    checkpoint = _checkpoint(stage_ab_root, mode, profile, seed) if stage_a else None
                    all_records.extend(run_stage_ab_condition(
                        test_examples, encoder, reranker, mode, profile, seed,
                        stage_a, stage_b, checkpoint,
                        output_dir / f"seed_{seed}" / f"{mode}_{profile}_A{int(stage_a)}B{int(stage_b)}",
                    ))
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
        for mode, profile in SYSTEMS:
            adapter = stage_ab_root / f"{mode}_{profile}" / f"seed_{seed}" / "stage_b"
            encoder = LocalBGEEncoder(
                str(embedding_model), str(adapter), batch_size=embedding_batch_size, device=device
            )
            all_records.extend(run_stage_ab_condition(
                test_examples, encoder, reranker, mode, profile, seed, True, True,
                _checkpoint(stage_ab_root, mode, profile, seed),
                output_dir / f"seed_{seed}" / f"{mode}_{profile}_A1B1",
            ))
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
        del reranker
        gc.collect()
        torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "all_records.json").write_text(
        json.dumps([asdict(item) for item in all_records], indent=2), encoding="utf-8"
    )
    summary = aggregate_stage_ab(all_records)
    effects = stage_ab_effects(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(
        render_stage_ab(summary, effects, len(test_examples), seed_values), encoding="utf-8"
    )
    typer.echo(f"wrote {len(all_records)} records to {output_dir}")


def _mapping(value):
    return {int(seed): path for seed, path in (item.split("=", 1) for item in value.split(","))}


def _checkpoint(root, mode, profile, seed):
    return torch.load(
        root / f"{mode}_{profile}" / f"seed_{seed}" / "stage_a.pt",
        map_location="cpu", weights_only=True,
    )


if __name__ == "__main__":
    typer.run(main)

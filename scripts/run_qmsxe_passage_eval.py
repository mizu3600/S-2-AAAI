from __future__ import annotations

import gc
import json
from pathlib import Path
from statistics import mean, stdev

import torch
import typer

from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.evaluation.local_models import LocalBGEEncoder, LocalBGEReranker
from qmshe.evaluation.retrieval_metrics import complete_at_k, hit_at_k, recall_at_k, reciprocal_rank
from qmshe.evaluation.splits import fixed_partition
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline


SYSTEMS = (
    ("graph", "entity_relation"),
    ("graph", "reified_fact"),
    ("hypergraph", "evidence_hypergraph"),
)


def main(
    embedding_model: Path = typer.Option(...),
    reranker_model: Path = typer.Option(...),
    reranker_adapters: str = typer.Option(...),
    stage_ab_root: Path = typer.Option(Path("data/models/stage_ab")),
    seeds: str = typer.Option("13,42,73"),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output_dir: Path = typer.Option(Path("reports/qmsxe_passage_hotpot2000")),
    limit: int = typer.Option(2000),
    embedding_batch_size: int = typer.Option(16),
    reranker_batch_size: int = typer.Option(8),
    device: str = typer.Option("cuda"),
) -> None:
    seed_values = [int(item) for item in seeds.split(",")]
    reranker_paths = _mapping(reranker_adapters)
    suite = load_benchmark("hotpotqa", input_path, split="test", limit=limit)
    examples = fixed_partition(suite.examples)["test"]
    records = []
    for seed in seed_values:
        reranker = LocalBGEReranker(
            str(reranker_model),
            reranker_paths[seed],
            batch_size=reranker_batch_size,
            device=device,
        )
        for mode, profile in SYSTEMS:
            system_root = stage_ab_root / f"{mode}_{profile}" / f"seed_{seed}"
            encoder = LocalBGEEncoder(
                str(embedding_model),
                str(system_root / "stage_b"),
                batch_size=embedding_batch_size,
                device=device,
            )
            checkpoint = torch.load(
                system_root / "stage_a.pt",
                map_location="cpu",
                weights_only=True,
            )
            for example in examples:
                built = build_example_corpus(example)
                if mode == "hypergraph":
                    pipeline = QMSHEPipeline(
                        built.corpus,
                        text_encoder=encoder,
                        reranker=reranker,
                        seed=seed,
                        enable_remote_reranker=False,
                    )
                else:
                    pipeline = QMSGEGraphPipeline(
                        built.corpus,
                        text_encoder=encoder,
                        reranker=reranker,
                        profile=GraphProfile(profile),
                        seed=seed,
                        enable_remote_reranker=False,
                    )
                pipeline.load_stage_a_checkpoint(checkpoint)
                pipeline.generator.client = None
                result = pipeline.query(
                    example.question, top_k=40, return_debug=False, candidate_count=60
                )
                facts = (
                    result.retrieved_hyperedges if mode == "hypergraph" else result.retrieved_facts
                )
                chunk_to_document = {
                    chunk.chunk_id: chunk.document_id for chunk in built.corpus.chunks
                }
                fact_to_document = {
                    fact.hyperedge_id: chunk_to_document[fact.evidence_chunk_ids[0]]
                    for fact in built.corpus.evidence_hyperedges
                }
                ranking = _deduplicate(
                    [fact_to_document[fact_id] for fact_id in facts if fact_id in fact_to_document]
                )
                gold = {fact_to_document[fact_id] for fact_id in built.gold_fact_ids}
                records.append(
                    {
                        "example_id": example.example_id,
                        "framework": f"qmsxe:{mode}:{profile}",
                        "seed": seed,
                        "ranking": ranking,
                        "gold_document_ids": sorted(gold),
                        "recall_at_1": recall_at_k(ranking, gold, 1),
                        "recall_at_2": recall_at_k(ranking, gold, 2),
                        "recall_at_5": recall_at_k(ranking, gold, 5),
                        "recall_at_10": recall_at_k(ranking, gold, 10),
                        "hit_at_1": hit_at_k(ranking, gold, 1),
                        "complete_at_2": complete_at_k(ranking, gold, 2),
                        "complete_at_5": complete_at_k(ranking, gold, 5),
                        "mrr": reciprocal_rank(ranking, gold),
                    }
                )
            del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del reranker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = _aggregate(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(f"wrote {len(records)} records to {output_dir}")


def _mapping(value: str) -> dict[int, str]:
    return {int(seed): path for seed, path in (item.split("=", 1) for item in value.split(","))}


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _aggregate(records: list[dict]) -> dict:
    metrics = (
        "recall_at_1",
        "recall_at_2",
        "recall_at_5",
        "recall_at_10",
        "hit_at_1",
        "complete_at_2",
        "complete_at_5",
        "mrr",
    )
    grouped = {}
    for record in records:
        grouped.setdefault(record["framework"], {}).setdefault(record["seed"], []).append(record)
    output = {}
    for framework, seeds in grouped.items():
        output[framework] = {"seed_count": len(seeds)}
        for metric in metrics:
            values = [mean(row[metric] for row in rows) for rows in seeds.values()]
            output[framework][f"{metric}_mean"] = mean(values)
            output[framework][f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return output


if __name__ == "__main__":
    typer.run(main)

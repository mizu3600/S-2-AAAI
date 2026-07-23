from __future__ import annotations

import gc
from pathlib import Path

import torch
import typer

from qmshe.benchmarks import load_benchmark
from qmshe.evaluation.local_models import LocalBGEEncoder, LocalBGEReranker
from qmshe.evaluation.rag_baselines import (
    dense_ranking,
    make_record,
    prepare_rag_examples,
    reciprocal_rank_fusion_ids,
    rerank,
    write_rag_baseline_report,
)
from qmshe.evaluation.splits import fixed_partition


def main(
    embedding_model: Path = typer.Option(...),
    reranker_model: Path = typer.Option(...),
    embedding_adapters: str = typer.Option(...),
    reranker_adapters: str = typer.Option(...),
    seeds: str = typer.Option("13,42,73"),
    input_path: Path = typer.Option(
        Path("data/benchmarks/hotpot_dev_distractor_v1.json")
    ),
    output_dir: Path = typer.Option(Path("reports/rag_baselines")),
    dataset: str = typer.Option("hotpotqa"),
    limit: int = typer.Option(2000),
    candidate_count: int = typer.Option(60),
    embedding_batch_size: int = typer.Option(16),
    reranker_batch_size: int = typer.Option(8),
    device: str = typer.Option("cuda"),
) -> None:
    seed_values = [int(item) for item in seeds.split(",") if item]
    embedding_paths = _mapping(embedding_adapters)
    reranker_paths = _mapping(reranker_adapters)
    suite = load_benchmark(dataset, input_path, split="test", limit=limit)
    examples = fixed_partition(suite.examples)["test"]

    base_encoder = LocalBGEEncoder(
        str(embedding_model), batch_size=embedding_batch_size, device=device
    )
    prepared = prepare_rag_examples(examples, base_encoder, candidate_count)
    base_queries = base_encoder.encode_queries([item.question for item in prepared])
    base_rankings = {}
    records = []
    for item, query in zip(prepared, base_queries, strict=True):
        dense = dense_ranking(item, query, candidate_count)
        hybrid = reciprocal_rank_fusion_ids(item.bm25_ranking, dense)
        base_rankings[item.example_id] = {"dense": dense, "hybrid": hybrid}
        records.extend([
            make_record(item, item.bm25_ranking, "bm25", 0),
            make_record(item, dense, "vanilla_dense_bge_m3", 0),
            make_record(item, hybrid, "hybrid_bm25_dense", 0),
        ])

    del base_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for seed in seed_values:
        tuned_encoder = LocalBGEEncoder(
            str(embedding_model), embedding_paths[seed],
            batch_size=embedding_batch_size, device=device,
        )
        tuned_queries = tuned_encoder.encode_queries([item.question for item in prepared])
        tuned_rankings = {}
        for item, query in zip(prepared, tuned_queries, strict=True):
            dense = dense_ranking(item, query, candidate_count)
            tuned_rankings[item.example_id] = {
                "dense": dense,
                "hybrid": reciprocal_rank_fusion_ids(item.bm25_ranking, dense),
            }
            records.extend([
                make_record(item, dense, "tuned_query_dense", seed),
                make_record(
                    item, tuned_rankings[item.example_id]["hybrid"],
                    "tuned_query_hybrid", seed,
                ),
            ])

        reranker_model_instance = LocalBGEReranker(
            str(reranker_model), reranker_paths[seed],
            batch_size=reranker_batch_size, device=device,
        )
        for item in prepared:
            base = base_rankings[item.example_id]
            tuned = tuned_rankings[item.example_id]
            rankings = {
                "bm25_tuned_reranker": item.bm25_ranking,
                "dense_tuned_reranker": base["dense"],
                "hybrid_tuned_reranker": base["hybrid"],
                "tuned_query_dense_tuned_reranker": tuned["dense"],
                "tuned_query_hybrid_tuned_reranker": tuned["hybrid"],
            }
            for method, ranking in rankings.items():
                records.append(
                    make_record(item, rerank(item, ranking, reranker_model_instance), method, seed)
                )
        del tuned_encoder, reranker_model_instance
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = write_rag_baseline_report(records, output_dir, len(examples))
    typer.echo(f"wrote {len(records)} records and {len(summary)} methods to {output_dir}")


def _mapping(value: str):
    return {
        int(seed): path
        for seed, path in (item.split("=", 1) for item in value.split(","))
    }


if __name__ == "__main__":
    typer.run(main)

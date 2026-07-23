from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import typer

from qmshe.benchmark_v2.dataset import build_candidate_view, global_passage_pool, load_hotpot_dev
from qmshe.benchmark_v2.evaluator import evaluate_prediction
from qmshe.benchmark_v2.report import aggregate, paired_comparisons, render_markdown
from qmshe.benchmark_v2.schemas import RankingPrediction
from qmshe.benchmark_v2.trained_parity import (
    install_graph_parity_hooks,
    install_hypergraph_parity_hooks,
)
from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.evaluation.local_models import LocalBGEEncoder, LocalBGEReranker
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
    reranker_adapter: Path = typer.Option(...),
    stage_ab_root: Path = typer.Option(Path("data/models/stage_ab")),
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    selection_path: Path = typer.Option(
        Path("data/benchmarks/hotpotqa_official_baselines_288.json")
    ),
    output_dir: Path = typer.Option(Path("reports/benchmark_v2_trained")),
    seed: int = typer.Option(42),
    limit: int = typer.Option(288),
    fact_budget: int = typer.Option(60),
    device: str = typer.Option("cuda"),
) -> None:
    selected_ids = [row["example_id"] for row in json.loads(selection_path.read_text())][:limit]
    legacy_all = load_benchmark("hotpotqa", input_path, split="dev").examples
    v2_all = load_hotpot_dev(input_path)
    legacy = {example.example_id: example for example in legacy_all}
    v2 = {example.example_id: example for example in v2_all}
    pool = global_passage_pool(v2_all)
    views = {
        example_id: build_candidate_view(v2[example_id], pool, 10) for example_id in selected_ids
    }
    records = []
    reranker = LocalBGEReranker(
        str(reranker_model), str(reranker_adapter), batch_size=8, device=device
    )
    for mode, profile in SYSTEMS:
        root = stage_ab_root / f"{mode}_{profile}" / f"seed_{seed}"
        encoder = LocalBGEEncoder(
            str(embedding_model), str(root / "stage_b"), batch_size=16, device=device
        )
        checkpoint = torch.load(root / "stage_a.pt", map_location="cpu", weights_only=True)
        for number, example_id in enumerate(selected_ids, 1):
            started = time.perf_counter()
            built = build_example_corpus(legacy[example_id])
            if mode == "graph":
                pipeline = QMSGEGraphPipeline(
                    built.corpus,
                    encoder,
                    GraphProfile(profile),
                    reranker=reranker,
                    seed=seed,
                    enable_remote_reranker=False,
                )
                pipeline.load_stage_a_checkpoint(checkpoint)
                install_graph_parity_hooks(pipeline, fact_budget)
            else:
                pipeline = QMSHEPipeline(
                    built.corpus,
                    encoder,
                    reranker=reranker,
                    seed=seed,
                    enable_remote_reranker=False,
                )
                pipeline.load_stage_a_checkpoint(checkpoint)
                install_hypergraph_parity_hooks(pipeline, fact_budget)
            pipeline.generator.client = None
            result = pipeline.query(
                legacy[example_id].question, top_k=40, return_debug=False, candidate_count=60
            )
            old_fact_ids = (
                result.retrieved_facts if mode == "graph" else result.retrieved_hyperedges
            )
            chunk_to_document = {chunk.chunk_id: chunk.document_id for chunk in built.corpus.chunks}
            old_fact_to_document = {
                fact.hyperedge_id: chunk_to_document[fact.evidence_chunk_ids[0]]
                for fact in built.corpus.evidence_hyperedges
            }
            canonical_document_ranking = list(
                dict.fromkeys(
                    old_fact_to_document[fact_id]
                    for fact_id in old_fact_ids
                    if fact_id in old_fact_to_document
                )
            )
            text_by_old = {
                fact.hyperedge_id: fact.evidence_sentence
                for fact in built.corpus.evidence_hyperedges
            }
            text_to_v2 = {fact.text: fact for fact in views[example_id].facts}
            v2_facts = [
                text_to_v2[text_by_old[fact_id]]
                for fact_id in old_fact_ids
                if text_by_old.get(fact_id) in text_to_v2
            ]
            fact_ranking = tuple(fact.fact_id for fact in v2_facts)
            passage_ranking = tuple(dict.fromkeys(fact.passage_id for fact in v2_facts))
            prediction = RankingPrediction(
                system=f"trained_parity:{mode}:{profile}:seed_{seed}",
                fact_ranking=fact_ranking,
                passage_ranking=passage_ranking,
                path=passage_ranking[:2],
                answer=result.answer,
                citations=fact_ranking[:2],
                diagnostics={
                    "ranking_origin": "internal_fact_scores",
                    "reranker_inputs": fact_budget,
                },
            )
            record = evaluate_prediction(views[example_id], prediction)
            record.update(
                {
                    "canonical_fact_ranking": fact_ranking,
                    "canonical_document_ranking": canonical_document_ranking,
                    "elapsed_seconds": time.perf_counter() - started,
                    "fact_budget": fact_budget,
                }
            )
            records.append(record)
            if number % 25 == 0:
                typer.echo(f"{mode}:{profile} {number}/{len(selected_ids)}")
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    summary = aggregate(records)
    comparisons = paired_comparisons(records)
    manifest = {
        "protocol": "benchmark_v2_trained_parity",
        "examples": len(selected_ids),
        "seed": seed,
        "fact_budget": fact_budget,
        "reranker_inputs": fact_budget,
        "structured_track": False,
        "note": "Uses existing trained checkpoints and legacy extraction; structured-role track requires retraining.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(summary, manifest), encoding="utf-8")


if __name__ == "__main__":
    typer.run(main)

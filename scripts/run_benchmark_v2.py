from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import typer

from qmshe.benchmark_v2.dataset import build_candidate_view, global_passage_pool, load_hotpot_dev
from qmshe.benchmark_v2.evaluator import evaluate_prediction
from qmshe.benchmark_v2.extraction import StructuredFactExtractor
from qmshe.benchmark_v2.nary_stress import build_nary_stress_suite
from qmshe.benchmark_v2.report import aggregate, paired_comparisons, render_markdown
from qmshe.benchmark_v2.systems import ControlledTopologyRetriever


def main(
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output_dir: Path = typer.Option(Path("reports/benchmark_v2")),
    candidate_counts: str = typer.Option("10,100,1000"),
    limits: str = typer.Option("0,1000,200", help="Per candidate-count limits; 0 means full dev."),
    retrieval_budget: int = typer.Option(60),
    output_facts: int = typer.Option(40),
    nary_size: int = typer.Option(500),
    seed: int = typer.Option(42),
) -> None:
    counts = [int(value) for value in candidate_counts.split(",")]
    count_limits = [int(value) for value in limits.split(",")]
    if len(counts) != len(count_limits):
        raise typer.BadParameter("candidate-counts and limits must have equal lengths")
    examples = load_hotpot_dev(input_path)
    pool = global_passage_pool(examples)
    extractor = StructuredFactExtractor()
    retriever = ControlledTopologyRetriever(retrieval_budget, output_facts)
    records, started = [], time.perf_counter()
    for candidate_count, limit in zip(counts, count_limits, strict=True):
        selected = examples if limit == 0 else examples[:limit]
        for number, example in enumerate(selected, 1):
            view = build_candidate_view(example, pool, candidate_count, extractor, seed)
            for profile in retriever.profiles:
                records.append(evaluate_prediction(view, retriever.rank(view, profile)))
            if number % 100 == 0:
                typer.echo(f"candidate_count={candidate_count}: {number}/{len(selected)}")
    for view in build_nary_stress_suite(nary_size, seed):
        for profile in retriever.profiles:
            records.append(evaluate_prediction(view, retriever.rank(view, profile)))
    summary = aggregate(records)
    comparisons = paired_comparisons(records)
    manifest = {
        "protocol": "benchmark_v2",
        "dataset": "HotpotQA distractor dev v1",
        "dataset_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "dataset_examples": len(examples),
        "evaluation_split": "official dev (distractor)",
        "training_split_used": False,
        "query_types": {kind: sum(example.query_type == kind for example in examples) for kind in ("bridge", "comparison")},
        "candidate_matrix": [
            {"candidate_count": count, "examples": len(examples) if limit == 0 else limit}
            for count, limit in zip(counts, count_limits, strict=True)
        ],
        "nary_examples": nary_size,
        "shared_retrieval_budget": retrieval_budget,
        "shared_reranker_inputs": retrieval_budget,
        "shared_output_facts": output_facts,
        "ranking_source": "internal fact scores",
        "structured_extractor": extractor.version,
        "expanded_candidate_source": "label-free deterministic sampling from other dev passages",
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(summary, manifest), encoding="utf-8")
    typer.echo(f"wrote {len(records)} records to {output_dir}")


if __name__ == "__main__":
    typer.run(main)

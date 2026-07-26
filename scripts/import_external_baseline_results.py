from __future__ import annotations

import hashlib
from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.evaluation.experiment import EvaluationConfig, score_external_result
from qmshe.evaluation.external_adapters import BASELINE_SPECS, load_external_results
from qmshe.evaluation.report import write_report


def main(
    baseline: str = typer.Option(..., help=f"One of: {', '.join(BASELINE_SPECS)}"),
    input_path: Path = typer.Option(..., help="Source benchmark dataset"),
    result_path: Path = typer.Option(..., help="Native baseline JSON or JSONL result file"),
    dataset: str = typer.Option("hotpotqa"),
    output_dir: Path = typer.Option(Path("data/experiments/external")),
    split: str = typer.Option("validation"),
    limit: int = typer.Option(0),
    seed: int = typer.Option(42),
) -> None:
    baseline = baseline.casefold()
    dataset = dataset.casefold()
    if dataset != "hotpotqa":
        raise typer.BadParameter(
            "the strict external comparison protocol currently supports hotpotqa only",
            param_hint="--dataset",
        )
    suite = load_benchmark(
        dataset, input_path, split=split, limit=None if limit == 0 else limit
    )
    normalized = load_external_results(baseline, result_path, suite)
    config = EvaluationConfig()
    examples = {example.example_id: example for example in suite.examples}
    records = []
    for result in normalized:
        example = examples[result.example_id]
        records.append(score_external_result(
            example, build_example_corpus(example), result, seed, config
        ))
    target = output_dir / baseline
    missing = sum(item.status == "missing" for item in normalized)
    failed = sum(item.status == "failed" for item in normalized)
    unscorable = sum(item.status == "unscorable" for item in normalized)
    received = [item for item in normalized if item.status != "missing"]
    mapping_coverage_received = (
        sum(item.mapping_coverage for item in received) / len(received)
        if received
        else 0.0
    )
    end_to_end_mapping_coverage = (
        sum(item.mapping_coverage for item in normalized) / len(normalized)
        if normalized
        else 0.0
    )
    write_report(
        records,
        target,
        {
            "dataset": dataset,
            "split": split,
            "input_path": str(input_path),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "result_path": str(result_path),
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "examples": len(suite.examples),
            "received_results": len(normalized) - missing,
            "missing_results": missing,
            "failed_results": failed,
            "unscorable_results": unscorable,
            "result_coverage": len(received) / len(normalized) if normalized else 0.0,
            "mapping_coverage_received": mapping_coverage_received,
            "end_to_end_mapping_coverage": end_to_end_mapping_coverage,
            "seed": seed,
            "baseline": baseline,
            "protocol": "hotpotqa_native_external_passage_v2",
            "output_k": config.output_k,
            "candidate_k": "native",
            "context_k": config.context_k,
            "generation_protocol": "native_or_NA",
            "repository": BASELINE_SPECS[baseline].repository,
            "ranking_contract": BASELINE_SPECS[baseline].ranking_contract,
        },
    )
    typer.echo(
        f"scored {len(records)}/{len(suite.examples)} {baseline} examples "
        f"(missing={missing}, failed={failed}, "
        f"mapping={mapping_coverage_received:.3f}) into {target}"
    )


if __name__ == "__main__":
    typer.run(main)

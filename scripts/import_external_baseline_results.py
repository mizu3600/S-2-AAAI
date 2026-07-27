from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import typer

from s2rag.benchmarks import load_benchmark
from s2rag.benchmarks.corpus_builder import build_native_passage_evaluation_corpus
from s2rag.evaluation.experiment import (
    UNIFIED_RETRIEVAL_PROTOCOL,
    EvaluationConfig,
    expected_shared_model_trace,
    score_external_result,
)
from s2rag.evaluation.external_adapters import BASELINE_SPECS, load_external_results
from s2rag.evaluation.official_metrics import official_metric_spec
from s2rag.evaluation.report import write_report


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
    if dataset not in {"hotpotqa", "musique", "2wikimultihopqa", "ultradomain"}:
        raise typer.BadParameter(
            "dataset must be one of hotpotqa, musique, 2wikimultihopqa, ultradomain",
            param_hint="--dataset",
        )
    suite = load_benchmark(dataset, input_path, split=split, limit=None if limit == 0 else limit)
    shared_model_trace = expected_shared_model_trace()
    normalized = load_external_results(
        baseline,
        result_path,
        suite,
        expected_shared_model_trace=shared_model_trace,
    )
    config = EvaluationConfig()
    examples = {example.example_id: example for example in suite.examples}
    records = []
    for result in normalized:
        example = examples[result.example_id]
        records.append(
            score_external_result(
                example,
                build_native_passage_evaluation_corpus(example),
                result,
                seed,
                config,
            )
        )
    target = output_dir / baseline
    missing = sum(item.status == "missing" for item in normalized)
    failed = sum(item.status == "failed" for item in normalized)
    unscorable = sum(item.status == "unscorable" for item in normalized)
    protocol_mismatch = any(not item.generation_protocol_matched for item in normalized)
    received = [item for item in normalized if item.status != "missing"]
    mapping_coverage_received = (
        sum(item.mapping_coverage for item in received) / len(received) if received else 0.0
    )
    end_to_end_mapping_coverage = (
        sum(item.mapping_coverage for item in normalized) / len(normalized) if normalized else 0.0
    )
    official_spec = official_metric_spec(dataset)
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
            "expected_examples": len(suite.examples),
            "expected_example_ids_sha256": hashlib.sha256(
                "\n".join(sorted(examples)).encode()
            ).hexdigest(),
            "received_results": len(normalized) - missing,
            "missing_results": missing,
            "failed_results": failed,
            "unscorable_results": unscorable,
            "result_coverage": len(received) / len(normalized) if normalized else 0.0,
            "mapping_coverage_received": mapping_coverage_received,
            "end_to_end_mapping_coverage": end_to_end_mapping_coverage,
            "seed": seed,
            "baseline": baseline,
            "protocol": UNIFIED_RETRIEVAL_PROTOCOL,
            "evaluation_view": "native_external_passage",
            "corpus_protocol": official_spec["corpus_protocol"],
            "fact_extraction": "not_applicable",
            "output_k": config.output_k,
            "rerank_input_k": "native",
            "context_k": config.context_k,
            "generation_protocol": BASELINE_SPECS[baseline].capability.generation_protocol,
            "generation_protocol_matched": not protocol_mismatch,
            "shared_generation_eligible": not protocol_mismatch,
            "shared_model_trace": shared_model_trace,
            "shared_model_protocol_matched": all(
                item.shared_model_protocol_matched for item in normalized
            ),
            "capability": asdict(BASELINE_SPECS[baseline].capability),
            "repository": BASELINE_SPECS[baseline].repository,
            "ranking_contract": BASELINE_SPECS[baseline].ranking_contract,
            "official_metrics": official_spec,
        },
    )
    typer.echo(
        f"scored {len(records)}/{len(suite.examples)} {baseline} examples "
        f"(missing={missing}, failed={failed}, "
        f"mapping={mapping_coverage_received:.3f}) into {target}"
    )


if __name__ == "__main__":
    typer.run(main)

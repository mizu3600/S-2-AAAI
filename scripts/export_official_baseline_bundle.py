from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from s2rag.benchmarks import load_benchmark
from s2rag.evaluation.splits import fixed_partition
from s2rag.evaluation.experiment import (
    UNIFIED_RETRIEVAL_PROTOCOL,
    expected_shared_model_trace,
)


def main(
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_official_baselines_288.json")),
    limit: int = typer.Option(2000),
) -> None:
    suite = load_benchmark("hotpotqa", input_path, split="test", limit=limit)
    examples = fixed_partition(suite.examples)["test"]
    payload = []
    for example in examples:
        gold_passage_ids = sorted({support.passage_id for support in example.supporting_facts})
        payload.append(
            {
                "schema": "hotpotqa_canonical_source_passage_v1",
                "dataset": "hotpotqa",
                "split": "test",
                "example_id": example.example_id,
                "question": example.question,
                "answer": example.answer,
                "documents": [
                    {
                        "document_id": passage.passage_id,
                        "passage_id": passage.passage_id,
                        "title": passage.title,
                        "text": " ".join(passage.sentences),
                    }
                    for passage in example.passages
                ],
                "gold_passage_ids": gold_passage_ids,
                "source_id_map": {
                    passage.title: passage.passage_id for passage in example.passages
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "protocol": UNIFIED_RETRIEVAL_PROTOCOL,
        "evaluation_view": "native_external_passage",
        "source": str(input_path),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "examples": len(payload),
        "requested_limit": limit,
        "partition": "stable_id_test_15_percent",
        "ranking_unit": "canonical_passage_id",
        "corpus_protocol": "canonical_source_passages_v1",
        "fact_extraction": "not_applicable",
        "required_shared_model_trace": expected_shared_model_trace(),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"wrote {len(payload)} examples to {output_path}")


if __name__ == "__main__":
    typer.run(main)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from qmshe.benchmarks import load_benchmark
from qmshe.benchmarks.corpus_builder import build_example_corpus
from qmshe.evaluation.splits import fixed_partition


def _verbalize_fact(fact, entity_names: dict[str, str]) -> str:
    arguments = ", ".join(
        f"{argument.role}={entity_names.get(argument.entity_id, argument.entity_id)}"
        for argument in fact.arguments
    )
    qualifiers = ", ".join(
        f"{key}={value}" for key, value in fact.qualifiers.items() if value is not None
    )
    return f"{fact.predicate}: {arguments}" + (f"; {qualifiers}" if qualifiers else "")


def main(
    input_path: Path = typer.Option(Path("data/benchmarks/hotpot_dev_distractor_v1.json")),
    output_path: Path = typer.Option(Path("data/benchmarks/hotpotqa_official_baselines_288.json")),
    limit: int = typer.Option(2000),
) -> None:
    suite = load_benchmark("hotpotqa", input_path, split="test", limit=limit)
    examples = fixed_partition(suite.examples)["test"]
    payload = []
    for example in examples:
        built = build_example_corpus(example)
        entity_names = {entity.entity_id: entity.canonical_name for entity in built.corpus.entities}
        gold_passage_ids = sorted({
            built.fact_to_passage[fact_id] for fact_id in built.gold_fact_ids
        })
        payload.append(
            {
                "schema": "hotpotqa_canonical_passage_v2",
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
                "facts": [
                    {
                        "fact_id": fact.hyperedge_id,
                        "text": _verbalize_fact(fact, entity_names),
                        "sentence": fact.evidence_sentence,
                        "document_id": built.fact_to_passage[fact.hyperedge_id],
                        "passage_id": built.fact_to_passage[fact.hyperedge_id],
                    }
                    for fact in built.corpus.evidence_hyperedges
                ],
                "gold_fact_ids": sorted(built.gold_fact_ids),
                "gold_passage_ids": gold_passage_ids,
                "source_id_map": {
                    passage.title: passage.passage_id for passage in example.passages
                },
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "protocol": "hotpotqa_native_external_passage_v2",
        "source": str(input_path),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "examples": len(payload),
        "requested_limit": limit,
        "partition": "stable_id_test_15_percent",
        "ranking_unit": "canonical_passage_id",
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"wrote {len(payload)} examples to {output_path}")


if __name__ == "__main__":
    typer.run(main)

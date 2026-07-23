from __future__ import annotations

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
        chunk_by_id = {chunk.chunk_id: chunk for chunk in built.corpus.chunks}
        chunks_by_document: dict[str, list] = {}
        for chunk in built.corpus.chunks:
            chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
        payload.append(
            {
                "example_id": example.example_id,
                "question": example.question,
                "answer": example.answer,
                "documents": [
                    {
                        "document_id": document.document_id,
                        "title": document.title,
                        "text": " ".join(
                            chunk.text for chunk in chunks_by_document[document.document_id]
                        ),
                    }
                    for document in built.corpus.documents
                ],
                "facts": [
                    {
                        "fact_id": fact.hyperedge_id,
                        "text": _verbalize_fact(fact, entity_names),
                        "sentence": fact.evidence_sentence,
                        "document_id": chunk_by_id[fact.evidence_chunk_ids[0]].document_id,
                    }
                    for fact in built.corpus.evidence_hyperedges
                ],
                "gold_fact_ids": sorted(built.gold_fact_ids),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"wrote {len(payload)} examples to {output_path}")


if __name__ == "__main__":
    typer.run(main)

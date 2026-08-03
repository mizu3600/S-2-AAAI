from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from s2rag.benchmarks import load_benchmark


DATASETS = ("hotpotqa", "musique", "2wikimultihopqa", "ultradomain")


def main(
    benchmark_dir: Path = typer.Option(Path("data/benchmarks")),
    output_dir: Path = typer.Option(Path("data/benchmarks/protocol_v2_100")),
    sample_size: int = typer.Option(100, min=1),
    smoke_size: int = typer.Option(10, min=1),
    seed: int = typer.Option(42),
) -> None:
    if smoke_size > sample_size:
        raise typer.BadParameter("smoke-size must not exceed sample-size")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for dataset in DATASETS:
        source = benchmark_dir / f"{dataset}_1000.jsonl"
        suite = load_benchmark(dataset, source, split="test", limit=sample_size)
        rows = [_canonical_row(example, seed) for example in suite.examples]
        if len(rows) != sample_size:
            raise ValueError(
                f"{dataset} contains {len(rows)} prepared examples; expected {sample_size}"
            )

        full_path = output_dir / f"{dataset}_seed_42_100.json"
        smoke_path = output_dir / f"{dataset}_seed_42_smoke_10.json"
        _write_json(full_path, rows)
        _write_json(smoke_path, rows[:smoke_size])
        manifests[dataset] = {
            "dataset": dataset,
            "seed": seed,
            "source_file": str(source.resolve()),
            "source_sha256": _sha256(source),
            "full": _suite_manifest(full_path, rows),
            "smoke": _suite_manifest(smoke_path, rows[:smoke_size]),
            "selection": "first_n_from_seed_42_fixed_1000_suite",
            "corpus_scope": _corpus_scope(rows),
        }

    manifest_path = output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "protocol": "s2rag_protocol_v2_fixed_100_v1",
            "seed": seed,
            "sample_size": sample_size,
            "smoke_size": smoke_size,
            "ultradomain_document_policy": "complete_document_no_prefilter",
            "datasets": manifests,
        },
    )
    typer.echo(f"wrote protocol-v2 suites to {output_dir}")


def _canonical_row(example, seed: int) -> dict:
    facts = []
    fact_id_by_location = {}
    documents = []
    for passage in example.passages:
        documents.append(
            {
                "document_id": passage.passage_id,
                "title": passage.title,
                "text": " ".join(passage.sentences),
            }
        )
        for sentence_index, sentence in enumerate(passage.sentences):
            fact_id = _fact_id(passage.passage_id, sentence_index)
            facts.append(
                {
                    "fact_id": fact_id,
                    "document_id": passage.passage_id,
                    "text": sentence,
                    "sentence": sentence,
                }
            )
            fact_id_by_location[(passage.passage_id, sentence_index)] = fact_id

    gold_fact_ids = [
        fact_id_by_location[(support.passage_id, support.sentence_index)]
        for support in example.supporting_facts
        if (support.passage_id, support.sentence_index) in fact_id_by_location
    ]
    answer = example.answer[0] if isinstance(example.answer, list) else example.answer
    return {
        "schema": "canonical_multihop_example_v1",
        "dataset": example.dataset,
        "seed": seed,
        "example_id": example.example_id,
        "source_example_id": example.example_id,
        "question": example.question,
        "answer": answer,
        "answer_aliases": (
            list(example.answer) if isinstance(example.answer, list) else [example.answer]
        ),
        "documents": documents,
        "facts": facts,
        "gold_fact_ids": gold_fact_ids,
        "metadata": dict(example.metadata),
    }


def _fact_id(passage_id: str, sentence_index: int) -> str:
    digest = hashlib.sha256(
        f"{passage_id}:{sentence_index}".encode("utf-8")
    ).hexdigest()[:16]
    return f"fact_{digest}"


def _suite_manifest(path: Path, rows: list[dict]) -> dict:
    return {
        "file_path": str(path.resolve()),
        "target_sample_size": len(rows),
        "dataset_sha256": _sha256(path),
        "example_ids_sha256": hashlib.sha256(
            "\n".join(row["example_id"] for row in rows).encode("utf-8")
        ).hexdigest(),
    }


def _corpus_scope(rows: list[dict]) -> list[str]:
    scopes = {
        str(row.get("metadata", {}).get("corpus_scope", "unspecified"))
        for row in rows
    }
    return sorted(scopes)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    typer.run(main)

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from s2rag.benchmarks.schemas import (
    BenchmarkExample,
    BenchmarkSuite,
    Passage,
    SupportingFact,
)
from s2rag.embedding.text_encoder import LocalBGEEncoder, reject_peft_adapter_directory
from s2rag.evaluation.experiment import BenchmarkExperimentRunner
from s2rag.retrieval.candidates import aggregate_passages
from s2rag.retrieval.local_reranker import LocalBGEReranker


SYSTEM = "s2rag:reified_fact_hybrid"
MODEL_PROTOCOL = "base_model_only"


class ProtocolV2Runner(BenchmarkExperimentRunner):
    """Expose S2RAG's native fact-to-passage ranking for protocol-v2."""

    def _score_method(self, *args, **kwargs) -> dict:
        record = super()._score_method(*args, **kwargs)
        built = args[1] if len(args) > 1 else kwargs["built"]
        candidates = args[4] if len(args) > 4 else kwargs["candidates"]
        record["passage_ranking"] = aggregate_passages(
            candidates, built.fact_to_passage
        )[: self.config.output_k]
        return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run base-model-only S2RAG on a protocol-v2 canonical suite."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--suite-sha256", required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from S2RAG records.partial.jsonl or records.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.resume and (
        (args.work_dir / "records.partial.jsonl").exists()
        or (args.work_dir / "records.json").exists()
        or args.output.exists()
    ):
        raise FileExistsError(
            "existing bridge output found; pass --resume to reuse its checkpoints"
        )
    suite_sha256 = sha256_file(args.input)
    if suite_sha256 != args.suite_sha256:
        raise ValueError(
            f"canonical input SHA256 changed: {suite_sha256} != {args.suite_sha256}"
        )
    _validate_base_model_directory(args.embedding_model, "BGE-M3")
    _validate_base_model_directory(args.reranker_model, "BGE reranker")

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    suite = canonical_rows_to_suite(
        rows,
        source=args.input,
        dataset=args.dataset,
        seed=args.seed,
        expected_count=args.expected_count,
    )
    runner = ProtocolV2Runner(
        methods=("reified_fact_hybrid",),
        generate_for_methods=(),
        encoder=LocalBGEEncoder(
            model_path=args.embedding_model,
            device=args.device,
        ),
        reranker=LocalBGEReranker(
            model_path=args.reranker_model,
            device=args.device,
        ),
    )
    records = runner.run(suite, args.work_dir, seed=args.seed)
    bridged = [
        bridge_record(
            record,
            dataset=args.dataset,
            seed=args.seed,
            suite_sha256=suite_sha256,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
        )
        for record in records
    ]
    _validate_bridge_records(
        bridged,
        rows=rows,
        expected_count=args.expected_count,
    )
    atomic_write_json(args.output, bridged)
    print(f"wrote {len(bridged)} {SYSTEM} records to {args.output}", flush=True)


def canonical_rows_to_suite(
    rows: list[dict[str, Any]],
    *,
    source: Path,
    dataset: str,
    seed: int,
    expected_count: int,
) -> BenchmarkSuite:
    if len(rows) != expected_count:
        raise ValueError(f"suite has {len(rows)} examples; expected {expected_count}")
    examples = [
        canonical_row_to_example(row, dataset=dataset, seed=seed) for row in rows
    ]
    example_ids = [example.example_id for example in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("canonical suite contains duplicate example_id values")
    return BenchmarkSuite(
        name=dataset,
        split="validation",
        examples=examples,
        source=str(source),
        version="protocol-v2",
    )


def canonical_row_to_example(
    row: dict[str, Any],
    *,
    dataset: str,
    seed: int,
) -> BenchmarkExample:
    if row.get("dataset") != dataset:
        raise ValueError(
            f"example {row.get('example_id')} dataset mismatch: "
            f"{row.get('dataset')!r} != {dataset!r}"
        )
    if row.get("seed") != seed:
        raise ValueError(
            f"example {row.get('example_id')} seed mismatch: "
            f"{row.get('seed')!r} != {seed}"
        )
    documents = row.get("documents") or []
    document_ids = [str(document["document_id"]) for document in documents]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError(f"example {row.get('example_id')} has duplicate document IDs")

    facts_by_document: dict[str, list[dict[str, Any]]] = {
        document_id: [] for document_id in document_ids
    }
    fact_location: dict[str, tuple[str, int]] = {}
    for fact in row.get("facts") or []:
        document_id = str(fact["document_id"])
        if document_id not in facts_by_document:
            raise ValueError(
                f"fact {fact.get('fact_id')} references unknown document {document_id}"
            )
        fact_id = str(fact["fact_id"])
        if fact_id in fact_location:
            raise ValueError(f"duplicate fact ID: {fact_id}")
        sentence_index = len(facts_by_document[document_id])
        facts_by_document[document_id].append(fact)
        fact_location[fact_id] = (document_id, sentence_index)

    passages = []
    for document in documents:
        document_id = str(document["document_id"])
        document_facts = facts_by_document[document_id]
        if document_facts:
            sentences = []
            for fact in document_facts:
                sentence = str(fact.get("sentence") or fact.get("text") or "").strip()
                if not sentence:
                    raise ValueError(
                        f"fact {fact.get('fact_id')} contains no usable sentence/text"
                    )
                sentences.append(sentence)
        else:
            sentences = _sentence_split(str(document.get("text", "")))
        if not sentences:
            raise ValueError(f"document {document_id} contains no usable text")
        passages.append(
            Passage(
                passage_id=document_id,
                title=str(document.get("title") or document_id),
                sentences=sentences,
                source_uri=(
                    f"protocol-v2://{dataset}/{row.get('example_id')}/{document_id}"
                ),
            )
        )

    supporting_facts = []
    for fact_id in row.get("gold_fact_ids") or []:
        if fact_id not in fact_location:
            raise ValueError(
                f"gold fact {fact_id} is absent from example {row.get('example_id')}"
            )
        document_id, sentence_index = fact_location[fact_id]
        supporting_facts.append(
            SupportingFact(
                passage_id=document_id,
                sentence_index=sentence_index,
            )
        )
    gold_path = list(
        dict.fromkeys(item.passage_id for item in supporting_facts)
    )
    metric_profile = (
        "2wikimultihopqa_official"
        if dataset == "2wikimultihopqa"
        else "hotpotqa_official"
    )
    return BenchmarkExample(
        example_id=str(row["example_id"]),
        question=str(row["question"]),
        answer=row.get("answer", ""),
        passages=passages,
        supporting_facts=supporting_facts,
        gold_path=gold_path,
        hop_count=max(1, len(gold_path)),
        query_type="canonical_multihop",
        dataset=dataset,
        split="validation",
        metadata={
            "source_example_id": row.get("source_example_id"),
            "metric_profile": metric_profile,
            "corpus_scope": "per_question_candidate_passages",
            "canonical_document_ids": document_ids,
        },
    )


def bridge_record(
    record: dict[str, Any],
    *,
    dataset: str,
    seed: int,
    suite_sha256: str,
    embedding_model: Path,
    reranker_model: Path,
) -> dict[str, Any]:
    ranking = list(dict.fromkeys(record.get("passage_ranking") or []))
    return {
        "system": SYSTEM,
        "framework": "s2rag",
        "example_id": record["example_id"],
        "status": record.get("status", "failed"),
        "ranking": ranking,
        "index_seconds": _milliseconds_to_seconds(
            record.get("total_preparation_ms")
        ),
        "retrieval_seconds": _milliseconds_to_seconds(record.get("retrieval_ms")),
        "error": record.get("error"),
        "dataset": dataset,
        "seed": seed,
        "suite_sha256": suite_sha256,
        "ranking_origin": "s2rag_training_free_fact_to_document",
        "model_protocol": MODEL_PROTOCOL,
        "embedding_model": str(embedding_model),
        "reranker_model": str(reranker_model),
        "graph_model_type": "training_free",
        "graph_model_id": record.get("graph_model_id"),
        "extraction_coverage": record.get("extraction_coverage"),
        "usage": {
            "llm_calls": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "embedding_calls": None,
            "embedding_tokens": None,
            "token_count_mode": "deepseek_extraction_unmeasured_local_base_models",
        },
    }


def _validate_bridge_records(
    records: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    expected_count: int,
) -> None:
    if len(records) != expected_count:
        raise ValueError(f"bridge emitted {len(records)} rows; expected {expected_count}")
    canonical_ids = {
        str(row["example_id"]): {
            str(document["document_id"]) for document in row["documents"]
        }
        for row in rows
    }
    seen: set[str] = set()
    for record in records:
        example_id = str(record["example_id"])
        if example_id in seen:
            raise ValueError(f"duplicate bridge record for {example_id}")
        seen.add(example_id)
        if record.get("system") != SYSTEM:
            raise ValueError(f"unexpected bridge system: {record.get('system')}")
        invalid = set(record.get("ranking") or []) - canonical_ids.get(example_id, set())
        if invalid:
            raise ValueError(
                f"{example_id} ranking contains non-canonical document IDs: "
                f"{sorted(invalid)}"
            )
    missing = set(canonical_ids) - seen
    if missing:
        raise ValueError(f"bridge output is missing examples: {sorted(missing)[:5]}")


def _validate_base_model_directory(path: Path, model_name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{model_name} directory does not exist: {path}")
    reject_peft_adapter_directory(path, model_name)


def _sentence_split(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", text)
        if item.strip()
    ]


def _milliseconds_to_seconds(value: Any) -> float | None:
    return None if value is None else float(value) / 1000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()

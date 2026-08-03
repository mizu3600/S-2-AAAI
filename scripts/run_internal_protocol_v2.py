from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from scripts.run_protocol_v2_bridge import canonical_rows_to_suite
from s2rag.embedding.text_encoder import TEIBGEEncoder
from s2rag.evaluation.experiment import BenchmarkExperimentRunner
from s2rag.evaluation.internal_baselines import BENCHMARK_METHODS
from s2rag.retrieval.candidates import aggregate_passages
from s2rag.retrieval.local_reranker import TEIBGEReranker


SYSTEM_NAMES = {
    "bm25": "internal:bm25",
    "dense": "internal:dense",
    "reified_fact_hybrid": "s2rag:reified_fact_hybrid",
}


class InternalProtocolRunner(BenchmarkExperimentRunner):
    def _score_method(self, *args, **kwargs) -> dict:
        record = super()._score_method(*args, **kwargs)
        built = args[1] if len(args) > 1 else kwargs["built"]
        candidates = args[4] if len(args) > 4 else kwargs["candidates"]
        record["passage_ranking"] = aggregate_passages(
            candidates,
            built.fact_to_passage,
        )[: self.config.output_k]
        return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BM25, Dense and training-free S2RAG in one shared build."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--suite-sha256", required=True)
    parser.add_argument(
        "--embedding-base-url",
        default="http://127.0.0.1:18080/v1",
    )
    parser.add_argument(
        "--reranker-url",
        default="http://127.0.0.1:18081/rerank",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actual_sha256 = sha256_file(args.input)
    if actual_sha256 != args.suite_sha256:
        raise ValueError(
            f"canonical input SHA256 changed: {actual_sha256} != {args.suite_sha256}"
        )
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    suite = canonical_rows_to_suite(
        rows,
        source=args.input,
        dataset=args.dataset,
        seed=args.seed,
        expected_count=args.expected_count,
    )
    encoder = TEIBGEEncoder(base_url=args.embedding_base_url)
    reranker = TEIBGEReranker(endpoint=args.reranker_url)
    try:
        runner = InternalProtocolRunner(
            methods=BENCHMARK_METHODS,
            generate_for_methods=(),
            encoder=encoder,
            reranker=reranker,
        )
        records = runner.run(suite, args.work_dir, seed=args.seed)
    finally:
        encoder.close()
        reranker.close()

    by_method = {method: [] for method in BENCHMARK_METHODS}
    for record in records:
        method = str(record["system"])
        by_method[method].append(
            {
                "system": SYSTEM_NAMES[method],
                "framework": (
                    "s2rag" if method == "reified_fact_hybrid" else method
                ),
                "example_id": record["example_id"],
                "status": record.get("status", "failed"),
                "ranking": list(
                    dict.fromkeys(record.get("passage_ranking") or [])
                ),
                "index_seconds": milliseconds_to_seconds(
                    record.get("total_preparation_ms")
                ),
                "retrieval_seconds": milliseconds_to_seconds(
                    record.get("retrieval_ms")
                ),
                "error": record.get("error"),
                "dataset": args.dataset,
                "seed": args.seed,
                "suite_sha256": args.suite_sha256,
                "ranking_origin": (
                    "s2rag_training_free_fact_to_document"
                    if method == "reified_fact_hybrid"
                    else f"s2rag_shared_{method}_fact_to_document"
                ),
                "native_candidate_only": True,
                "shared_build": True,
                "model_trace": {
                    "embedding_model": "BAAI/bge-m3",
                    "embedding_service": args.embedding_base_url,
                    "reranker_model": "BAAI/bge-reranker-v2-m3",
                    "reranker_service": args.reranker_url,
                    "graph_model_type": "training_free",
                    "graph_model_id": record.get("graph_model_id"),
                },
                "extraction_coverage": record.get("extraction_coverage"),
            }
        )
    for method, method_rows in by_method.items():
        if len(method_rows) != args.expected_count:
            raise ValueError(
                f"{method} produced {len(method_rows)} rows; "
                f"expected {args.expected_count}"
            )
        output = args.output_root / args.dataset / method / "retrieval.jsonl"
        atomic_write_jsonl(output, method_rows)
        print(f"wrote {len(method_rows)} {method} rows to {output}", flush=True)


def milliseconds_to_seconds(value) -> float | None:
    return None if value is None else float(value) / 1000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

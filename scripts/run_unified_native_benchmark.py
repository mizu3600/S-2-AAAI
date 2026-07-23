from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

from qmshe.benchmark_framework.adapters import (
    SiliconFlowDenseAdapter,
    bm25_trace,
    load_internal_text_unit_traces,
    load_official_traces,
    load_qmsxe_parity_traces,
    load_qmsxe_passage_traces,
)
from qmshe.benchmark_framework.answerer import SharedDeepSeekAnswerer
from qmshe.benchmark_framework.dataset import load_canonical_examples
from qmshe.benchmark_framework.metrics import evaluate_trace
from qmshe.benchmark_framework.report import aggregate, render_report
from qmshe.benchmark_framework.schemas import StandardTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/benchmarks/hotpotqa_official_baselines_288.json")
    )
    parser.add_argument(
        "--official-records",
        type=Path,
        default=Path("reports/official_baselines_hotpot2000/records.json"),
    )
    parser.add_argument(
        "--qmsxe-records", type=Path, default=Path("reports/qmsxe_passage_hotpot2000/records.json")
    )
    parser.add_argument(
        "--qmsxe-parity-records",
        type=Path,
        default=Path("reports/benchmark_v2_trained/records.json"),
    )
    parser.add_argument(
        "--pathrag-internal",
        type=Path,
        default=Path("reports/unified_native_benchmark/pathrag_internal.jsonl"),
    )
    parser.add_argument(
        "--hypergraphrag-internal",
        type=Path,
        default=Path("reports/unified_native_benchmark/hypergraphrag_internal.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/unified_native_benchmark"))
    parser.add_argument("--limit", type=int, default=288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--generation-concurrency", type=int, default=12)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    selected = load_canonical_examples(args.input)[: args.limit]
    examples = {example.example_id: example for example in selected}
    traces = [
        *[bm25_trace(example) for example in selected],
        *[
            trace
            for trace in load_official_traces(args.official_records, examples)
            if trace.example_id in examples
        ],
    ]
    fresh_internal = [
        *load_internal_text_unit_traces(args.pathrag_internal, examples),
        *load_internal_text_unit_traces(args.hypergraphrag_internal, examples),
    ]
    if fresh_internal:
        fresh_keys = {(trace.system, trace.example_id) for trace in fresh_internal}
        traces = [trace for trace in traces if (trace.system, trace.example_id) not in fresh_keys]
        traces.extend(fresh_internal)
    parity = load_qmsxe_parity_traces(args.qmsxe_parity_records, examples, args.seed)
    traces.extend(parity or load_qmsxe_passage_traces(args.qmsxe_records, examples, args.seed))
    if not args.skip_dense:
        dense = SiliconFlowDenseAdapter()
        try:
            for number, example in enumerate(selected, 1):
                try:
                    traces.append(dense.rank(example))
                except Exception as error:
                    traces.append(
                        StandardTrace(
                            system="baseline:dense_bge_m3",
                            example_id=example.example_id,
                            status="error",
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
                if number % 25 == 0:
                    print(f"dense {number}/{len(selected)}", flush=True)
        finally:
            dense.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "generation_cache.jsonl"
    generation_cache = _read_generation_cache(cache_path, args.output_dir / "traces.json")
    if not args.skip_generation:
        answerer = SharedDeepSeekAnswerer(concurrency=args.generation_concurrency)
        lock = asyncio.Lock()

        async def enrich(trace):
            key = _generation_cache_key(trace)
            cached = generation_cache.get(key)
            if cached:
                trace.answer = cached["answer"]
                trace.citations = cached["citations"]
                trace.usage.llm_calls += cached["usage"].get("llm_calls", 0)
                trace.usage.prompt_tokens = (trace.usage.prompt_tokens or 0) + (
                    cached["usage"].get("prompt_tokens") or 0
                )
                trace.usage.completion_tokens = (trace.usage.completion_tokens or 0) + (
                    cached["usage"].get("completion_tokens") or 0
                )
                trace.usage.api_cost_usd = (trace.usage.api_cost_usd or 0) + (
                    cached["usage"].get("api_cost_usd") or 0
                )
                trace.usage.token_count_mode = (
                    "generation_measured_index_unavailable"
                    if "unavailable" in trace.usage.token_count_mode
                    else "measured"
                )
                trace.timing.generation_seconds = cached["generation_seconds"]
                trace.timing.total_seconds = (trace.timing.total_seconds or 0) + cached[
                    "generation_seconds"
                ]
                return
            try:
                before = {
                    "llm_calls": trace.usage.llm_calls,
                    "prompt_tokens": trace.usage.prompt_tokens or 0,
                    "completion_tokens": trace.usage.completion_tokens or 0,
                    "api_cost_usd": trace.usage.api_cost_usd or 0.0,
                }
                await answerer.enrich(examples[trace.example_id], trace)
            except Exception as error:
                trace.metadata["generation_error"] = f"{type(error).__name__}: {error}"
                return
            row = {
                "key": key,
                "answer": trace.answer,
                "citations": trace.citations,
                "usage": {
                    "llm_calls": trace.usage.llm_calls - before["llm_calls"],
                    "prompt_tokens": (trace.usage.prompt_tokens or 0) - before["prompt_tokens"],
                    "completion_tokens": (trace.usage.completion_tokens or 0)
                    - before["completion_tokens"],
                    "api_cost_usd": (trace.usage.api_cost_usd or 0.0) - before["api_cost_usd"],
                    "token_count_mode": "measured",
                },
                "generation_seconds": trace.timing.generation_seconds,
            }
            async with lock:
                with cache_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")

        await asyncio.gather(*(enrich(trace) for trace in traces if trace.status == "success"))
        await answerer.close()
    records = [evaluate_trace(examples[trace.example_id], trace) for trace in traces]
    summary = aggregate(records)
    manifest = {
        "protocol": "unified_native_benchmark_v1",
        "examples": len(selected),
        "candidate_documents_per_example": dict(
            sorted(Counter(len(example.documents) for example in selected).items())
        ),
        "dataset_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "systems": sorted(summary),
        "ks": [1, 2, 5, 10, 20, 30, 40],
        "shared_answer_generator": None if args.skip_generation else "deepseek-chat temperature=0",
        "api_cost_scope": "DeepSeek calls only; SiliconFlow embedding price and unavailable historical index-build calls are excluded.",
        "citation_level": "document",
        "fact_fallback": "document-induced for systems without canonical text-unit IDs",
        "path_fallback": "top-2 induced document path",
        "limitations": [
            *(
                []
                if len(fresh_internal) == 2 * len(selected)
                else [
                    "PathRAG/HyperGraphRAG records without fresh captures use legacy context order."
                ]
            ),
            "Official index-build token usage is unavailable; fresh PathRAG/HyperGraphRAG query tokens and shared generation tokens are measured.",
            "API cost is partial where SiliconFlow embeddings or historical official index-build calls are involved; provider-reported tokens remain visible.",
            *(
                []
                if parity
                else [
                    "QMSxE legacy timing is unavailable until the timed parity rerun is imported."
                ]
            ),
            "This run is the HotpotQA distractor track (normally 10 candidate passages); passage cutoffs above the candidate count are reported for schema completeness but saturate.",
        ],
    }
    (args.output_dir / "traces.json").write_text(
        json.dumps([trace.to_dict() for trace in traces], indent=2), encoding="utf-8"
    )
    (args.output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_report(summary, manifest), encoding="utf-8")
    print(f"wrote {len(records)} records for {len(summary)} systems", flush=True)


def _read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        row["key"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        if row.get("key")
    }


def _generation_cache_key(trace) -> str:
    ranking = json.dumps(trace.document_ranking[:5], ensure_ascii=False, separators=(",", ":"))
    fingerprint = hashlib.sha256(ranking.encode()).hexdigest()[:16]
    return f"{trace.system}:{trace.example_id}:{fingerprint}"


def _read_generation_cache(path: Path, previous_traces_path: Path) -> dict[str, dict]:
    cached = _read_jsonl(path)
    if not previous_traces_path.exists():
        return cached
    # Migrate legacy system/example keys in memory using the ranking that actually
    # produced them. A changed retrieval result must never reuse an old answer.
    for row in json.loads(previous_traces_path.read_text(encoding="utf-8")):
        legacy = f"{row['system']}:{row['example_id']}"
        if legacy not in cached:
            continue
        trace = StandardTrace(
            system=row["system"],
            example_id=row["example_id"],
            status=row["status"],
            document_ranking=row.get("document_ranking", []),
        )
        cached.setdefault(_generation_cache_key(trace), cached[legacy])
    return cached


if __name__ == "__main__":
    asyncio.run(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize 10-question smoke throughput and project 100 questions."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gateway-health-json", type=Path)
    parser.add_argument("--runtime-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="ultradomain")
    parser.add_argument("--smoke-count", type=int, default=10)
    parser.add_argument("--target-count", type=int, default=100)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    systems = {}
    dataset_root = args.output_root / args.dataset
    for path in sorted(dataset_root.glob("*/retrieval.jsonl")):
        rows = read_rows(path)
        framework = path.parent.name
        success = [row for row in rows if row.get("status") == "success"]
        index_seconds = sum(float(row.get("index_seconds") or 0) for row in rows)
        retrieval_seconds = sum(
            float(row.get("retrieval_seconds") or 0) for row in rows
        )
        observed = index_seconds + retrieval_seconds
        systems[framework] = {
            "rows": len(rows),
            "success": len(success),
            "failures": len(rows) - len(success),
            "failure_rate": (
                (len(rows) - len(success)) / len(rows) if rows else 1.0
            ),
            "index_seconds": index_seconds,
            "retrieval_seconds": retrieval_seconds,
            "observed_worker_seconds": observed,
            "questions_per_worker_hour": (
                len(rows) / observed * 3600 if observed else None
            ),
            "projected_100_worker_hours": (
                observed / max(len(rows), 1) * args.target_count / 3600
            ),
            "index_cache_hits": sum(
                bool(row.get("index_cache_hit")) for row in rows
            ),
        }
    gateway = (
        json.loads(args.gateway_health_json.read_text(encoding="utf-8"))
        if args.gateway_health_json
        else None
    )
    runtime = (
        json.loads(args.runtime_audit.read_text(encoding="utf-8"))
        if args.runtime_audit
        else None
    )
    report = {
        "dataset": args.dataset,
        "document_policy": (
            "complete_document_no_prefilter"
            if args.dataset == "ultradomain"
            else "dataset_official_candidate_protocol"
        ),
        "smoke_count": args.smoke_count,
        "target_count": args.target_count,
        "systems": systems,
        "gateway": gateway,
        "runtime": runtime,
        "decision_required": (
            "Confirm complete-document continuation after reviewing this report."
            if args.dataset == "ultradomain"
            else None
        ),
        "automatic_screened_fallback": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote smoke throughput report to {args.output}", flush=True)


if __name__ == "__main__":
    main()

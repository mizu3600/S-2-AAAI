from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qmsxe-summary", type=Path)
    return parser.parse_args()


def recall_at(ranking: list[str], gold: set[str], k: int) -> float:
    return len(set(ranking[:k]) & gold) / max(len(gold), 1)


def complete_at(ranking: list[str], gold: set[str], k: int) -> float:
    return float(gold.issubset(set(ranking[:k])))


def reciprocal_rank(ranking: list[str], gold: set[str]) -> float:
    return next((1 / rank for rank, item in enumerate(ranking, 1) if item in gold), 0.0)


def main() -> None:
    args = parse_args()
    examples = {
        example["example_id"]: example
        for example in json.loads(args.input.read_text(encoding="utf-8"))
    }
    indexed_records = {}
    raw_counts: Counter[str] = Counter()
    raw_success_counts: Counter[str] = Counter()
    raw_error_counts: Counter[str] = Counter()
    duplicate_success_counts: Counter[str] = Counter()
    for path in sorted(args.records_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            framework = record["framework"]
            raw_counts[framework] += 1
            if record.get("error") is None:
                raw_success_counts[framework] += 1
            else:
                raw_error_counts[framework] += 1
            key = (record["framework"], record["example_id"])
            previous = indexed_records.get(key)
            if previous is not None and record.get("error") is not None:
                continue
            if previous is not None and previous.get("error") is None:
                duplicate_success_counts[framework] += 1
            example = examples[record["example_id"]]
            fact_to_document = {fact["fact_id"]: fact["document_id"] for fact in example["facts"]}
            gold = {fact_to_document[fact_id] for fact_id in example["gold_fact_ids"]}
            ranking = record["ranking"]
            indexed_records[key] = {
                **record,
                "gold_document_ids": sorted(gold),
                "ranking_length": len(ranking),
                "recall_at_1": recall_at(ranking, gold, 1),
                "recall_at_2": recall_at(ranking, gold, 2),
                "recall_at_5": recall_at(ranking, gold, 5),
                "recall_at_10": recall_at(ranking, gold, 10),
                "hit_at_1": float(bool(set(ranking[:1]) & gold)),
                "complete_at_2": complete_at(ranking, gold, 2),
                "complete_at_5": complete_at(ranking, gold, 5),
                "mrr": reciprocal_rank(ranking, gold),
            }
    records = list(indexed_records.values())
    metrics = (
        "recall_at_1",
        "recall_at_2",
        "recall_at_5",
        "recall_at_10",
        "hit_at_1",
        "complete_at_2",
        "complete_at_5",
        "mrr",
        "ranking_length",
        "elapsed_seconds",
    )
    summary = {}
    for framework in sorted({record["framework"] for record in records}):
        rows = [record for record in records if record["framework"] == framework]
        successful = [record for record in rows if record["error"] is None]
        summary[framework] = {
            "example_count": len(rows),
            "successful_count": len(successful),
            "error_count": len(rows) - len(successful),
            **{
                f"{metric}_mean": mean(record[metric] for record in successful)
                if successful
                else 0.0
                for metric in metrics
            },
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    audit = {
        framework: {
            "raw_record_count": raw_counts[framework],
            "raw_success_count": raw_success_counts[framework],
            "raw_error_count": raw_error_counts[framework],
            "duplicate_success_count": duplicate_success_counts[framework],
            "selected_record_count": summary[framework]["example_count"],
            "selected_error_count": summary[framework]["error_count"],
        }
        for framework in sorted(raw_counts)
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report_summary = dict(summary)
    if args.qmsxe_summary:
        report_summary.update(json.loads(args.qmsxe_summary.read_text(encoding="utf-8")))
    (args.output_dir / "report.md").write_text(render_report(report_summary), encoding="utf-8")
    print(f"wrote {len(records)} records and {len(summary)} official methods")


def render_report(summary: dict) -> str:
    rows = [
        "# Official Graph RAG passage retrieval comparison",
        "",
        "| Method | R@1 | R@2 | R@5 | R@10 | Hit@1 | Complete@2 | Complete@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, item in summary.items():
        rows.append(
            f"| {method} | {item['recall_at_1_mean']:.4f} | "
            f"{item['recall_at_2_mean']:.4f} | {item['recall_at_5_mean']:.4f} | "
            f"{item['recall_at_10_mean']:.4f} | {item['hit_at_1_mean']:.4f} | "
            f"{item['complete_at_2_mean']:.4f} | {item['complete_at_5_mean']:.4f} | "
            f"{item['mrr_mean']:.4f} |"
        )
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path

from s2rag.evaluation.statistics import summarize


NON_METRIC_FIELDS = {
    "dataset",
    "split",
    "example_id",
    "system",
    "query_type",
    "status",
    "error",
    "ranking_origin",
    "answer",
    "citation_level",
    "generation_protocol",
    "generation_trace",
    "shared_model_trace",
    "retrieval_protocol",
    "end_to_end_llm_extraction",
    "graph_model_type",
    "graph_model_id",
    "citation_capability",
    "citation_source",
    "citation_status",
    "unmapped_citation_ids",
    "passage_aggregation",
    "per_channel_candidate_counts",
    "reranker_model_sha256",
    "unmapped_ranking_ids",
    "retrieval_evidence_fact_ids",
    "retrieval_evidence_sentence_ids",
    "generated_fact_citations",
    "generated_fact_citation_sentence_ids",
    "generated_passage_citations",
}


def aggregate(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["system"]].append(record)
    metric_names = (
        sorted(
            {
                key
                for record in records
                for key, value in record.items()
                if (
                    (isinstance(value, (int, float)) or value is None)
                    and key not in NON_METRIC_FIELDS
                    and key != "seed"
                )
            }
        )
        if records
        else []
    )
    return {
        system: {
            metric: summarize(
                [float(item[metric]) for item in rows if item.get(metric) is not None]
            )
            for metric in metric_names
        }
        for system, rows in grouped.items()
    }


def render_markdown(summary: dict, metadata: dict) -> str:
    lines = [
        f"# Benchmark report: {metadata.get('dataset', 'unknown')}",
        "",
        f"- split: `{metadata.get('split', 'unknown')}`",
        f"- examples: `{metadata.get('expected_examples', metadata.get('examples', 0))}`",
        f"- seed: `{metadata.get('seed', 42)}`",
        f"- protocol: `{metadata.get('protocol', 'unspecified')}`",
        f"- output/rerank-input/context K: "
        f"`{metadata.get('output_k', 'N/A')}/"
        f"{metadata.get('rerank_input_k', metadata.get('candidate_k', 'N/A'))}/"
        f"{metadata.get('context_k', 'N/A')}`",
        f"- generation protocol: `{metadata.get('generation_protocol', 'N/A')}`",
        "",
        "## Retrieval",
        "",
        "| System | Passage R@5 | Passage strict P@5 | Fact R@5 | "
        "Fact strict P@5 | Fact returned P@5 | Fact MRR@20 | Fact nDCG@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, metrics in summary.items():
        lines.append(
            f"| {system} | {_value(metrics, 'passage_recall_at_5')} | "
            f"{_value(metrics, 'passage_precision_at_5')} | "
            f"{_value(metrics, 'fact_recall_at_5')} | "
            f"{_value(metrics, 'fact_precision_at_5')} | "
            f"{_value(metrics, 'fact_returned_precision_at_5')} | "
            f"{_value(metrics, 'fact_mrr_at_20')} | "
            f"{_value(metrics, 'fact_ndcg_at_5')} |"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "| System | Retrieved passage F1 | Retrieved fact F1 | "
            "Generated passage citation F1 | Generated fact citation F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for system, metrics in summary.items():
        lines.append(
            f"| {system} | "
            f"{_value(metrics, 'retrieval_evidence_passage_f1')} | "
            f"{_value(metrics, 'retrieval_evidence_fact_f1')} | "
            f"{_value(metrics, 'generated_passage_citation_f1')} | "
            f"{_value(metrics, 'generated_fact_citation_f1')} |"
        )
    lines.extend(
        [
            "",
            "## Generation",
            "",
            "| System | Answer EM | Answer F1 | Joint EM | Joint F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for system, metrics in summary.items():
        lines.append(
            f"| {system} | {_value(metrics, 'answer_em')} | "
            f"{_value(metrics, 'answer_f1')} | "
            f"{_value(metrics, 'joint_em')} | "
            f"{_value(metrics, 'joint_f1')} |"
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| System | Extraction | Mapping | Failure rate | "
            "Expected | Scored | Missing | Failed | Unavailable |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for system, metrics in summary.items():
        counts = metadata.get("audit_by_system", {}).get(system, {})
        lines.append(
            f"| {system} | {_value(metrics, 'extraction_coverage')} | "
            f"{_value(metrics, 'mapping_coverage')} | "
            f"{_failure_value(metrics)} | "
            f"{counts.get('expected', 'N/A')} | "
            f"{counts.get('scored', 'N/A')} | "
            f"{counts.get('missing', 'N/A')} | "
            f"{counts.get('failed', 'N/A')} | "
            f"{counts.get('unavailable', 'N/A')} |"
        )
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| System | First stage | Rerank | Generation | Query total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for system, metrics in summary.items():
        lines.append(
            f"| {system} | {_value(metrics, 'first_stage_ms', suffix=' ms')} | "
            f"{_value(metrics, 'shared_rerank_ms', suffix=' ms')} | "
            f"{_value(metrics, 'generation_ms', suffix=' ms')} | "
            f"{_value(metrics, 'query_total_ms', suffix=' ms')} |"
        )
    return "\n".join(lines) + "\n"


def write_report(records: list[dict], output_dir: str | Path, metadata: dict) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _validate_expected_examples(records, metadata)
    summary = aggregate(records)
    categorized = _categorize_summary(summary)
    successful = [record for record in records if record.get("status") == "success"]
    successful_only_summary = aggregate(successful)
    failed = [
        record for record in records if record.get("status") in {"missing", "failed", "unscorable"}
    ]
    by_system: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_system[record["system"]].append(record)
    per_system_audit = {}
    for system, rows in by_system.items():
        unavailable = sum(
            not bool(row.get("passage_ranking_available", True))
            and not bool(row.get("fact_ranking_available", True))
            for row in rows
        )
        per_system_audit[system] = {
            "expected": metadata.get("expected_examples", len(rows)),
            "received": sum(not row.get("result_missing", False) for row in rows),
            "scored": sum(row.get("status") == "success" for row in rows),
            "missing": sum(row.get("status") == "missing" for row in rows),
            "failed": sum(row.get("status") in {"failed", "unscorable"} for row in rows),
            "unavailable": unavailable,
        }
    audit = {
        "expected_records": len(records),
        "successful_records": len(successful),
        "failed_records": len(failed),
        "failure_rate": len(failed) / len(records) if records else 0.0,
        "systems": per_system_audit,
    }
    (output / "records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "summary_v2.json").write_text(
        json.dumps(categorized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "successful_only_summary.json").write_text(
        json.dumps(successful_only_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_metadata = {**metadata, "audit_by_system": per_system_audit}
    (output / "report.md").write_text(
        render_markdown(summary, report_metadata),
        encoding="utf-8",
    )
    return summary


def _value(metrics: dict, name: str, suffix: str = "") -> str:
    values = metrics.get(name, {})
    mean_value = values.get("mean")
    if mean_value is None:
        return "N/A"
    low, high, count = (
        values.get("ci95_low"),
        values.get("ci95_high"),
        values.get("n", 0),
    )
    if low is None or high is None:
        return f"{mean_value:.4f}{suffix} (n={count})"
    return f"{mean_value:.4f}{suffix} [{low:.4f}, {high:.4f}] (n={count})"


def _failure_value(metrics: dict) -> str:
    values = metrics.get("result_failed", {})
    rate = values.get("mean")
    count = values.get("n", 0)
    if rate is None:
        return "N/A"
    failed = round(rate * count)
    return f"{100 * rate:.2f}% ({failed}/{count})"


def _categorize_summary(summary: dict) -> dict:
    categories = {
        "quality_metrics": {},
        "coverage_metrics": {},
        "efficiency_metrics": {},
        "audit": {},
    }
    for system, metrics in summary.items():
        for category in categories:
            categories[category][system] = {}
        for name, values in metrics.items():
            if name.endswith("_ms") or name.endswith("_count"):
                category = "efficiency_metrics"
            elif "coverage" in name or name.endswith("_available"):
                category = "coverage_metrics"
            elif name.startswith("result_") or name in {
                "protocol_mismatch",
                "shared_model_protocol_mismatch",
                "generation_failed",
                "no_gold_evidence",
                "seed_complete",
            }:
                category = "audit"
            else:
                category = "quality_metrics"
            categories[category][system][name] = values
    return categories


def _validate_expected_examples(records: list[dict], metadata: dict) -> None:
    expected_count = metadata.get("expected_examples")
    expected_hash = metadata.get("expected_example_ids_sha256")
    if expected_count is None and expected_hash is None:
        return
    by_system: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_system[record["system"]].add(record["example_id"])
    for system, example_ids in by_system.items():
        if expected_count is not None and len(example_ids) != expected_count:
            raise ValueError(
                f"{system} scored {len(example_ids)} unique examples; expected {expected_count}"
            )
        actual_hash = hashlib.sha256("\n".join(sorted(example_ids)).encode()).hexdigest()
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"{system} example ID hash does not match the manifest")

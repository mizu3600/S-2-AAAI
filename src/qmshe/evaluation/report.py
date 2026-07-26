from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from qmshe.evaluation.statistics import summarize


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
    "unmapped_ranking_ids",
    "retrieval_evidence_fact_ids",
    "generated_fact_citations",
    "generated_passage_citations",
}


def aggregate(records: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["system"]].append(record)
    metric_names = sorted(
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
    ) if records else []
    return {
        system: {
            metric: summarize([float(item[metric]) for item in rows if item.get(metric) is not None])
            for metric in metric_names
        }
        for system, rows in grouped.items()
    }


def render_markdown(summary: dict, metadata: dict) -> str:
    lines = [
        f"# Benchmark report: {metadata.get('dataset', 'unknown')}",
        "",
        f"- split: `{metadata.get('split', 'unknown')}`",
        f"- examples: `{metadata.get('examples', 0)}`",
        f"- seed: `{metadata.get('seed', 42)}`",
        f"- protocol: `{metadata.get('protocol', 'unspecified')}`",
        f"- output/candidate/context K: "
        f"`{metadata.get('output_k', 'N/A')}/"
        f"{metadata.get('candidate_k', 'N/A')}/"
        f"{metadata.get('context_k', 'N/A')}`",
        f"- generation protocol: `{metadata.get('generation_protocol', 'N/A')}`",
        "",
        "| System | Passage Recall@5 | Fact Recall@5 | Fact MRR | Answer F1 | "
        "Generated Fact Citation F1 | Joint F1 | Mapping | Failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system, metrics in summary.items():
        def value(name: str) -> str:
            mean_value = metrics.get(name, {}).get("mean")
            return "N/A" if mean_value is None else f"{mean_value:.4f}"

        lines.append(
            f"| {system} | {value('passage_recall_at_5')} | {value('fact_recall_at_5')} | "
            f"{value('fact_mrr')} | {value('answer_f1')} | "
            f"{value('generated_fact_citation_f1')} | "
            f"{value('joint_f1')} | {value('mapping_coverage')} | "
            f"{value('result_failed')} |"
        )
    return "\n".join(lines) + "\n"


def write_report(records: list[dict], output_dir: str | Path, metadata: dict) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate(records)
    (output / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "report.md").write_text(render_markdown(summary, metadata), encoding="utf-8")
    return summary

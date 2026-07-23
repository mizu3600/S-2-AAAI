from __future__ import annotations

from collections import defaultdict
from statistics import mean

from qmshe.benchmark_framework.metrics import KS


def aggregate(records: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["system"]].append(record)
    output = {}
    ignored = {"example_id", "system", "status", "error", "metadata"}
    for system, rows in sorted(grouped.items()):
        summary = {
            "example_count": len(rows),
            "successful_count": sum(row["status"] == "success" for row in rows),
            "success_rate": mean(row["success"] for row in rows),
            "ranking_origins": sorted({row["ranking_origin"] for row in rows}),
            "path_origins": sorted({row["path_origin"] for row in rows}),
            "token_count_modes": sorted({row["token_count_mode"] for row in rows}),
            "timing_scopes": sorted({row["timing_scope"] for row in rows}),
        }
        keys = sorted(set().union(*(row.keys() for row in rows)) - ignored)
        for key in keys:
            values = [row.get(key) for row in rows]
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            if numeric:
                summary[f"{key}_mean"] = mean(numeric)
                summary[f"{key}_sum"] = sum(numeric)
                summary[f"{key}_coverage"] = len(numeric) / len(rows)
        output[system] = summary
    return output


def render_report(summary: dict[str, dict], manifest: dict) -> str:
    lines = ["# Unified Native RAG Benchmark", ""]
    for level in ("passage", "fact"):
        columns = ["Method", "MRR"]
        for k in KS:
            columns.extend([f"R@{k}", f"Hit/Acc@{k}", f"Complete@{k}"])
        lines.extend(
            [
                f"## {level.title()} retrieval",
                "",
                "| " + " | ".join(columns) + " |",
                "| "
                + " | ".join("---" if index == 0 else "---:" for index in range(len(columns)))
                + " |",
            ]
        )
        for system, row in summary.items():
            values = [system, _fmt(row.get(f"{level}_mrr_mean"))]
            for k in KS:
                values.extend(
                    [
                        _fmt(row.get(f"{level}_recall_at_{k}_mean")),
                        _fmt(row.get(f"{level}_hit_at_{k}_mean")),
                        _fmt(row.get(f"{level}_complete_at_{k}_mean")),
                    ]
                )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "## Path, answer, citation and joint",
            "",
            "| Method | Path EM | Path P | Path R | Path F1 | Answer EM | Answer P | Answer R | Answer F1 | Citation EM | Citation P | Citation R | Citation F1 | Joint EM | Joint P | Joint R | Joint F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for system, row in summary.items():
        metrics = (
            "path_em",
            "path_precision",
            "path_recall",
            "path_f1",
            "answer_em",
            "answer_precision",
            "answer_recall",
            "answer_f1",
            "citation_em",
            "citation_precision",
            "citation_recall",
            "citation_f1",
            "joint_em",
            "joint_precision",
            "joint_recall",
            "joint_f1",
        )
        lines.append(
            "| "
            + " | ".join([system, *(_fmt(row.get(f"{metric}_mean")) for metric in metrics)])
            + " |"
        )
    lines.extend(
        [
            "",
            "## Efficiency, usage and reliability",
            "",
            "| Method | Success | Total s/query | Index s | Retrieval s | Generation s | LLM calls | Embedding calls | Reranker calls | Observed tokens | Prompt tokens | Completion tokens | Embedding tokens | API cost USD/query | Timing scope | Token scope |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for system, row in summary.items():
        values = [
            system,
            _fmt(row.get("success_rate")),
            _fmt(row.get("total_seconds_mean")),
            _fmt(row.get("index_seconds_mean")),
            _fmt(row.get("retrieval_seconds_mean")),
            _fmt(row.get("generation_seconds_mean")),
            _fmt(row.get("llm_calls_mean")),
            _fmt(row.get("embedding_calls_mean")),
            _fmt(row.get("reranker_calls_mean")),
            _fmt(row.get("total_model_tokens_mean")),
            _fmt(row.get("prompt_tokens_mean")),
            _fmt(row.get("completion_tokens_mean")),
            _fmt(row.get("embedding_tokens_mean")),
            _fmt(row.get("api_cost_usd_mean"), 6),
            ", ".join(row.get("timing_scopes", [])),
            ", ".join(row.get("token_count_modes", [])),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        ["", "## Manifest", "", "```json", __import__("json").dumps(manifest, indent=2), "```", ""]
    )
    return "\n".join(lines)


def _fmt(value, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"

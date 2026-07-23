from __future__ import annotations

from collections import defaultdict
from statistics import mean


def aggregate(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        suite = record.get("suite", "hotpotqa_dev")
        grouped[(suite, record["system"], record["candidate_count"])].append(record)
    output = []
    for (suite, system, candidate_count), rows in sorted(grouped.items()):
        metrics = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "candidate_count"]
        output.append({
            "suite": suite, "system": system,
            "candidate_count": candidate_count,
            "examples": len(rows),
            **{metric: mean(float(row[metric]) for row in rows) for metric in metrics},
        })
    return output


def paired_comparisons(
    records: list[dict], bootstrap_samples: int = 2000, seed: int = 42,
) -> list[dict]:
    import numpy as np

    metrics = (
        "fact_recall_at_10", "fact_recall_at_40", "passage_recall_at_10",
        "path_f1", "answer_f1",
    )
    indexed = defaultdict(dict)
    for row in records:
        key = (row.get("suite", "hotpotqa_dev"), row["candidate_count"], row["system"])
        indexed[key][row["example_id"]] = row
    output, rng = [], np.random.default_rng(seed)
    groups = sorted({(row.get("suite", "hotpotqa_dev"), row["candidate_count"]) for row in records})
    for suite, candidate_count in groups:
        systems = sorted(
            system for group_suite, count, system in indexed
            if (group_suite, count) == (suite, candidate_count)
        )
        if len(systems) < 2:
            continue
        baseline = systems[0]
        for challenger in systems[1:]:
            baseline_rows = indexed[(suite, candidate_count, baseline)]
            challenger_rows = indexed[(suite, candidate_count, challenger)]
            common = sorted(set(baseline_rows) & set(challenger_rows))
            for metric in metrics:
                differences = np.asarray([
                    challenger_rows[example_id][metric] - baseline_rows[example_id][metric]
                    for example_id in common
                ], dtype=float)
                samples = rng.choice(
                    differences, size=(bootstrap_samples, len(differences)), replace=True
                ).mean(axis=1)
                output.append({
                    "suite": suite, "candidate_count": candidate_count,
                    "baseline": baseline, "challenger": challenger, "metric": metric,
                    "examples": len(common), "mean_difference": float(differences.mean()),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "win_rate": float((differences > 0).mean()),
                    "tie_rate": float((differences == 0).mean()),
                })
    return output


def render_markdown(summary: list[dict], manifest: dict) -> str:
    columns = [
        "suite", "system", "candidate_count", "examples", "fact_hit_at_1", "fact_mrr",
        "fact_recall_at_10", "fact_recall_at_40",
        "passage_recall_at_10", "path_f1", "answer_em", "answer_f1", "citation_f1", "joint_f1",
    ]
    lines = [
        "# QMSxE-RAG Benchmark V2 Results", "",
        "This report is produced by the independent protocol-first chain. Existing production pipelines are unchanged.", "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Protocol manifest", "", "```json", __import__("json").dumps(manifest, indent=2), "```", ""])
    return "\n".join(lines)

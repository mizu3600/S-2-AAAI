from __future__ import annotations

import json
from pathlib import Path

import typer


def main(
    input_dir: Path = typer.Option(Path("data/experiments/configured")),
    output: Path = typer.Option(Path("data/experiments/configured/all_summary.json")),
) -> None:
    summaries = {}
    for summary_path in sorted(input_dir.glob("*/summary.json")):
        summaries[summary_path.parent.name] = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = output.with_suffix(".md")
    lines = [
        "# Benchmark suites summary",
        "",
        "| Suite | System | Fact Recall@5 | Answer F1 | Citation F1 | Joint F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for suite, systems in summaries.items():
        for system, metrics in systems.items():
            def value(name: str) -> str:
                mean_value = metrics.get(name, {}).get("mean")
                return "N/A" if mean_value is None else f"{mean_value:.4f}"

            lines.append(
                f"| {suite} | {system} | {value('fact_recall_at_5')} | "
                f"{value('answer_f1')} | {value('generated_fact_citation_f1')} | "
                f"{value('joint_f1')} |"
            )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"aggregated {len(summaries)} suites into {output}")


if __name__ == "__main__":
    typer.run(main)

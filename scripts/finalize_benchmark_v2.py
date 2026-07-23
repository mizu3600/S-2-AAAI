from __future__ import annotations

import json
from pathlib import Path

import typer

from qmshe.benchmark_v2.report import aggregate, paired_comparisons, render_markdown


def main(output_dir: Path = typer.Option(...)) -> None:
    records = json.loads((output_dir / "records.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = aggregate(records)
    comparisons = paired_comparisons(records)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(summary, manifest), encoding="utf-8")
    typer.echo(f"finalized {len(records)} records in {output_dir}")


if __name__ == "__main__":
    typer.run(main)

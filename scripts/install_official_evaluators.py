from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer


def main(
    evaluator: str = typer.Option("all", help="Pinned evaluator key, or 'all'"),
    root: Path = typer.Option(Path("third_party/official_evaluators")),
    lock_path: Path = typer.Option(Path("third_party/official_evaluators.lock.json")),
) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected = lock if evaluator == "all" else {evaluator: lock[evaluator]}
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in selected.items():
        target = root / name
        if target.exists():
            typer.echo(f"{name}: already exists at {target}")
            continue
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", spec["repository"], str(target)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", spec["commit"]],
            check=True,
        )
        script = target / spec["script"]
        if not script.is_file():
            raise FileNotFoundError(f"{name} evaluator missing expected script: {script}")
        typer.echo(f"{name}: installed {script} at {spec['commit']}")


if __name__ == "__main__":
    typer.run(main)

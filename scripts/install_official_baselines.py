from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer


def main(
    baseline: str = typer.Option("all", help="Pinned baseline key, or 'all'"),
    root: Path = typer.Option(Path("third_party/official_baselines")),
    lock_path: Path = typer.Option(Path("third_party/official_baselines.lock.json")),
) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected = lock if baseline == "all" else {baseline: lock[baseline]}
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in selected.items():
        target = root / name
        if target.exists():
            typer.echo(f"{name}: already exists at {target}")
            continue
        subprocess.run(["git", "clone", "--depth", "1", spec["repository"], str(target)], check=True)
        if spec["commit"] != "latest":
            subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", spec["commit"]], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]], check=True)
        typer.echo(f"{name}: installed at {target}")


if __name__ == "__main__":
    typer.run(main)

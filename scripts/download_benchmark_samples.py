import json
from pathlib import Path

import httpx
import typer

ENDPOINTS = {
    "hotpotqa": (
        "https://datasets-server.huggingface.co/rows",
        {"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation"},
    ),
    "qasper": (
        "https://datasets-server.huggingface.co/rows",
        {"dataset": "allenai/qasper", "config": "qasper", "split": "validation"},
    ),
}


def main(
    output_dir: Path = typer.Option(Path("data/benchmarks")),
    rows: int = typer.Option(5, min=1, max=100),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name, (url, base_params) in ENDPOINTS.items():
            params = {**base_params, "offset": 0, "length": rows}
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            compact = {"rows": [item.get("row", item) for item in payload.get("rows", [])]}
            target = output_dir / f"{name}_sample.json"
            target.write_text(json.dumps(compact, ensure_ascii=False), encoding="utf-8")
            typer.echo(f"downloaded {len(compact['rows'])} {name} rows to {target}")


if __name__ == "__main__":
    typer.run(main)

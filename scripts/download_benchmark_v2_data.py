from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import typer


FILES = {
    "dev": (
        "hotpot_dev_distractor_v1.json",
        "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json",
        "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
    ),
    "train": (
        "hotpot_train_v1.1.json",
        "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_train_v1.1.json",
        None,
    ),
}


def main(
    split: str = typer.Option("dev", help="dev or train"),
    output_dir: Path = typer.Option(Path("data/benchmarks")),
) -> None:
    if split not in FILES:
        raise typer.BadParameter("split must be dev or train")
    filename, url, expected = FILES[split]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    typer.echo(f"downloading {url}")
    urllib.request.urlretrieve(url, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if expected and digest != expected:
        destination.unlink()
        raise RuntimeError(f"checksum mismatch: expected {expected}, got {digest}")
    typer.echo(f"wrote {destination} sha256={digest}")


if __name__ == "__main__":
    typer.run(main)

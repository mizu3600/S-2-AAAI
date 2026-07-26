from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from qmshe.benchmarks.adapters import _read_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible multi-seed benchmark suites")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/configured_suites"))
    parser.add_argument("--datasets", default="hotpotqa")
    parser.add_argument("--seeds", default="13,42,73")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=2026)
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "single_pipeline_multi_seed_v1",
        "seeds": seeds,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "datasets": datasets,
        "suites": {},
    }
    for dataset in datasets:
        source = _find_source(args.input_dir, dataset)
        records = _read_records(source)
        rng = random.Random(args.sample_seed)
        selected = list(records)
        rng.shuffle(selected)
        selected = selected[: args.sample_size]
        for seed in seeds:
            target = args.output_dir / f"{dataset}_seed_{seed}.json"
            target.write_text(
                json.dumps({"rows": selected}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest["suites"][f"{dataset}_seed_{seed}"] = {
                "dataset": dataset,
                "seed": seed,
                "source": str(source),
                "path": str(target),
                "examples": len(selected),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _find_source(input_dir: Path, dataset: str) -> Path:
    candidates = [
        input_dir / f"{dataset}.json",
        input_dir / f"{dataset}_sample.json",
        input_dir / f"{dataset}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no input file found for {dataset} under {input_dir}")


if __name__ == "__main__":
    main()

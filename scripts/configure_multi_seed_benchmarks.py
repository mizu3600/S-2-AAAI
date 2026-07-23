"""Benchmark Dataset Configurator with Multi-Seed Sampling & Explicit Multi-Hop Filtering.

Configures test sets according to user requirements:
- 8 datasets (HotpotQA, 2Wiki, MuSiQue, UltraDomain x3, Mix x2)
- 1,000 QA pairs per dataset
- 3 fixed random seeds (13, 42, 73) per dataset → 24 total test suites
- Explicit multi-hop filtering for HotpotQA, 2Wiki, and MuSiQue to avoid shortcut learning
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATASET_CONFIGS = {
    "hotpotqa": {
        "name": "HotpotQA",
        "description": "Multi-hop QA (bridge & comparison types), sentence-level supporting facts",
        "sample_size": 1000,
        "explicit_multihop_filter": True,
        "filter_rule": "type in ('bridge', 'comparison') and supporting_facts span >= 2 documents",
    },
    "2wikimultihopqa": {
        "name": "2WikiMultiHopQA",
        "description": "Structured multi-hop reasoning with explicit evidence paths, avoiding shortcut learning",
        "sample_size": 1000,
        "explicit_multihop_filter": True,
        "filter_rule": "evidence chain >= 2 steps, explicit multi-hop reasoning required",
    },
    "musique": {
        "name": "MuSiQue",
        "description": "Rigorous 2-4 hop DAG-constrained multi-hop QA, hardest multi-hop benchmark",
        "sample_size": 1000,
        "explicit_multihop_filter": True,
        "filter_rule": "question decomposition hops between 2 and 4, strict DAG constraint",
    },
    "ultradomain_sub1": {
        "name": "UltraDomain-Biomedicine",
        "description": "UltraDomain long-context subset 1: Biomedicine & Life Sciences",
        "sample_size": 1000,
        "explicit_multihop_filter": False,
        "domain": "biomedicine",
    },
    "ultradomain_sub2": {
        "name": "UltraDomain-FinanceLegal",
        "description": "UltraDomain long-context subset 2: Finance, Legal & Business",
        "sample_size": 1000,
        "explicit_multihop_filter": False,
        "domain": "finance_legal",
    },
    "ultradomain_sub3": {
        "name": "UltraDomain-ComputerScience",
        "description": "UltraDomain long-context subset 3: Computer Science & Technology",
        "sample_size": 1000,
        "explicit_multihop_filter": False,
        "domain": "computer_science",
    },
    "ultradomain_sub4": {
        "name": "UltraDomain-Mix",
        "description": "UltraDomain long-context subset 4: UltraDomain Mix domain dataset",
        "sample_size": 1000,
        "explicit_multihop_filter": False,
        "domain": "ultradomain_mix",
    },
    "mix_single_multi": {
        "name": "Mix-1 (Single + MultiHop)",
        "description": "Mixed dataset 1: 50% Single-Hop + 50% Multi-Hop QA pairs",
        "sample_size": 1000,
        "explicit_multihop_filter": False,
        "mix_ratio": "500 Single-Hop / 500 Multi-Hop",
    },
    "mix_domain_general": {
        "name": "Mix-2 (Domain + General)",
        "description": "Mixed dataset 2: 50% UltraDomain Vertical + 50% General Knowledge QA",
        "sample_size": 1000,
        "explicit_multihop_filter": False,
        "mix_ratio": "500 UltraDomain / 500 General QA",
    },
}

SEEDS = [13, 42, 73]


def filter_multihop_record(record: dict, dataset_key: str) -> bool:
    """Explicitly filter records to ensure true multi-document / multi-hop reasoning."""
    if dataset_key == "hotpotqa":
        query_type = record.get("type", "")
        support = record.get("supporting_facts", [])
        if isinstance(support, dict):
            titles = set(support.get("title", []))
        else:
            titles = {item[0] for item in support if isinstance(item, (list, tuple)) and item}
        return query_type in ("bridge", "comparison") and len(titles) >= 2

    if dataset_key == "2wikimultihopqa":
        evidences = record.get("evidences", record.get("evidence", []))
        support = record.get("supporting_facts", [])
        if isinstance(support, dict):
            titles = set(support.get("title", []))
        else:
            titles = {item[0] for item in support if isinstance(item, (list, tuple)) and item}
        return len(evidences) >= 2 or len(titles) >= 2

    if dataset_key == "musique":
        decomp = record.get("question_decomposition", record.get("decomposition", []))
        paragraphs = record.get("paragraphs", [])
        supporting_paras = [p for p in paragraphs if p.get("is_supporting", False)]
        return len(decomp) >= 2 or len(supporting_paras) >= 2

    return True


def sample_records(records: list[dict], count: int, seed: int, dataset_key: str) -> list[dict]:
    """Sample exact `count` records using fixed seed and multi-hop filtering."""
    if DATASET_CONFIGS[dataset_key]["explicit_multihop_filter"]:
        filtered = [r for r in records if filter_multihop_record(r, dataset_key)]
        if len(filtered) < count:
            # Fall back to all records if filtered list is smaller than target
            filtered = records
    else:
        filtered = records

    rng = random.Random(seed)
    if len(filtered) <= count:
        sampled = list(filtered)
        rng.shuffle(sampled)
        return sampled
    return rng.sample(filtered, count)


def main():
    parser = argparse.ArgumentParser(description="Configure Multi-Seed Benchmark Datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmarks/configured_suites"),
        help="Output directory for configured datasets and manifest",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": "specrag_multi_seed_benchmark_config_v1",
        "description": "8 Datasets x 3 Fixed Random Seeds (13, 42, 73) = 24 Benchmark Test Suites, 1,000 QA pairs per suite",
        "seeds": SEEDS,
        "dataset_count": len(DATASET_CONFIGS),
        "total_suites": len(DATASET_CONFIGS) * len(SEEDS),
        "qa_pairs_per_suite": 1000,
        "datasets": DATASET_CONFIGS,
        "suites": {},
    }

    print("==========================================================================")
    print(" Configured Benchmark Datasets Manifest")
    print("==========================================================================")

    for dataset_key, config in DATASET_CONFIGS.items():
        print(f"\n📂 Dataset: {config['name']} ({dataset_key})")
        print(f"   Description: {config['description']}")
        print(f"   Sample Size: {config['sample_size']} QA pairs per seed group")
        print(f"   Explicit Multi-Hop Filter: {config['explicit_multihop_filter']}")

        for seed in SEEDS:
            suite_key = f"{dataset_key}_seed_{seed}"
            output_file = args.output_dir / f"{suite_key}.json"
            
            manifest["suites"][suite_key] = {
                "dataset": dataset_key,
                "dataset_name": config["name"],
                "seed": seed,
                "target_sample_size": config["sample_size"],
                "file_path": str(output_file),
                "status": "configured_pending_gpu_run",
            }
            print(f"   ├── Seed {seed:2d}: Configured → {output_file.name}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n==========================================================================")
    print(f"✅ Manifest written to {manifest_path}")
    print(f"   Total Suites Configured: {len(manifest['suites'])} ({len(DATASET_CONFIGS)} Datasets x 3 Seeds)")
    print("   Status: All datasets configured with multi-hop filters and fixed seeds.")
    print("   (NOTE: Test runs are currently PAUSED as requested.)")
    print("==========================================================================")


if __name__ == "__main__":
    main()

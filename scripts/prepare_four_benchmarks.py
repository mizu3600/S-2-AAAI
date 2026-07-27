from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from s2rag.benchmarks.adapters import _read_records


OFFICIAL_SOURCES = {
    "hotpotqa": "https://hotpotqa.github.io/",
    "musique": "https://github.com/stonybrooknlp/musique",
    "2wikimultihopqa": "https://github.com/Alab-NII/2wikimultihop",
    "ultradomain": "https://huggingface.co/datasets/TommyChien/UltraDomain",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare fixed 1,000-question S2RAG benchmark samples"
    )
    parser.add_argument("--hotpot-source", type=Path, required=True)
    parser.add_argument("--musique-source", type=Path, required=True)
    parser.add_argument("--two-wiki-source", type=Path, required=True)
    parser.add_argument("--two-wiki-alias-source", type=Path)
    parser.add_argument("--ultradomain-source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmarks"),
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_size != 1000:
        parser.error("the benchmark protocol requires --sample-size 1000")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specifications = {
        "hotpotqa": (
            args.hotpot_source,
            _hotpot_distractor_multihop,
            "distractor_context_and_at_least_two_supporting_passages",
        ),
        "musique": (
            args.musique_source,
            _musique_multihop,
            "answerable_and_decomposition_hops_gte_2",
        ),
        "2wikimultihopqa": (
            args.two_wiki_source,
            _explicit_multihop,
            "at_least_two_supporting_passages",
        ),
    }
    manifest = {
        "protocol": "s2rag_four_benchmarks_1000_v2",
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "datasets": {},
    }
    for dataset, (source, predicate, hop_filter) in specifications.items():
        records = [row for row in _read_records(source) if predicate(row)]
        selected = _fixed_sample(
            records,
            args.sample_size,
            args.sample_seed,
            dataset,
        )
        if dataset == "hotpotqa":
            selected = [
                {
                    **row,
                    "_s2rag_benchmark_config": "distractor",
                    "_s2rag_corpus_scope": "per_question_candidate_passages",
                }
                for row in selected
            ]
        elif dataset == "2wikimultihopqa":
            alias_source = args.two_wiki_alias_source or source.parent / "id_aliases.json"
            if not alias_source.is_file():
                raise FileNotFoundError(
                    "2WikiMultiHopQA official v1.1 scoring requires id_aliases.json; "
                    "pass --two-wiki-alias-source"
                )
            selected = _add_2wiki_answer_aliases(selected, alias_source)
        target = args.output_dir / f"{dataset}_1000.jsonl"
        _write_jsonl(target, selected)
        entry = _manifest_entry(
            target,
            source,
            len(records),
            hop_filter,
        )
        if dataset == "hotpotqa":
            entry.update(
                {
                    "benchmark_config": "distractor",
                    "corpus_scope": "per_question_candidate_passages",
                    "official_evaluator": "hotpot_evaluate_v1.py",
                }
            )
        elif dataset == "musique":
            entry.update(
                {
                    "benchmark_config": "answerable_dev",
                    "corpus_scope": "per_question_candidate_paragraphs",
                    "official_evaluator": "evaluate_v1.0.py",
                }
            )
        else:
            entry.update(
                {
                    "benchmark_config": "dev_ids_april7",
                    "corpus_scope": "per_question_candidate_passages",
                    "official_evaluator": "2wikimultihop_evaluate_v1.1.py",
                    "answer_alias_source": str(alias_source),
                    "answer_alias_source_sha256": _sha256(alias_source),
                }
            )
        manifest["datasets"][dataset] = entry

    ultra_rows, ultra_sources = _stratified_ultradomain_sample(
        args.ultradomain_source_dir,
        args.sample_size,
        args.sample_seed,
    )
    ultra_target = args.output_dir / "ultradomain_1000.jsonl"
    _write_jsonl(ultra_target, ultra_rows)
    manifest["datasets"]["ultradomain"] = {
        **_manifest_entry(
            ultra_target,
            args.ultradomain_source_dir,
            sum(_line_count(path) for path in ultra_sources),
            "not_available",
        ),
        "sampling": "equal_domain_stratified",
        "benchmark_config": "official_19_domain_files",
        "corpus_scope": "per_example_long_document",
        "official_evaluator": None,
        "source_files": [str(path) for path in ultra_sources],
    }
    manifest["official_sources"] = OFFICIAL_SOURCES
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(manifest_path)


def _explicit_multihop(row: dict) -> bool:
    supporting = row.get("supporting_facts", [])
    if isinstance(supporting, dict):
        titles = supporting.get("title", [])
    else:
        titles = [
            item[0]
            for item in supporting
            if isinstance(item, (list, tuple)) and item
        ]
    return len({str(title) for title in titles}) >= 2


def _hotpot_distractor_multihop(row: dict) -> bool:
    context = row.get("context", [])
    if isinstance(context, dict):
        context_count = len(context.get("title", []))
    else:
        context_count = len(context)
    return context_count == 10 and _explicit_multihop(row)


def _musique_multihop(row: dict) -> bool:
    if row.get("answerable") is False:
        return False
    decomposition = row.get("question_decomposition", row.get("decomposition", []))
    return isinstance(decomposition, list) and len(decomposition) >= 2


def _add_2wiki_answer_aliases(rows: list[dict], alias_source: Path) -> list[dict]:
    with alias_source.open(encoding="utf-8") as handle:
        alias_rows = [json.loads(line) for line in handle if line.strip()]
    aliases = {
        str(row["Q_id"]): [
            *row.get("aliases", []),
            *row.get("demonyms", []),
        ]
        for row in alias_rows
    }
    selected = []
    for row in rows:
        item = dict(row)
        item["answer_aliases"] = aliases.get(str(row.get("answer_id")), [])
        selected.append(item)
    return selected


def _fixed_sample(
    records: list[dict],
    sample_size: int,
    sample_seed: int,
    namespace: str,
) -> list[dict]:
    if len(records) < sample_size:
        raise ValueError(
            f"{namespace} has only {len(records)} eligible examples; "
            f"{sample_size} are required"
        )
    rng = random.Random(_derived_seed(sample_seed, namespace))
    return rng.sample(records, sample_size)


def _stratified_ultradomain_sample(
    source_dir: Path,
    sample_size: int,
    sample_seed: int,
) -> tuple[list[dict], list[Path]]:
    sources = sorted(
        path
        for path in source_dir.glob("*.jsonl")
        if path.stem != "mix"
    )
    if not sources:
        raise FileNotFoundError(f"no UltraDomain domain JSONL files under {source_dir}")
    domain_order = list(sources)
    random.Random(_derived_seed(sample_seed, "ultradomain-quotas")).shuffle(
        domain_order
    )
    base, remainder = divmod(sample_size, len(sources))
    quotas = {
        path: base + (1 if index < remainder else 0)
        for index, path in enumerate(domain_order)
    }
    selected = []
    for path in sources:
        rows = _read_records(path)
        domain_rows = _fixed_sample(
            rows,
            quotas[path],
            sample_seed,
            f"ultradomain:{path.stem}",
        )
        for row in domain_rows:
            row = dict(row)
            row["_s2rag_domain"] = path.stem
            selected.append(row)
    random.Random(_derived_seed(sample_seed, "ultradomain-order")).shuffle(
        selected
    )
    return selected, sources


def _manifest_entry(
    target: Path,
    source: Path,
    eligible_examples: int,
    hop_filter: str,
) -> dict:
    return {
        "path": str(target),
        "source": str(source),
        "examples": _line_count(target),
        "eligible_examples_before_sampling": eligible_examples,
        "explicit_multi_hop_filter": hop_filter,
        "sha256": _sha256(target),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _derived_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


if __name__ == "__main__":
    main()

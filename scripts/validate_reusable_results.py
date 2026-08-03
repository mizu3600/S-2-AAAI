from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly audit and import only protocol-identical old results."
    )
    parser.add_argument("--new-suite", type=Path, required=True)
    parser.add_argument("--old-suite", type=Path, required=True)
    parser.add_argument("--old-results", type=Path, required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--official-commit", required=True)
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--prompt-sha256", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--imported-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--rerun-output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path} does not contain a JSON array")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def example_keys(row: dict) -> list[str]:
    return list(
        dict.fromkeys(
            str(value)
            for value in (row.get("example_id"), row.get("source_example_id"))
            if value is not None
        )
    )


def canonical_documents(row: dict) -> list[dict]:
    return [
        {
            "document_id": str(document["document_id"]),
            "title": str(document.get("title") or ""),
            "text": str(document.get("text") or ""),
        }
        for document in row.get("documents") or []
    ]


def document_mapping(old: dict, new: dict) -> tuple[dict[str, str], list[str]]:
    new_by_content: dict[tuple[str, str], list[str]] = {}
    for document in canonical_documents(new):
        key = (document["title"], document["text"])
        new_by_content.setdefault(key, []).append(document["document_id"])
    mapping = {}
    errors = []
    for document in canonical_documents(old):
        key = (document["title"], document["text"])
        candidates = new_by_content.get(key, [])
        if len(candidates) != 1:
            errors.append(
                f"old document {document['document_id']} has "
                f"{len(candidates)} exact new matches"
            )
            continue
        mapping[document["document_id"]] = candidates[0]
    if len(mapping) != len(canonical_documents(old)):
        errors.append("candidate passage texts are not exactly mappable")
    if set(mapping.values()) != {
        item["document_id"] for item in canonical_documents(new)
    }:
        errors.append("new candidate passage set differs from old suite")
    return mapping, errors


def main() -> None:
    args = parse_args()
    parameters = json.loads(args.parameters_json)
    expected_artifact = {
        "framework": args.framework,
        "official_commit": args.official_commit,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "prompt_sha256": args.prompt_sha256,
        "parameters": parameters,
    }
    new_rows = read_rows(args.new_suite)
    old_rows = read_rows(args.old_suite)
    result_rows = read_rows(args.old_results)

    old_by_key = {
        key: row for row in old_rows for key in example_keys(row)
    }
    result_by_key = {
        key: row for row in result_rows for key in example_keys(row)
    }
    imported = []
    audit = []
    rerun = []
    for new in new_rows:
        example_id = str(new["example_id"])
        keys = example_keys(new)
        old = next((old_by_key[key] for key in keys if key in old_by_key), None)
        result = next(
            (result_by_key[key] for key in keys if key in result_by_key),
            None,
        )
        reasons = []
        mapping: dict[str, str] = {}
        if old is None:
            reasons.append("old_suite_example_missing")
        if result is None:
            reasons.append("old_result_missing")
        if old is not None:
            if str(old.get("question") or "") != str(new.get("question") or ""):
                reasons.append("question_mismatch")
            if str(old.get("answer") or "") != str(new.get("answer") or ""):
                reasons.append("answer_mismatch")
            mapping, document_errors = document_mapping(old, new)
            reasons.extend(document_errors)
        if result is not None:
            artifact = result.get("artifact_key")
            if not isinstance(artifact, dict):
                reasons.append("missing_artifact_key")
            else:
                for key, expected in expected_artifact.items():
                    if artifact.get(key) != expected:
                        reasons.append(f"artifact_{key}_mismatch")
            if result.get("status") != "success":
                reasons.append("old_result_not_success")

        if not reasons and result is not None:
            ranking = []
            unmapped = []
            for old_document_id in result.get("ranking") or []:
                mapped = mapping.get(str(old_document_id))
                if mapped is None:
                    unmapped.append(str(old_document_id))
                elif mapped not in ranking:
                    ranking.append(mapped)
            if unmapped:
                reasons.append("unmapped_result_document_ids")
            else:
                imported_row = dict(result)
                imported_row["example_id"] = example_id
                imported_row["ranking"] = ranking
                imported_row["suite_sha256"] = sha256_file(args.new_suite)
                imported_row["reuse_audit"] = {
                    "old_results_sha256": sha256_file(args.old_results),
                    "old_suite_sha256": sha256_file(args.old_suite),
                    "document_id_mapping": mapping,
                }
                imported.append(imported_row)
        if reasons:
            rerun.append(example_id)
        audit.append(
            {
                "example_id": example_id,
                "decision": "reused" if not reasons else "rerun",
                "reasons": sorted(set(reasons)),
                "document_id_mapping": mapping,
            }
        )

    atomic_write_jsonl(args.imported_output, imported)
    atomic_write_json(
        args.audit_output,
        {
            "framework": args.framework,
            "new_suite_sha256": sha256_file(args.new_suite),
            "old_suite_sha256": sha256_file(args.old_suite),
            "old_results_sha256": sha256_file(args.old_results),
            "expected_artifact": expected_artifact,
            "total": len(new_rows),
            "reused": len(imported),
            "rerun": len(rerun),
            "examples": audit,
        },
    )
    atomic_write_json(args.rerun_output, rerun)
    print(
        f"reused {len(imported)}/{len(new_rows)}; "
        f"{len(rerun)} examples require rerun",
        flush=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Awaitable, Callable

RETRYABLE_ERROR_MARKERS = (
    "rate limit",
    "ratelimit",
    "timeout",
    "timed out",
    "connectionerror",
    "apiconnectionerror",
    "service unavailable",
    "servererror",
)


class PersistentLLMCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_responses (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self._lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Task[str]] = {}

    async def get_or_compute(
        self,
        request: dict,
        compute: Callable[[], Awaitable[str]],
    ) -> str:
        serialized = json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        async with self._lock:
            row = self._connection.execute(
                "SELECT response FROM llm_responses WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None:
                return str(row[0])
            task = self._inflight.get(cache_key)
            if task is None:
                task = asyncio.create_task(self._compute_and_store(cache_key, compute))
                self._inflight[cache_key] = task
                task.add_done_callback(
                    lambda finished, key=cache_key: self._inflight.pop(key, None)
                )
        return await asyncio.shield(task)

    async def _compute_and_store(
        self,
        cache_key: str,
        compute: Callable[[], Awaitable[str]],
    ) -> str:
        response = await compute()
        async with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO llm_responses(cache_key, response) VALUES (?, ?)",
                (cache_key, response),
            )
            self._connection.commit()
        return response

    async def close(self) -> None:
        tasks = list(self._inflight.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._connection.close()


def add_strict_suite_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--suite-sha256", required=True)
    parser.add_argument("--resume", action="store_true")


def load_strict_rows(args: argparse.Namespace) -> list[dict]:
    if args.expected_count <= 0:
        raise ValueError("the strict benchmark protocol requires a positive example count")
    actual_sha256 = sha256_file(args.input)
    if actual_sha256 != args.suite_sha256:
        raise ValueError(
            f"canonical input SHA256 changed: {actual_sha256} != {args.suite_sha256}"
        )
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    validate_canonical_rows(
        rows,
        expected_dataset=args.dataset,
        expected_seed=args.seed,
        expected_count=args.expected_count,
    )
    if getattr(args, "offset", 0) != 0:
        raise ValueError("--offset must be 0 in the strict protocol")
    limit = getattr(args, "limit", None)
    if limit is not None and limit != args.expected_count:
        raise ValueError("--limit must equal --expected-count in the strict protocol")
    return rows


def load_resume_rows(
    output: Path,
    *,
    args: argparse.Namespace,
    system: str,
) -> tuple[Path, list[dict]]:
    partial = output.with_suffix(output.suffix + ".partial")
    if not args.resume:
        partial.unlink(missing_ok=True)
        return partial, []
    source = partial if partial.exists() else output
    if not source.exists():
        return partial, []
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_trace_rows(
        rows,
        dataset=args.dataset,
        seed=args.seed,
        suite_sha256=args.suite_sha256,
        system=system,
        expected_count=len(rows),
    )
    retained = [
        row
        for row in rows
        if row.get("status") == "success"
        and bool(row.get("ranking"))
        and not is_retryable_error(row.get("error"))
    ]
    if rows:
        atomic_write_jsonl(partial, retained)
    return partial, retained


def is_retryable_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.casefold().replace("_", "")
    return any(marker in normalized for marker in RETRYABLE_ERROR_MARKERS)


def base_record(example_id: str, framework: str, args: argparse.Namespace) -> dict:
    official_repo = getattr(args, "official_repo", None)
    commit = git_commit(official_repo) if official_repo else "package_import"
    return {
        "example_id": example_id,
        "framework": framework,
        "system": f"official:{framework}",
        "dataset": args.dataset,
        "seed": args.seed,
        "suite_sha256": args.suite_sha256,
        "status": "success",
        "ranking": [],
        "index_seconds": None,
        "retrieval_seconds": None,
        "error": None,
        "native_candidate_only": True,
        "model_trace": {
            "llm_model": getattr(args, "llm_model", "deepseek-v4-flash"),
            "embedding_model": getattr(args, "embedding_model", "BAAI/bge-m3"),
            "official_repository": str(official_repo) if official_repo else None,
            "official_commit": commit,
        },
    }


def finalize(
    *,
    output: Path,
    partial: Path,
    args: argparse.Namespace,
    system: str,
    expected_ids: set[str],
) -> None:
    rows = [
        json.loads(line)
        for line in partial.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_trace_rows(
        rows,
        dataset=args.dataset,
        seed=args.seed,
        suite_sha256=args.suite_sha256,
        system=system,
        expected_count=args.expected_count,
    )
    if {row["example_id"] for row in rows} != expected_ids:
        raise ValueError(f"{system} trace IDs do not exactly match the canonical suite")
    retryable_ids = [
        row["example_id"] for row in rows if is_retryable_error(row.get("error"))
    ]
    if retryable_ids:
        raise RuntimeError(
            f"{system} has {len(retryable_ids)} retryable failures; "
            "the partial trace was retained for resume"
        )
    unsuccessful = [
        row["example_id"]
        for row in rows
        if row.get("status") != "success" or not row.get("ranking")
    ]
    if unsuccessful:
        raise RuntimeError(
            f"{system} has {len(unsuccessful)} failed or empty native rankings; "
            "the partial trace was retained for audit"
        )
    atomic_write_jsonl(output, rows)
    partial.unlink(missing_ok=True)


def validate_canonical_rows(
    rows: list[dict],
    *,
    expected_dataset: str,
    expected_seed: int,
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"suite has {len(rows)} rows; expected {expected_count}")
    ids = []
    for row in rows:
        if row.get("dataset") != expected_dataset:
            raise ValueError(
                f"{row.get('example_id')} dataset mismatch: "
                f"{row.get('dataset')!r} != {expected_dataset!r}"
            )
        if row.get("seed") != expected_seed:
            raise ValueError(
                f"{row.get('example_id')} seed mismatch: "
                f"{row.get('seed')!r} != {expected_seed}"
            )
        example_id = str(row.get("example_id") or "")
        if not example_id:
            raise ValueError("suite row has no example_id")
        ids.append(example_id)
        documents = row.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError(f"{example_id} has no candidate documents")
        document_ids = [str(item.get("document_id") or "") for item in documents]
        if any(not item for item in document_ids):
            raise ValueError(f"{example_id} has a document without document_id")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(f"{example_id} has duplicate document IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("suite has duplicate example IDs")


def validate_trace_rows(
    rows: list[dict],
    *,
    dataset: str,
    seed: int,
    suite_sha256: str,
    system: str,
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"{system} trace has {len(rows)} rows; expected {expected_count}")
    ids = []
    for row in rows:
        if row.get("dataset") != dataset:
            raise ValueError(f"{system} trace dataset mismatch")
        if row.get("seed") != seed:
            raise ValueError(f"{system} trace seed mismatch")
        if row.get("suite_sha256") != suite_sha256:
            raise ValueError(f"{system} trace suite hash mismatch")
        if row.get("system") != system:
            raise ValueError(f"{system} trace system mismatch")
        ids.append(str(row.get("example_id") or ""))
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{system} trace has missing or duplicate example IDs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit(path: Path | str | None) -> str:
    if path is None:
        return "unknown"
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def artifact_key(
    *,
    framework: str,
    commit: str,
    llm_model: str,
    embedding_model: str,
    prompt_sha256: str,
    documents: list[dict],
    parameters: dict,
) -> dict:
    return {
        "framework": framework,
        "official_commit": commit,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "prompt_sha256": prompt_sha256,
        "document_sha256": sha256_json(documents),
        "parameters": parameters,
    }


def artifact_is_current(path: Path, key: dict) -> bool:
    marker = path / ".s2rag-artifact.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == key
    except (OSError, json.JSONDecodeError):
        return False


def mark_artifact(path: Path, key: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path / ".s2rag-artifact.json", key)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

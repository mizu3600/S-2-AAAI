from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np


class JsonFileCache:
    """Content-addressed JSON cache with atomic, thread-safe writes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    @staticmethod
    def digest(namespace: str, payload: Any) -> str:
        encoded = json.dumps(
            {"namespace": namespace, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def get(self, namespace: str, payload: Any) -> Any | None:
        path = self._path(namespace, self.digest(namespace, payload), ".json")
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, namespace: str, payload: Any, value: Any) -> None:
        path = self._path(namespace, self.digest(namespace, payload), ".json")
        with self._lock:
            if path.exists():
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, path)

    def delete(self, namespace: str, payload: Any) -> None:
        path = self._path(namespace, self.digest(namespace, payload), ".json")
        with self._lock:
            path.unlink(missing_ok=True)

    def _path(self, namespace: str, digest: str, suffix: str) -> Path:
        safe_namespace = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in namespace
        )
        return self.root / safe_namespace / digest[:2] / f"{digest}{suffix}"


class NumpyFileCache:
    """Content-addressed float32 vector cache."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def get(self, namespace: str, text: str) -> np.ndarray | None:
        path = self._path(namespace, text)
        if not path.exists():
            return None
        try:
            return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        except (OSError, ValueError):
            return None

    def put(self, namespace: str, text: str, vector: np.ndarray) -> None:
        path = self._path(namespace, text)
        with self._lock:
            if path.exists():
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
            np.save(temporary, np.asarray(vector, dtype=np.float32), allow_pickle=False)
            os.replace(temporary, path)

    def _path(self, namespace: str, text: str) -> Path:
        digest = hashlib.sha256(
            json.dumps(
                {"namespace": namespace, "text": text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        safe_namespace = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in namespace
        )
        return self.root / safe_namespace / digest[:2] / f"{digest}.npy"

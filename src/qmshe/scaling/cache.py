import hashlib
import json
from collections import OrderedDict
from typing import Any

from qmshe.scaling.versioning import ArtifactVersion


class VersionedQueryCache:
    def __init__(self, max_items: int = 1024):
        self.max_items = max_items
        self._items: OrderedDict[str, Any] = OrderedDict()

    def key(self, question: str, version: ArtifactVersion, options: dict | None = None) -> str:
        payload = {"question": question, "version": version.model_dump(), "options": options or {}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, key: str):
        if key not in self._items:
            return None
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key: str, value: Any) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()


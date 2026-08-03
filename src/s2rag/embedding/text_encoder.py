from collections.abc import Sequence
from pathlib import Path
import threading

import httpx
import numpy as np
import torch

from s2rag.cache import NumpyFileCache
from s2rag.dynamic_batching import DynamicBatcher
from s2rag.providers import ProviderError
from s2rag.settings import Settings, get_settings

PEFT_ADAPTER_MARKERS = (
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
)


class LocalBGEEncoder:
    """BGE-M3 encoder loaded exclusively from a local model directory."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        *,
        model=None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self._settings = settings
        self.model_path = Path(model_path or settings.bge_embedding_model_path)
        self.device = resolve_device(device or settings.bge_embedding_device or settings.bge_device)
        self.batch_size = batch_size or settings.bge_embedding_batch_size
        self._model = model
        self._model_lock = threading.Lock()
        self._cache = NumpyFileCache(settings.embedding_cache_dir)
        self._cache_namespace = f"bge-m3:{self.model_path.resolve()}:normalize=true"
        self._batcher = DynamicBatcher(
            self._encode_request_batch,
            max_items=settings.bge_embedding_micro_batch_max_texts,
            wait_seconds=settings.bge_dynamic_batch_wait_ms / 1000.0,
            name=f"bge-encoder-{self.device}",
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        unique = list(dict.fromkeys(values))
        vectors_by_text = {text: self._cache.get(self._cache_namespace, text) for text in unique}
        missing = [text for text, vector in vectors_by_text.items() if vector is None]
        if missing:
            encoded = self._batcher.submit(
                tuple(missing),
                item_count=len(missing),
            )
            vectors_by_text.update(zip(missing, encoded, strict=True))
        return np.stack([vectors_by_text[text] for text in values]).astype(np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def close(self) -> None:
        self._batcher.close()

    def _encode_request_batch(
        self,
        requests: list[tuple[str, ...]],
    ) -> list[np.ndarray]:
        unique = list(
            dict.fromkeys(text for request in requests for text in request)
        )
        vectors_by_text = {
            text: self._cache.get(self._cache_namespace, text) for text in unique
        }
        missing = [
            text for text, vector in vectors_by_text.items() if vector is None
        ]
        if missing:
            encoded = np.asarray(
                self._model_instance().encode(
                    missing,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            for text, vector in zip(missing, encoded, strict=True):
                vectors_by_text[text] = vector
                self._cache.put(self._cache_namespace, text, vector)
        return [
            np.stack([vectors_by_text[text] for text in request]).astype(np.float32)
            for request in requests
        ]

    def _model_instance(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_dir():
                raise ProviderError(
                    f"local BGE-M3 model directory does not exist: {self.model_path}"
                )
            reject_peft_adapter_directory(self.model_path, "BGE-M3")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ProviderError(
                    "sentence-transformers is required for the local BGE-M3 encoder"
                ) from exc
            try:
                self._model = SentenceTransformer(
                    str(self.model_path),
                    device=self.device,
                    local_files_only=True,
                    model_kwargs=(
                        {"torch_dtype": getattr(torch, self._settings.bge_dtype)}
                        if self.device.startswith("cuda")
                        and hasattr(torch, self._settings.bge_dtype)
                        else {}
                    ),
                )
            except Exception as exc:
                raise ProviderError(
                    f"failed to load local BGE-M3 model from {self.model_path}: {exc}"
                ) from exc
            return self._model


class TEIBGEEncoder:
    """BGE-M3 client for one shared Text Embeddings Inference service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18080/v1",
        model_id: str = "BAAI/bge-m3",
        batch_size: int = 32,
        *,
        http_client: httpx.Client | None = None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = "tei:cuda:0"
        self.model_path = None
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": "Bearer local-tei"},
            timeout=180,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=32),
        )
        self._cache = NumpyFileCache(settings.embedding_cache_dir)
        self._cache_namespace = (
            f"tei:{self.base_url}:{self.model_id}:normalize=true"
        )
        self._encode_lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        unique = list(dict.fromkeys(values))
        vectors = {
            text: self._cache.get(self._cache_namespace, text) for text in unique
        }
        missing = [text for text, vector in vectors.items() if vector is None]
        with self._encode_lock:
            still_missing = []
            for text in missing:
                cached = self._cache.get(self._cache_namespace, text)
                if cached is None:
                    still_missing.append(text)
                else:
                    vectors[text] = cached
            for start in range(0, len(still_missing), self.batch_size):
                batch = still_missing[start : start + self.batch_size]
                response = self._http_client.post(
                    "embeddings",
                    json={
                        "model": self.model_id,
                        "input": batch,
                        "encoding_format": "float",
                    },
                )
                if response.is_error:
                    raise ProviderError(
                        f"TEI embedding failed ({response.status_code}): "
                        f"{response.text[:300]}"
                    )
                ordered = sorted(
                    response.json()["data"],
                    key=lambda item: int(item["index"]),
                )
                encoded = np.asarray(
                    [item["embedding"] for item in ordered],
                    dtype=np.float32,
                )
                norms = np.linalg.norm(encoded, axis=1, keepdims=True)
                encoded = encoded / np.maximum(norms, 1e-12)
                for text, vector in zip(batch, encoded, strict=True):
                    vectors[text] = vector
                    self._cache.put(self._cache_namespace, text, vector)
        return np.stack([vectors[text] for text in values]).astype(np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()


def reject_peft_adapter_directory(model_path: Path, model_name: str) -> None:
    markers = [name for name in PEFT_ADAPTER_MARKERS if (model_path / name).exists()]
    if markers:
        raise ProviderError(
            f"{model_name} must use an original base-model directory; "
            f"PEFT/LoRA adapter files are not allowed: {', '.join(markers)}"
        )


def encode_documents(encoder, texts: Sequence[str]) -> np.ndarray:
    method = getattr(encoder, "encode_documents", encoder.encode)
    return method(texts)


def encode_queries(encoder, texts: Sequence[str]) -> np.ndarray:
    method = getattr(encoder, "encode_queries", encoder.encode)
    return method(texts)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

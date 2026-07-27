from collections.abc import Sequence
from pathlib import Path
import threading

import numpy as np
import torch

from s2rag.cache import NumpyFileCache
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
        self._encode_lock = threading.Lock()
        self._cache = NumpyFileCache(settings.embedding_cache_dir)
        self._cache_namespace = f"bge-m3:{self.model_path.resolve()}:normalize=true"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, 0), dtype=np.float32)
        unique = list(dict.fromkeys(values))
        vectors_by_text = {text: self._cache.get(self._cache_namespace, text) for text in unique}
        missing = [text for text, vector in vectors_by_text.items() if vector is None]
        if missing:
            model = self._model_instance()
            with self._encode_lock:
                still_missing = []
                for text in missing:
                    cached = self._cache.get(self._cache_namespace, text)
                    if cached is None:
                        still_missing.append(text)
                    else:
                        vectors_by_text[text] = cached
                if still_missing:
                    encoded = np.asarray(
                        model.encode(
                            still_missing,
                            batch_size=self.batch_size,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                        ),
                        dtype=np.float32,
                    )
                    for text, vector in zip(still_missing, encoded, strict=True):
                        vectors_by_text[text] = vector
                        self._cache.put(self._cache_namespace, text, vector)
        return np.stack([vectors_by_text[text] for text in values]).astype(np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

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

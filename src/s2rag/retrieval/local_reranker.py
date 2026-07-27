from collections.abc import Sequence
from pathlib import Path
import threading

import numpy as np
import torch

from s2rag.embedding.text_encoder import reject_peft_adapter_directory, resolve_device
from s2rag.providers import ProviderError
from s2rag.settings import Settings, get_settings


class LocalBGEReranker:
    """BGE reranker loaded exclusively from a local model directory."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        *,
        tokenizer=None,
        model=None,
        settings: Settings | None = None,
    ):
        settings = settings or get_settings()
        self.model_path = Path(model_path or settings.bge_reranker_model_path)
        self.device = resolve_device(device or settings.bge_reranker_device or settings.bge_device)
        self.batch_size = batch_size or settings.bge_reranker_batch_size
        self.max_length = max_length or settings.bge_reranker_max_length
        self._tokenizer = tokenizer
        self._model = model
        self._model_lock = threading.Lock()
        self._score_lock = threading.Lock()

    def rank(self, query: str, documents: Sequence[str]) -> list[int]:
        scores = self.score(query, documents)
        return np.argsort(-np.asarray(scores), kind="stable").tolist()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        if not documents:
            return []
        tokenizer, model = self._model_instances()
        scores: list[float] = []
        with self._score_lock:
            for start in range(0, len(documents), self.batch_size):
                batch = documents[start : start + self.batch_size]
                encoded = tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    logits = model(**encoded).logits.reshape(-1)
                scores.extend(float(value) for value in logits.detach().cpu())
        return scores

    def _model_instances(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        with self._model_lock:
            if self._tokenizer is not None and self._model is not None:
                return self._tokenizer, self._model
            if not self.model_path.is_dir():
                raise ProviderError(
                    f"local BGE reranker model directory does not exist: {self.model_path}"
                )
            reject_peft_adapter_directory(self.model_path, "BGE reranker")
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise ProviderError(
                    "transformers is required for the local BGE reranker"
                ) from exc
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                )
                self._model.to(self.device)
                self._model.eval()
            except Exception as exc:
                raise ProviderError(
                    f"failed to load local BGE reranker from {self.model_path}: {exc}"
                ) from exc
            return self._tokenizer, self._model

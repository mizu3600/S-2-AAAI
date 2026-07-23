from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as functional


class LocalBGEEncoder:
    """Cached BGE encoder with an optional query-only PEFT adapter."""

    def __init__(
        self, base_model: str, query_adapter: str | None = None, batch_size: int = 32,
        max_length: int = 256, device: str | None = None,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the 'lora' extra to use local BGE models") from exc
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        model = AutoModel.from_pretrained(base_model, local_files_only=True)
        if query_adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("install peft to load the query adapter") from exc
            model = PeftModel.from_pretrained(model, query_adapter, local_files_only=True)
        self.model = model.to(self.device).eval()
        self.has_adapter = query_adapter is not None
        self.batch_size = batch_size
        self.max_length = max_length
        self._document_cache: dict[str, np.ndarray] = {}
        self._query_cache: dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_documents(texts)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        context = self.model.disable_adapter() if self.has_adapter else nullcontext()
        return self._cached_encode(texts, self._document_cache, context)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._cached_encode(texts, self._query_cache, nullcontext())

    def _cached_encode(self, texts, cache, context) -> np.ndarray:
        texts = list(texts)
        missing = list(dict.fromkeys(text for text in texts if text not in cache))
        with context, torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                encoded = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {name: value.to(self.device) for name, value in encoded.items()}
                vectors = functional.normalize(
                    self.model(**encoded).last_hidden_state[:, 0], p=2, dim=-1
                ).float().cpu().numpy()
                cache.update(zip(batch, vectors, strict=True))
        return np.stack([cache[text] for text in texts]).astype(np.float32)


class LocalBGEReranker:
    """Cached BGE cross-encoder reranker with an optional PEFT adapter."""

    def __init__(
        self, base_model: str, adapter: str | None = None, batch_size: int = 32,
        max_length: int = 512, device: str | None = None,
    ):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the 'lora' extra to use the local reranker") from exc
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, local_files_only=True
        )
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("install peft to load the reranker adapter") from exc
            model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self._cache: dict[tuple[str, str], float] = {}

    def rank(self, query: str, documents: Sequence[str]) -> list[int]:
        documents = list(documents)
        missing = [(query, document) for document in documents if (query, document) not in self._cache]
        with torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                encoded = self.tokenizer(
                    [item[0] for item in batch], [item[1] for item in batch], padding=True,
                    truncation=True, max_length=self.max_length, return_tensors="pt",
                )
                encoded = {name: value.to(self.device) for name, value in encoded.items()}
                scores = self.model(**encoded).logits.reshape(-1).float().cpu().tolist()
                self._cache.update(zip(batch, scores, strict=True))
        return sorted(
            range(len(documents)),
            key=lambda index: self._cache[(query, documents[index])], reverse=True,
        )

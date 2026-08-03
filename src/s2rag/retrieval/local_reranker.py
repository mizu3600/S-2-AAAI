from collections.abc import Sequence
from pathlib import Path
import threading

import httpx
import numpy as np
import torch

from s2rag.dynamic_batching import DynamicBatcher
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
        self._batcher = DynamicBatcher(
            self._score_request_batch,
            max_items=settings.bge_reranker_micro_batch_max_pairs,
            wait_seconds=settings.bge_dynamic_batch_wait_ms / 1000.0,
            name=f"bge-reranker-{self.device}",
        )

    def rank(self, query: str, documents: Sequence[str]) -> list[int]:
        scores = self.score(query, documents)
        return np.argsort(-np.asarray(scores), kind="stable").tolist()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        if not documents:
            return []
        return self._batcher.submit(
            (query, tuple(documents)),
            item_count=len(documents),
        )

    def close(self) -> None:
        self._batcher.close()

    def _score_request_batch(
        self,
        requests: list[tuple[str, tuple[str, ...]]],
    ) -> list[list[float]]:
        tokenizer, model = self._model_instances()
        queries = [
            query
            for query, documents in requests
            for _ in range(len(documents))
        ]
        documents = [
            document
            for _, request_documents in requests
            for document in request_documents
        ]
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch_documents = documents[start : start + self.batch_size]
            encoded = tokenizer(
                queries[start : start + len(batch_documents)],
                batch_documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits.reshape(-1)
            scores.extend(float(value) for value in logits.detach().cpu())

        results = []
        offset = 0
        for _, request_documents in requests:
            end = offset + len(request_documents)
            results.append(scores[offset:end])
            offset = end
        return results

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


class TEIBGEReranker:
    """BGE reranker client backed by the shared GPU 1 TEI service."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:18081/rerank",
        model_id: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 32,
        max_length: int = 8192,
        *,
        http_client: httpx.Client | None = None,
    ):
        self.endpoint = endpoint
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "tei:cuda:1"
        self.model_path = None
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(timeout=180)
        self._score_lock = threading.Lock()

    def rank(self, query: str, documents: Sequence[str]) -> list[int]:
        scores = self.score(query, documents)
        return np.argsort(-np.asarray(scores), kind="stable").tolist()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        documents = list(documents)
        if not documents:
            return []
        scores = []
        with self._score_lock:
            for start in range(0, len(documents), self.batch_size):
                batch = documents[start : start + self.batch_size]
                response = self._http_client.post(
                    self.endpoint,
                    json={"query": query, "texts": batch, "truncate": True},
                )
                if response.is_error:
                    raise ProviderError(
                        f"TEI reranking failed ({response.status_code}): "
                        f"{response.text[:300]}"
                    )
                payload = response.json()
                items = payload if isinstance(payload, list) else payload["results"]
                by_index = {
                    int(item["index"]): float(item["score"]) for item in items
                }
                scores.extend(by_index[index] for index in range(len(batch)))
        return scores

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

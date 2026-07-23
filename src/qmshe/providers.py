import hashlib
import json
from collections.abc import Sequence

import httpx
import numpy as np

from qmshe.settings import Settings, get_settings


class ProviderError(RuntimeError):
    pass


class DeterministicEmbedder:
    """Stable keyless test encoder; not used for scientific comparisons."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode(), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "little") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                matrix[row, index] += sign
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norm, 1e-12)


class SiliconFlowClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.siliconflow_api_key:
            raise ProviderError("SILICONFLOW_API_KEY is not configured")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        payload = {"model": self.settings.siliconflow_embedding_model, "input": list(texts)}
        response = self._post("/embeddings", payload)
        vectors = [row["embedding"] for row in sorted(response["data"], key=lambda x: x["index"])]
        array = np.asarray(vectors, dtype=np.float32)
        return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)

    def rerank(self, query: str, documents: Sequence[str], top_n: int | None = None) -> list[dict]:
        payload = {
            "model": self.settings.siliconflow_reranker_model,
            "query": query,
            "documents": list(documents),
            "top_n": top_n or len(documents),
            "return_documents": False,
        }
        return self._post("/rerank", payload)["results"]

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.settings.siliconflow_api_key}"}
        with httpx.Client(timeout=self.settings.request_timeout) as client:
            response = client.post(f"{self.settings.siliconflow_base_url}{path}", headers=headers, json=payload)
        if response.is_error:
            raise ProviderError(f"SiliconFlow request failed ({response.status_code}): {response.text[:300]}")
        return response.json()


class DeepSeekClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not self.settings.deepseek_api_key:
            raise ProviderError("DEEPSEEK_API_KEY is not configured")

    def complete_json(self, system: str, prompt: str) -> dict:
        text = self.complete(system, prompt, response_format={"type": "json_object"})
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError("DeepSeek returned invalid JSON") from exc

    def complete(self, system: str, prompt: str, response_format: dict | None = None) -> str:
        payload = {
            "model": self.settings.deepseek_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0,
            "thinking": {"type": "disabled"},
        }
        if response_format:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self.settings.deepseek_api_key}"}
        with httpx.Client(timeout=self.settings.request_timeout) as client:
            response = client.post(
                f"{self.settings.deepseek_base_url}/chat/completions", headers=headers, json=payload
            )
        if response.is_error:
            raise ProviderError(f"DeepSeek request failed ({response.status_code}): {response.text[:300]}")
        return response.json()["choices"][0]["message"]["content"]


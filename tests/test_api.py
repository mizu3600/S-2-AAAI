import hashlib

import numpy as np
from fastapi.testclient import TestClient

from s2rag.api.dependencies import set_pipeline
from s2rag.api.main import app
from s2rag.pipeline import S2RAGPipeline
from s2rag.synthetic import make_synthetic_corpus


class LocalEncoder:
    def encode(self, texts):
        matrix = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.casefold().split():
                index = (
                    int.from_bytes(
                        hashlib.blake2b(token.encode(), digest_size=8).digest(),
                        "little",
                    )
                    % matrix.shape[1]
                )
                matrix[row, index] += 1
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class FakeGenerator:
    def generate(self, question, context):
        return "Generated from evidence."


def test_health_query_and_metrics():
    pipeline = S2RAGPipeline(
        make_synthetic_corpus(),
        text_encoder=LocalEncoder(),
        use_local_reranker=False,
        generator=FakeGenerator(),
    )
    set_pipeline(pipeline)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["pipeline"] == "reified_fact_hybrid"

    response = client.post(
        "/v1/query",
        json={
            "question": "How does request caching improve response latency?",
            "top_k": 3,
            "return_debug": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["retrieved_facts"]
    assert response.json()["citations"]

    metrics = client.get("/v1/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["queries"] == 1

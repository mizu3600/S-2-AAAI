from fastapi.testclient import TestClient

from qmshe.api.dependencies import set_pipeline
from qmshe.api.main import app
from qmshe.pipeline import QMSHERAGPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.synthetic import make_synthetic_corpus


class LocalEncoder:
    def encode(self, texts):
        return DeterministicEmbedder(64).embed(texts)


def test_health_query_and_metrics():
    pipeline = QMSHERAGPipeline(
        make_synthetic_corpus(),
        text_encoder=LocalEncoder(),
        enable_remote_reranker=False,
    )
    pipeline.generator.client = None
    set_pipeline(pipeline)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["pipeline"] == "reified_fact_hybrid"

    response = client.post(
        "/v1/query",
        json={
            "question": "How does PEAI improve Voc?",
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

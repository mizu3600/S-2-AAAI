from fastapi.testclient import TestClient

from qmshe.api.dependencies import set_graph_pipeline, set_pipeline
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.api.main import app
from qmshe.pipeline import QMSHEPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.synthetic import make_synthetic_corpus


class LocalEncoder:
    def encode(self, texts):
        return DeterministicEmbedder(64).embed(texts)


def test_health_and_query_api():
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["active_profile"] == "reified_fact"
    set_graph_pipeline(QMSGEGraphPipeline(
        make_synthetic_corpus(), text_encoder=LocalEncoder(),
        profile=GraphProfile.REIFIED_FACT,
    ))
    response = client.post("/v1/query", json={"question": "How does PEAI improve Voc?", "top_k": 3, "return_debug": True})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert set(body["band_weights"]) == {"raw", "low", "mid", "high"}
    assert body["profile"] == "reified_fact"
    evaluation = client.post("/v1/evaluate/run")
    assert evaluation.status_code == 200
    run_id = evaluation.json()["run_id"]
    assert client.get(f"/v1/evaluate/{run_id}").json()["status"] == "completed"
    metrics = client.get("/v1/metrics").json()
    assert metrics["queries"] >= 1
    assert metrics["p95_latency_ms"] >= 0


def test_disabled_runtime_modes_remain_present_but_require_opt_in():
    client = TestClient(app)
    corpus = make_synthetic_corpus()
    set_pipeline(QMSHEPipeline(corpus, text_encoder=LocalEncoder()))
    set_graph_pipeline(QMSGEGraphPipeline(
        corpus, text_encoder=LocalEncoder(), profile=GraphProfile.REIFIED_FACT
    ))
    set_graph_pipeline(QMSGEGraphPipeline(
        corpus, text_encoder=LocalEncoder(), profile=GraphProfile.ENTITY_RELATION
    ))
    graph_response = client.post("/v1/query", json={
        "question": "How does PEAI improve Voc?",
        "mode": "graph",
        "graph_profile": "reified_fact",
        "top_k": 3,
    })
    assert graph_response.status_code == 200
    assert graph_response.json()["mode"] == "graph"
    assert graph_response.json()["profile"] == "reified_fact"

    hypergraph_response = client.post("/v1/query", json={
        "question": "How does PEAI improve Voc?", "mode": "hypergraph", "top_k": 3
    })
    assert hypergraph_response.status_code == 409
    assert "disabled" in hypergraph_response.json()["detail"]

    entity_relation_response = client.post("/v1/query", json={
        "question": "How does PEAI improve Voc?",
        "mode": "graph",
        "graph_profile": "entity_relation",
        "top_k": 3,
    })
    assert entity_relation_response.status_code == 409
    assert "disabled" in entity_relation_response.json()["detail"]

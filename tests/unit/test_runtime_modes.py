from qmshe.api.routes_ingest import BuildRequest, IncrementalRequest
from qmshe.api.routes_query import QueryRequest
from qmshe.graph.ordinary import GraphProfile
from qmshe.settings import Settings


def test_reified_fact_is_the_only_default_runtime_mode():
    settings = Settings()

    assert settings.qmshe_enable_reified_fact is True
    assert settings.qmshe_enable_entity_relation is False
    assert settings.qmshe_enable_hypergraph is False
    assert BuildRequest().mode == "graph"
    assert BuildRequest().graph_profile is GraphProfile.REIFIED_FACT
    assert IncrementalRequest(corpus_path="corpus.json").mode == "graph"
    assert QueryRequest(question="test").mode == "graph"
    assert QueryRequest(question="test").graph_profile is GraphProfile.REIFIED_FACT


def test_preserved_modes_can_be_reenabled_from_environment(monkeypatch):
    monkeypatch.setenv("QMSHE_ENABLE_HYPERGRAPH", "true")
    monkeypatch.setenv("QMSHE_ENABLE_ENTITY_RELATION", "true")

    settings = Settings()

    assert settings.qmshe_enable_hypergraph is True
    assert settings.qmshe_enable_entity_relation is True

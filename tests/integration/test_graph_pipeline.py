from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.synthetic import make_synthetic_corpus


class LocalEncoder:
    def __init__(self):
        self.encoder = DeterministicEmbedder(64)

    def encode(self, texts):
        return self.encoder.embed(texts)


def test_both_ordinary_graph_profiles_return_traceable_facts():
    for profile in (GraphProfile.ENTITY_RELATION, GraphProfile.REIFIED_FACT):
        pipeline = QMSGEGraphPipeline(
            make_synthetic_corpus(), text_encoder=LocalEncoder(), profile=profile
        )
        assert pipeline.use_graph_rerank is False
        result = pipeline.query("How does PEAI improve Voc?", top_k=4)
        assert result.mode == "graph"
        assert result.profile == profile.value
        assert result.retrieved_nodes
        assert result.retrieved_facts
        assert result.citations
        assert abs(sum(result.band_weights.values()) - 1.0) < 1e-6


def test_single_and_multi_index_strategies_are_runnable():
    for strategy in ("single", "multi", "hybrid"):
        pipeline = QMSGEGraphPipeline(
            make_synthetic_corpus(), text_encoder=LocalEncoder(),
            profile=GraphProfile.REIFIED_FACT, index_strategy=strategy,
        )
        result = pipeline.query("How does PEAI improve Voc?", top_k=3)
        assert result.index_strategy == strategy
        assert result.retrieved_facts


def test_graph_stage_a_trains_without_changing_hypergraph_pipeline():
    pipeline = QMSGEGraphPipeline(
        make_synthetic_corpus(), text_encoder=LocalEncoder(),
        profile=GraphProfile.REIFIED_FACT,
    )
    history = pipeline.train_stage_a(
        [("How does PEAI improve Voc?", {"fact_1"})], epochs=2
    )
    assert len(history) == 2
    assert all(value >= 0 for value in history)

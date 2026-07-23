import asyncio
from types import SimpleNamespace

from qmshe.benchmark_v2.dataset import build_candidate_view
from qmshe.benchmark_v2.evaluator import answer_scores, evaluate_prediction
from qmshe.benchmark_v2.extraction import StructuredFactExtractor
from qmshe.benchmark_v2.official_rankings import capture_ranked_text_units
from qmshe.benchmark_v2.schemas import V2Example, V2Passage
from qmshe.benchmark_v2.systems import ControlledTopologyRetriever


def _example():
    passages = (
        V2Passage("p0", "Ada Lovelace", ("Ada Lovelace was born in London in 1815.",)),
        V2Passage("p1", "London", ("London is located in England.",)),
    )
    return V2Example(
        "e0", "Where was Ada Lovelace born?", "London", passages,
        frozenset({("p0", 0)}), "bridge", "easy",
    )


def test_candidate_views_preserve_gold_and_add_deterministic_distractors():
    example = _example()
    pool = example.passages + (
        V2Passage("p2", "Paris", ("Paris is located in France.",)),
        V2Passage("p3", "Rome", ("Rome is located in Italy.",)),
    )
    first = build_candidate_view(example, pool, 3, seed=7)
    second = build_candidate_view(example, pool, 3, seed=7)
    assert first.passages == second.passages
    assert len(first.passages) == 3
    assert len(first.gold_fact_ids) == 1


def test_structured_extractor_emits_roles_predicate_qualifiers_and_links():
    fact = StructuredFactExtractor().extract_passage(_example().passages[0])[0]
    assert fact.predicate == "born"
    assert {role for role, _ in fact.roles} == {"person", "birth"}
    assert ("year", "1815") in fact.qualifiers
    assert "ada lovelace" in fact.entity_ids


def test_all_topologies_obey_same_fact_and_reranker_budget():
    view = build_candidate_view(_example(), _example().passages, 2)
    retriever = ControlledTopologyRetriever(retrieval_budget=2, output_facts=1)
    for profile in retriever.profiles:
        prediction = retriever.rank(view, profile)
        assert len(prediction.fact_ranking) == 1
        assert prediction.diagnostics["reranker_inputs"] == 2
        assert prediction.diagnostics["ranking_origin"] == "internal_fact_scores"


def test_four_layer_metrics_include_multi_k_accuracy():
    view = build_candidate_view(_example(), _example().passages, 2)
    prediction = ControlledTopologyRetriever(2, 2).rank(view, "hypergraph")
    record = evaluate_prediction(view, prediction)
    for metric in ("fact_recall_at_10", "path_f1", "answer_em", "citation_f1", "joint_f1"):
        assert metric in record
    assert record["citation_valid"] == 1.0
    assert answer_scores("The London", "London")[0] == 1.0


def test_official_capture_uses_internal_objects_and_restores_function():
    async def select(*args, **kwargs):
        return [{"id": "chunk-2", "source_id": "doc-2"}, {"id": "chunk-1", "source_id": "doc-1"}]

    module = SimpleNamespace(__name__="official.operate", select=select)

    async def exercise():
        async with capture_ranked_text_units("pathrag", module, "select") as capture:
            await module.select()
            assert capture.source_ids() == ["doc-2", "doc-1"]
            assert capture.origin.startswith("official_internal:")
        assert module.select is select

    asyncio.run(exercise())

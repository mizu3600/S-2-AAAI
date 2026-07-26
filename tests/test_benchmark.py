from qmshe.benchmarks.schemas import (
    BenchmarkExample,
    BenchmarkSuite,
    Passage,
    SupportingFact,
)
from qmshe.evaluation.experiment import (
    HOTPOT_SHARED_GENERATION_PROTOCOL,
    BenchmarkExperimentRunner,
)
from qmshe.evaluation.experiment import score_external_result
from qmshe.evaluation.external_adapters import ADAPTERS, load_external_results
from qmshe.evaluation.internal_baselines import INTERNAL_BASELINES
from qmshe.benchmarks.corpus_builder import build_example_corpus


def test_benchmark_runner_scores_multiple_methods(tmp_path):
    example = BenchmarkExample(
        example_id="ex_1",
        question="What improves voltage?",
        answer="surface passivation",
        passages=[
            Passage(
                passage_id="p1",
                title="Material",
                sentences=["Surface passivation reduces defects."],
            ),
            Passage(
                passage_id="p2",
                title="Result",
                sentences=["Surface passivation improves voltage."],
            ),
        ],
        supporting_facts=[
            SupportingFact(passage_id="p1", sentence_index=0),
            SupportingFact(passage_id="p2", sentence_index=0),
        ],
        dataset="toy",
        split="test",
    )
    suite = BenchmarkSuite(
        name="toy", split="test", examples=[example], source="memory://toy"
    )

    records = BenchmarkExperimentRunner().run(suite, tmp_path, seed=42)

    assert {record["system"] for record in records} == {
        "bm25",
        "dense",
        "bm25_dense_rrf",
        "reified_fact_hybrid",
    }
    assert (tmp_path / "records.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert all("fact_recall_at_5" in record for record in records)


def test_all_internal_baselines_share_the_benchmark_contract(tmp_path):
    example = BenchmarkExample(
        example_id="ex_2",
        question="What improves voltage?",
        answer="surface passivation",
        passages=[
            Passage(
                passage_id="p1",
                title="Material",
                sentences=["Surface passivation reduces defects."],
            ),
            Passage(
                passage_id="p2",
                title="Result",
                sentences=["Surface passivation improves voltage."],
            ),
        ],
        supporting_facts=[
            SupportingFact(passage_id="p1", sentence_index=0),
            SupportingFact(passage_id="p2", sentence_index=0),
        ],
        dataset="toy",
        split="test",
    )
    suite = BenchmarkSuite(
        name="toy", split="test", examples=[example], source="memory://toy"
    )

    records = BenchmarkExperimentRunner(methods=INTERNAL_BASELINES).run(
        suite, tmp_path, seed=42
    )

    assert {record["system"] for record in records} == set(INTERNAL_BASELINES)
    assert all(record["ranking_origin"] for record in records)


def test_each_external_adapter_normalizes_title_rankings(tmp_path):
    example = BenchmarkExample(
        example_id="ex_3",
        question="What improves voltage?",
        answer="surface passivation",
        passages=[
            Passage(passage_id="p1", title="Material", sentences=["Passivation helps."]),
            Passage(passage_id="p2", title="Result", sentences=["Voltage improves."]),
        ],
        supporting_facts=[SupportingFact(passage_id="p2", sentence_index=0)],
        dataset="toy",
        split="test",
    )
    suite = BenchmarkSuite(
        name="toy", split="test", examples=[example], source="memory://toy"
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        '[{"example_id":"ex_3","document_ranking":["Result","Material"],'
        '"citations":["Result"],"answer":"surface passivation"}]',
        encoding="utf-8",
    )

    for name in ADAPTERS:
        normalized = load_external_results(name, result_path, suite)
        assert normalized[0].document_ranking == ["p2", "p1"]
        score = score_external_result(example, build_example_corpus(example), normalized[0])
        assert score["system"] == f"external:{name}"


def test_external_passage_ranking_is_not_expanded_into_fact_ranking(tmp_path):
    example = BenchmarkExample(
        example_id="ex_passage",
        question="Who won?",
        answer="Ada",
        passages=[
            Passage(passage_id="p1", title="Distractor", sentences=["No.", "Still no."]),
            Passage(passage_id="p2", title="Result", sentences=["Setup.", "Ada won."]),
        ],
        supporting_facts=[SupportingFact(passage_id="p2", sentence_index=1)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "external.json"
    result_path.write_text(
        '[{"example_id":"ex_passage","document_ranking":["Result"],'
        '"citations":["Result"],"answer":"Ada"}]',
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, build_example_corpus(example), result)

    assert score["passage_recall_at_1"] == 1.0
    assert score["fact_recall_at_1"] is None
    assert score["generated_passage_citation_f1"] == 1.0
    assert score["generated_fact_citation_f1"] is None
    assert score["answer_f1"] is None
    assert score["joint_f1"] is None


def test_missing_external_results_stay_in_the_denominator(tmp_path):
    examples = [
        BenchmarkExample(
            example_id=example_id,
            question="Who won?",
            answer="Ada",
            passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
            supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
            dataset="hotpotqa",
            split="validation",
        )
        for example_id in ("present", "missing")
    ]
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=examples,
        source="memory://hotpot",
    )
    result_path = tmp_path / "partial.json"
    result_path.write_text(
        '[{"example_id":"present","document_ranking":["Result"]}]',
        encoding="utf-8",
    )

    results = load_external_results("lightrag", result_path, suite)
    missing = next(item for item in results if item.example_id == "missing")
    score = score_external_result(
        examples[1], build_example_corpus(examples[1]), missing
    )

    assert len(results) == 2
    assert missing.status == "missing"
    assert score["passage_recall_at_1"] == 0.0
    assert score["result_missing"] is True


def test_each_native_ranking_field_uses_the_hotpot_source_map(tmp_path):
    fields = {
        "graphrag": "community_ranking",
        "lightrag": "entity_ranking",
        "pathrag": "path_context_ranking",
        "hypergraphrag": "hyperedge_ranking",
        "hipporag2": "ppr_ranking",
        "cograg": "dual_hypergraph_ranking",
        "hgrag": "diffusion_ranking",
        "hyperrag": "hypergraph_ranking",
    }
    example = BenchmarkExample(
        example_id="mapped",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )

    for baseline, field in fields.items():
        result_path = tmp_path / f"{baseline}.json"
        result_path.write_text(
            __import__("json").dumps(
                [{
                    "example_id": "mapped",
                    field: [{"id": "native-node-1"}],
                    "source_id_map": {"native-node-1": "Result"},
                }]
            ),
            encoding="utf-8",
        )
        result = load_external_results(baseline, result_path, suite)[0]
        assert result.document_ranking == ["p1"]
        assert result.mapping_coverage == 1.0
        assert result.status == "success"


def test_unmapped_native_ids_are_unscorable_instead_of_silently_dropped(tmp_path):
    example = BenchmarkExample(
        example_id="unmapped",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "unmapped.json"
    result_path.write_text(
        '[{"example_id":"unmapped","community_ranking":["unknown-community"]}]',
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, build_example_corpus(example), result)

    assert result.status == "unscorable"
    assert result.mapping_coverage == 0.0
    assert result.unmapped_ranking_ids == ["unknown-community"]
    assert score["passage_recall_at_1"] == 0.0
    assert score["result_failed"] is True


def test_shared_hotpot_generation_enables_answer_citation_and_joint_f1(tmp_path):
    example = BenchmarkExample(
        example_id="shared",
        question="Who won?",
        answer="Ada",
        passages=[Passage(passage_id="p1", title="Result", sentences=["Ada won."])],
        supporting_facts=[SupportingFact(passage_id="p1", sentence_index=0)],
        dataset="hotpotqa",
        split="validation",
    )
    built = build_example_corpus(example)
    fact_id = next(iter(built.gold_fact_ids))
    suite = BenchmarkSuite(
        name="hotpotqa",
        split="validation",
        examples=[example],
        source="memory://hotpot",
    )
    result_path = tmp_path / "shared.json"
    result_path.write_text(
        __import__("json").dumps([{
            "example_id": "shared",
            "document_ranking": ["p1"],
            "fact_ranking": [fact_id],
            "answer": f"Ada [{fact_id}]",
            "citations": [fact_id],
            "generation_protocol": HOTPOT_SHARED_GENERATION_PROTOCOL,
        }]),
        encoding="utf-8",
    )

    result = load_external_results("graphrag", result_path, suite)[0]
    score = score_external_result(example, built, result)

    assert score["answer_f1"] == 1.0
    assert score["generated_fact_citation_f1"] == 1.0
    assert score["joint_f1"] == 1.0

import importlib.util
from pathlib import Path

from s2rag.benchmarks.schemas import (
    BenchmarkExample,
    Passage,
    SupportingFact,
)

_SPEC = importlib.util.spec_from_file_location(
    "export_protocol_v2_suites",
    Path(__file__).parents[1] / "scripts" / "export_protocol_v2_suites.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_canonical_row = _MODULE._canonical_row


def test_canonical_row_preserves_complete_passages_and_gold_fact_mapping():
    example = BenchmarkExample(
        dataset="hotpotqa",
        split="test",
        example_id="q1",
        question="Where was the author born?",
        answer="London",
        passages=[
            Passage(
                passage_id="q1_p0",
                title="Author",
                sentences=["The author was born in London.", "They wrote a novel."],
            )
        ],
        supporting_facts=[
            SupportingFact(passage_id="q1_p0", sentence_index=0)
        ],
        metadata={"corpus_scope": "per_question_candidate_passages"},
    )

    row = _canonical_row(example, 42)

    assert row["documents"] == [
        {
            "document_id": "q1_p0",
            "title": "Author",
            "text": "The author was born in London. They wrote a novel.",
        }
    ]
    assert len(row["facts"]) == 2
    assert row["gold_fact_ids"] == [row["facts"][0]["fact_id"]]
    assert row["seed"] == 42


def test_canonical_row_keeps_answer_aliases_for_ultradomain():
    example = BenchmarkExample(
        dataset="ultradomain",
        split="test",
        example_id="u1",
        question="What is the answer?",
        answer=["primary", "alias"],
        passages=[
            Passage(
                passage_id="u1_p0",
                title="Document",
                sentences=["Complete source text."],
            )
        ],
    )

    row = _canonical_row(example, 42)

    assert row["answer"] == "primary"
    assert row["answer_aliases"] == ["primary", "alias"]
    assert row["documents"][0]["text"] == "Complete source text."

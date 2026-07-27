import json

import pytest

from scripts.prepare_four_benchmarks import (
    _add_2wiki_answer_aliases,
    _explicit_multihop,
    _fixed_sample,
    _hotpot_distractor_multihop,
    _musique_multihop,
    _stratified_ultradomain_sample,
)
from s2rag.benchmarks.adapters import (
    HotpotAdapter,
    MusiqueAdapter,
    TwoWikiAdapter,
    UltraDomainAdapter,
    _read_records,
)


def _hotpot_row():
    return {
        "_id": "multi-hop-1",
        "question": "Which city is home to both people?",
        "answer": "London",
        "type": "bridge",
        "context": [
            ["Person A", ["Person A lives in London."]],
            ["Person B", ["Person B also lives in London."]],
        ],
        "supporting_facts": [["Person A", 0], ["Person B", 0]],
    }


def test_hotpot_and_2wiki_adapters_preserve_explicit_multihop_evidence():
    for adapter in (HotpotAdapter(), TwoWikiAdapter()):
        example = adapter.convert(_hotpot_row(), "validation")

        assert example.hop_count == 2
        assert len({fact.passage_id for fact in example.supporting_facts}) == 2
        assert len(example.gold_path) == 2
        if type(adapter) is HotpotAdapter:
            assert example.metadata["benchmark_config"] == "unspecified"


def test_musique_adapter_preserves_decomposition_hops():
    row = {
        "id": "musique-1",
        "question": "Where was the author born?",
        "answer": "Paris",
        "answerable": True,
        "paragraphs": [
            {
                "title": "Book",
                "paragraph_text": "The book was written by Ada.",
                "is_supporting": True,
            },
            {
                "title": "Ada",
                "paragraph_text": "Ada was born in Paris.",
                "is_supporting": True,
            },
        ],
        "question_decomposition": [
            {"question": "Who wrote the book?", "answer": "Ada"},
            {"question": "Where was Ada born?", "answer": "Paris"},
        ],
    }

    example = MusiqueAdapter().convert(row, "validation")

    assert example.hop_count == 2
    assert example.bridge_entities == ["Ada"]
    assert len({fact.passage_id for fact in example.supporting_facts}) == 2


def test_ultradomain_adapter_marks_gold_evidence_and_hops_unavailable():
    row = {
        "_id": "ultra-1",
        "input": "What does the source say?",
        "answers": ["A concise answer."],
        "context": "The source contains a long domain document.",
        "_s2rag_domain": "legal",
    }

    example = UltraDomainAdapter().convert(row, "validation")

    assert example.example_id == "ultra-1"
    assert example.answer == ["A concise answer."]
    assert example.supporting_facts == []
    assert example.metadata["domain"] == "legal"
    assert example.metadata["evidence_level"] == "unavailable"
    assert example.metadata["multi_hop_annotation"] == "unavailable"


def test_jsonl_reader_does_not_split_unicode_line_separator(tmp_path):
    path = tmp_path / "unicode.jsonl"
    rows = [{"id": "one", "context": "before\u2028after"}, {"id": "two"}]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert _read_records(path) == rows


def test_multihop_filters_require_explicit_two_hop_structure():
    assert _explicit_multihop(_hotpot_row())
    assert not _explicit_multihop(
        {"supporting_facts": [["Only one title", 0], ["Only one title", 1]]}
    )
    assert _musique_multihop(
        {"answerable": True, "question_decomposition": [{}, {}]}
    )
    assert not _musique_multihop(
        {"answerable": False, "question_decomposition": [{}, {}]}
    )
    distractor = _hotpot_row()
    distractor["context"].extend(
        [[f"Distractor {index}", ["Not supporting."]] for index in range(8)]
    )
    assert _hotpot_distractor_multihop(distractor)
    assert not _hotpot_distractor_multihop(_hotpot_row())


def test_2wiki_aliases_are_embedded_for_official_v11_scoring(tmp_path):
    alias_path = tmp_path / "id_aliases.json"
    alias_path.write_text(
        json.dumps(
            {
                "Q_id": "Q1",
                "aliases": ["New York City"],
                "demonyms": ["New Yorker"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _add_2wiki_answer_aliases(
        [{"answer": "NYC", "answer_id": "Q1"}],
        alias_path,
    )

    assert rows[0]["answer_aliases"] == ["New York City", "New Yorker"]


def test_fixed_sample_is_deterministic_and_requires_enough_rows():
    rows = [{"id": index} for index in range(20)]

    first = _fixed_sample(rows, 10, 42, "test")
    second = _fixed_sample(rows, 10, 42, "test")

    assert first == second
    assert len({row["id"] for row in first}) == 10
    with pytest.raises(ValueError, match="only 20 eligible examples"):
        _fixed_sample(rows, 21, 42, "test")


def test_ultradomain_sampling_is_balanced_and_deterministic(tmp_path):
    for domain in ("art", "physics", "legal"):
        rows = [
            {
                "_id": f"{domain}-{index}",
                "input": f"Question {index}",
                "answers": [f"Answer {index}"],
                "context": f"Context {index}",
            }
            for index in range(10)
        ]
        (tmp_path / f"{domain}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    first, _ = _stratified_ultradomain_sample(tmp_path, 8, 42)
    second, _ = _stratified_ultradomain_sample(tmp_path, 8, 42)
    counts = {
        domain: sum(row["_s2rag_domain"] == domain for row in first)
        for domain in ("art", "physics", "legal")
    }

    assert first == second
    assert sorted(counts.values()) == [2, 3, 3]

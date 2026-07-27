from s2rag.retrieval.candidates import ScoredFact, aggregate_passages


def test_passage_aggregation_happens_before_fact_output_cutoff():
    facts = [
        ScoredFact(f"fact_{index}", float(100 - index), index, "test")
        for index in range(1, 21)
    ]
    facts.append(ScoredFact("fact_21", 79.0, 21, "test"))
    fact_to_passage = {
        **{f"fact_{index}": "p1" for index in range(1, 21)},
        "fact_21": "p2",
    }

    assert aggregate_passages(facts, fact_to_passage) == ["p1", "p2"]


def test_passage_aggregation_uses_max_score_and_deduplicates():
    facts = [
        ScoredFact("fact_a", 0.2, 1, "test"),
        ScoredFact("fact_b", 0.9, 2, "test"),
        ScoredFact("fact_c", 0.8, 3, "test"),
    ]

    ranking = aggregate_passages(
        facts,
        {"fact_a": "p1", "fact_b": "p1", "fact_c": "p2"},
    )

    assert ranking == ["p1", "p2"]


def test_passage_aggregation_ties_are_deterministic():
    facts = [
        ScoredFact("fact_b", 0.5, 1, "test"),
        ScoredFact("fact_a", 0.5, 1, "test"),
    ]
    mapping = {"fact_a": "passage_a", "fact_b": "passage_b"}

    assert aggregate_passages(facts, mapping) == ["passage_a", "passage_b"]

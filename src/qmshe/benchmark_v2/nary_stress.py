from __future__ import annotations

import hashlib
import random

from qmshe.benchmark_v2.schemas import CandidateView, StructuredFact, V2Example, V2Passage


def build_nary_stress_suite(size: int = 500, seed: int = 42) -> list[CandidateView]:
    """Create role-swap and qualifier-swap cases that binary co-occurrence cannot resolve."""
    rng = random.Random(seed)
    materials = ["alloy", "ceramic", "polymer", "catalyst", "composite"]
    processes = ["annealing", "quenching", "irradiation", "compression", "oxidation"]
    conditions = ["argon", "vacuum", "nitrogen", "steam", "high pressure"]
    results = ["phase alpha", "phase beta", "high conductivity", "low porosity", "red emission"]
    views = []
    for index in range(size):
        material, process, condition, result = (
            rng.choice(materials), rng.choice(processes), rng.choice(conditions), rng.choice(results)
        )
        temperature = f"{rng.randrange(4, 13) * 100} C"
        example_id = f"nary_{index:05d}"
        question = f"What result does {material} produce after {process} under {condition} at {temperature}?"
        specifications = [
            (material, process, condition, temperature, result, True),
            (material, process, rng.choice([x for x in conditions if x != condition]), temperature, result, False),
            (material, process, condition, temperature, rng.choice([x for x in results if x != result]), False),
            (material, rng.choice([x for x in processes if x != process]), condition, temperature, result, False),
            (rng.choice([x for x in materials if x != material]), process, condition, temperature, result, False),
        ]
        passages, facts, gold = [], [], set()
        for fact_index, (mat, proc, cond, temp, outcome, is_gold) in enumerate(specifications):
            passage_id = f"{example_id}_p{fact_index}"
            text = f"Under {cond}, {mat} treated by {proc} at {temp} produces {outcome}."
            passage = V2Passage(passage_id, f"Experiment {index}-{fact_index}", (text,))
            fact_id = f"fact_{hashlib.sha1(f'{example_id}:{fact_index}'.encode()).hexdigest()[:16]}"
            fact = StructuredFact(
                fact_id=fact_id, passage_id=passage_id, sentence_index=0, text=text,
                subject=mat, predicate="produces", object=outcome,
                roles=(("material", mat), ("process", proc), ("condition", cond), ("temperature", temp), ("result", outcome)),
                qualifiers=(("condition", cond), ("temperature", temp)),
                entity_ids=(mat, proc, cond, temp, outcome),
            )
            passages.append(passage)
            facts.append(fact)
            if is_gold:
                gold.add(fact_id)
        example = V2Example(
            example_id=example_id, question=question, answer=result, passages=tuple(passages),
            supporting_facts=frozenset({(passages[0].passage_id, 0)}), query_type="nary_role_binding", level="hard",
        )
        views.append(CandidateView(
            example=example, passages=tuple(passages), facts=tuple(facts),
            gold_fact_ids=frozenset(gold), gold_passage_ids=frozenset({passages[0].passage_id}),
            candidate_count=len(passages),
        ))
    return views

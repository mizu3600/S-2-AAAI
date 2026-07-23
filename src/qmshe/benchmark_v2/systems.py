from __future__ import annotations

import math
import re
import numpy as np
from rank_bm25 import BM25Okapi
from scipy import sparse

from qmshe.benchmark_v2.extraction import canonical_entity
from qmshe.benchmark_v2.schemas import CandidateView, RankingPrediction, StructuredFact


_TOKEN = re.compile(r"[\w'-]+", re.UNICODE)
_QUESTION_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "was", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "with",
}


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text)]


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min(initial=0.0)), float(values.max(initial=0.0))
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


class ControlledTopologyRetriever:
    """Fair, independent topology comparison with one shared ranking contract.

    Every profile starts from identical fact-level BM25 scores, receives exactly
    the same retrieval budget, and is reranked with the same lexical reranker.
    Topology can change candidate order but cannot expand entities into an
    unbounded number of facts after the budget has been applied.
    """

    profiles = ("entity_relation", "reified_fact", "hypergraph")

    def __init__(self, retrieval_budget: int = 60, output_facts: int = 40):
        if retrieval_budget < output_facts:
            raise ValueError("retrieval_budget must be >= output_facts")
        self.retrieval_budget = retrieval_budget
        self.output_facts = output_facts

    def rank(self, view: CandidateView, profile: str) -> RankingPrediction:
        if profile not in self.profiles:
            raise ValueError(f"unknown topology: {profile}")
        facts = view.facts
        if not facts:
            return RankingPrediction(system=f"controlled_{profile}", fact_ranking=(), passage_ranking=(), path=())
        question_tokens = tokenize(view.example.question)
        corpus = [tokenize(f"{fact.subject} {fact.predicate} {fact.object} {fact.text}") for fact in facts]
        raw = _unit(np.asarray(BM25Okapi(corpus).get_scores(question_tokens)))
        incidence, entity_names = self._incidence(facts, profile, view.example.question)
        z1 = self._propagate(incidence, raw, profile)
        z2 = self._propagate(incidence, z1, profile)
        bands = np.vstack([raw, _unit(z2), _unit(z1 - z2), _unit(raw - z1)])
        gate = self._gate(view.example.question, view.example.query_type)
        spectral = gate @ bands
        initial = np.argsort(-spectral, kind="stable")[: self.retrieval_budget]
        reranker_scores = self._shared_reranker(view.example.question, facts, initial, raw)
        ordered = initial[np.argsort(-reranker_scores, kind="stable")][: self.output_facts]
        fact_ranking = tuple(facts[index].fact_id for index in ordered)
        passage_ranking = tuple(dict.fromkeys(facts[index].passage_id for index in ordered))
        citations = fact_ranking[: min(2, len(fact_ranking))]
        return RankingPrediction(
            system=f"controlled_{profile}",
            fact_ranking=fact_ranking,
            passage_ranking=passage_ranking,
            path=passage_ranking[:2],
            answer=self._extractive_answer(view.example.question, [facts[index] for index in ordered[:5]]),
            citations=citations,
            diagnostics={
                "candidate_passages": view.candidate_count,
                "candidate_facts": len(facts),
                "retrieval_budget": self.retrieval_budget,
                "reranker_inputs": len(initial),
                "gate": gate.tolist(),
                "incidence_entities": len(entity_names),
                "ranking_origin": "internal_fact_scores",
            },
        )

    @staticmethod
    def _incidence(facts: tuple[StructuredFact, ...], profile: str, question: str):
        entity_to_column: dict[str, int] = {}
        rows, columns, values = [], [], []
        question_terms = set(tokenize(question)) - _QUESTION_STOP
        for row, fact in enumerate(facts):
            role_lookup = {canonical_entity(value): role for role, value in fact.roles}
            for entity in fact.entity_ids:
                column = entity_to_column.setdefault(entity, len(entity_to_column))
                weight = 1.0
                if profile == "reified_fact":
                    role = role_lookup.get(entity, "mention")
                    weight = 1.25 if role not in {"mention", "topic", "statement"} else 0.8
                elif profile == "hypergraph":
                    overlap = len(set(tokenize(entity)) & question_terms)
                    role = role_lookup.get(entity, "mention")
                    weight = (1.2 if role not in {"mention", "topic", "statement"} else 0.65) * (1 + 0.2 * overlap)
                rows.append(row)
                columns.append(column)
                values.append(weight)
        matrix = sparse.csr_matrix((values, (rows, columns)), shape=(len(facts), len(entity_to_column)))
        return matrix, tuple(entity_to_column)

    @staticmethod
    def _propagate(incidence: sparse.csr_matrix, scores: np.ndarray, profile: str) -> np.ndarray:
        if incidence.shape[1] == 0:
            return scores.copy()
        entity_degree = np.asarray(incidence.sum(axis=0)).ravel()
        fact_degree = np.asarray(incidence.sum(axis=1)).ravel()
        power = 1.0 if profile == "entity_relation" else (0.75 if profile == "reified_fact" else 1.25)
        inv_entity = np.power(np.maximum(entity_degree, 1e-9), -power)
        entity_signal = incidence.T @ (scores / np.maximum(fact_degree, 1e-9))
        propagated = incidence @ (entity_signal * inv_entity)
        return _unit(np.asarray(propagated).ravel())

    @staticmethod
    def _gate(question: str, query_type: str) -> np.ndarray:
        lowered = question.casefold()
        if query_type == "comparison" or re.search(r"\b(?:both|more|less|older|younger|same)\b", lowered):
            weights = np.array([0.25, 0.15, 0.35, 0.25])
        elif re.search(r"\b(?:which|who|where).*(?:besides|other|whose|that)\b", lowered):
            weights = np.array([0.20, 0.20, 0.50, 0.10])
        else:
            weights = np.array([0.35, 0.20, 0.35, 0.10])
        return weights / weights.sum()

    @staticmethod
    def _shared_reranker(question: str, facts: tuple[StructuredFact, ...], indices: np.ndarray, raw: np.ndarray) -> np.ndarray:
        query_terms = set(tokenize(question)) - _QUESTION_STOP
        scores = []
        for index in indices:
            fact = facts[int(index)]
            fact_terms = set(tokenize(f"{fact.subject} {fact.predicate} {fact.object} {fact.text}"))
            overlap = len(query_terms & fact_terms) / math.sqrt(max(len(query_terms) * len(fact_terms), 1))
            qualifier = 0.05 if any(value in question for _, value in fact.qualifiers) else 0.0
            scores.append(0.65 * raw[int(index)] + 0.35 * overlap + qualifier)
        return np.asarray(scores)

    @staticmethod
    def _extractive_answer(question: str, facts: list[StructuredFact]) -> str:
        question_entities = {canonical_entity(value) for value in re.findall(r"\b[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*)*", question)}
        candidates = []
        for rank, fact in enumerate(facts, 1):
            for entity in fact.entity_ids:
                if entity not in question_entities and len(entity) > 2:
                    candidates.append((1 / rank, entity))
            for _, value in fact.qualifiers:
                if value not in question:
                    candidates.append((0.8 / rank, value))
        if not candidates:
            return ""
        return max(candidates)[1]

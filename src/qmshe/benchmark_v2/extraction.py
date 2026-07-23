from __future__ import annotations

import hashlib
import re

from qmshe.benchmark_v2.schemas import StructuredFact, V2Passage


_PATTERNS = (
    (r"\bwas born (?:on|in)\b", "born", "person", "birth"),
    (r"\b(?:is|was) located in\b", "located_in", "place", "location"),
    (r"\b(?:is|was) (?:directed|written|produced) by\b", "created_by", "work", "creator"),
    (r"\b(?:is|was) (?:founded|established) (?:by|in)\b", "founded", "organization", "founding"),
    (r"\b(?:won|wins|received|awarded)\b", "won", "recipient", "award"),
    (r"\b(?:played|portrayed|starred)\b", "performed", "performer", "work"),
    (r"\b(?:married|spouse of)\b", "spouse", "person", "spouse"),
    (r"\b(?:member of|part of|belongs to)\b", "member_of", "member", "group"),
    (r"\b(?:released|published|opened|premiered)\b", "released", "work", "date"),
    (r"\b(?:is|was|are|were)\b", "is_a", "subject", "object"),
)
_ENTITY = re.compile(r"\b(?:[A-Z][\w'.-]*)(?:\s+(?:of|the|and|de|[A-Z][\w'.-]*)){0,5}\b")
_YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?(?:\s?(?:km|m|kg|years?|million|billion|%))?\b", re.I)


def canonical_entity(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)\s*", " ", value)
    value = re.sub(r"[^\w]+", " ", value.casefold()).strip()
    return re.sub(r"\s+", " ", value)


class StructuredFactExtractor:
    """Deterministic role/qualifier extractor for reproducible retrieval tests.

    It is intentionally label-free and is not presented as a replacement for
    the optional LLM extractor. Unlike v1, titles are provenance, not generic
    n-ary arguments, and every emitted argument has a named semantic role.
    """

    version = "rule-roles-v2"

    def extract_passage(self, passage: V2Passage) -> list[StructuredFact]:
        return [self.extract_sentence(passage, index, text) for index, text in enumerate(passage.sentences)]

    def extract_sentence(self, passage: V2Passage, sentence_index: int, text: str) -> StructuredFact:
        predicate, subject_role, object_role, split = "states", "topic", "statement", None
        for pattern, name, left_role, right_role in _PATTERNS:
            match = re.search(pattern, text, re.I)
            if match:
                predicate, subject_role, object_role, split = name, left_role, right_role, match
                break
        subject = text[: split.start()].strip(" ,;:-") if split else passage.title
        object_value = text[split.end() :].strip(" ,;:-") if split else text
        subject = subject or passage.title
        object_value = object_value or text
        roles = [(subject_role, subject), (object_role, object_value)]
        entities = [passage.title, subject]
        entities.extend(match.group(0).strip() for match in _ENTITY.finditer(text))
        unique_entities = tuple(dict.fromkeys(x for x in (canonical_entity(v) for v in entities) if x))
        qualifiers = [("year", year) for year in _YEAR.findall(text)]
        qualifiers.extend(("quantity", value) for value in _NUMBER.findall(text) if not _YEAR.fullmatch(value))
        if re.search(r"\b(?:during|after|before|when|while|under|with)\b", text, re.I):
            qualifiers.append(("condition", text))
        fact_id = f"fact_{hashlib.sha1(f'{passage.passage_id}:{sentence_index}'.encode()).hexdigest()[:16]}"
        return StructuredFact(
            fact_id=fact_id,
            passage_id=passage.passage_id,
            sentence_index=sentence_index,
            text=text,
            subject=subject,
            predicate=predicate,
            object=object_value,
            roles=tuple(roles),
            qualifiers=tuple(qualifiers),
            entity_ids=unique_entities,
        )

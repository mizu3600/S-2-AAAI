import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from s2rag.extraction import OPEN_DOMAIN_EXTRACTION_PROTOCOL
from s2rag.extraction.canonicalizer import canonicalize_entities, normalize_name
from s2rag.extraction.entity_extractor import (
    ENTITY_SYSTEM_PROMPT,
    extract_entities_with_llm,
)
from s2rag.extraction.fact_extractor import (
    FACT_SYSTEM_PROMPT,
    extract_facts_with_llm,
)
from s2rag.ingest.schemas import Chunk, Entity, EvidenceHyperedge
from s2rag.settings import get_settings


LEGACY_EXTRACTION_PROTOCOL = "shared_deepseek_entity_fact_v1"
LEGACY_ENTITY_SYSTEM_PROMPT = """You extract entities explicitly mentioned in source text.
Return JSON with key entities. Each entity contains canonical_name, aliases, entity_type,
description, and mention. aliases and mention must be exact surface forms from the supplied text.
Use a concise lower_snake_case entity_type. Do not infer entities that are not explicitly present."""
LEGACY_FACT_SYSTEM_PROMPT = """You extract only explicitly supported n-ary facts from source text.
Return JSON with key facts. Each fact contains predicate, arguments [{role, entity_id}],
qualifiers, evidence_sentence, and confidence. Use only entity_id values from known_entities.
evidence_sentence must quote the supplied text. Never infer causality or invent entities."""


@dataclass(frozen=True)
class ComparisonCase:
    case_id: str
    domain: str
    text: str
    expected_entities: tuple[str, ...]
    expected_facts: tuple[tuple[str, ...], ...]


CASES = (
    ComparisonCase(
        case_id="encyclopedia",
        domain="encyclopedia",
        text=(
            "Radio City is India's first private FM radio station and started on 3 July 2001. "
            "It launched PlanetRadiocity.com in May 2008 and offers Hindi and English songs."
        ),
        expected_entities=(
            "Radio City",
            "India",
            "private FM radio station",
            "3 July 2001",
            "PlanetRadiocity.com",
            "May 2008",
            "Hindi",
            "English",
        ),
        expected_facts=(
            ("Radio City", "India"),
            ("Radio City", "private FM radio station"),
            ("Radio City", "3 July 2001"),
            ("Radio City", "PlanetRadiocity.com", "May 2008"),
            ("Radio City", "Hindi"),
            ("Radio City", "English"),
        ),
    ),
    ComparisonCase(
        case_id="technical_evaluation",
        domain="computer_science",
        text=(
            "In 2024, the Atlas team evaluated S2RAG on HotpotQA using 500 questions. "
            "It reported an 8% recall gain under a zero-shot setting."
        ),
        expected_entities=(
            "2024",
            "Atlas team",
            "S2RAG",
            "HotpotQA",
            "500 questions",
            "8%",
            "recall",
            "zero-shot setting",
        ),
        expected_facts=(
            ("Atlas team", "S2RAG", "HotpotQA", "500 questions", "2024"),
            ("S2RAG", "8%", "recall", "zero-shot setting"),
        ),
    ),
    ComparisonCase(
        case_id="clinical_trial",
        domain="medicine",
        text=(
            "During a 12-week trial, metformin reduced HbA1c by 1.2 percentage points in "
            "240 adults with type 2 diabetes, compared with placebo."
        ),
        expected_entities=(
            "12-week trial",
            "metformin",
            "HbA1c",
            "1.2 percentage points",
            "240 adults",
            "type 2 diabetes",
            "placebo",
        ),
        expected_facts=(
            (
                "metformin",
                "HbA1c",
                "1.2 percentage points",
                "240 adults",
                "type 2 diabetes",
                "placebo",
            ),
        ),
    ),
    ComparisonCase(
        case_id="acquisition",
        domain="business",
        text=(
            "On 5 March 2025, Orion Labs acquired Nova Analytics for $42 million in Shanghai. "
            "Mei Chen advised Orion Labs, while Luis Park represented Nova Analytics."
        ),
        expected_entities=(
            "5 March 2025",
            "Orion Labs",
            "Nova Analytics",
            "$42 million",
            "Shanghai",
            "Mei Chen",
            "Luis Park",
        ),
        expected_facts=(
            ("Orion Labs", "Nova Analytics", "$42 million", "Shanghai", "5 March 2025"),
            ("Mei Chen", "Orion Labs"),
            ("Luis Park", "Nova Analytics"),
        ),
    ),
    ComparisonCase(
        case_id="no_proper_noun",
        domain="general_science",
        text=(
            "Water boils at 100 degrees Celsius under standard atmospheric pressure and "
            "freezes at 0 degrees Celsius."
        ),
        expected_entities=(
            "Water",
            "100 degrees Celsius",
            "standard atmospheric pressure",
            "0 degrees Celsius",
        ),
        expected_facts=(
            ("Water", "100 degrees Celsius", "standard atmospheric pressure"),
            ("Water", "0 degrees Celsius"),
        ),
    ),
    ComparisonCase(
        case_id="correlation",
        domain="observational_research",
        text=(
            "Users of Feature A showed 15% lower latency than users of Feature B, but the "
            "observational study did not establish causation."
        ),
        expected_entities=(
            "Feature A",
            "15%",
            "latency",
            "Feature B",
            "observational study",
            "causation",
        ),
        expected_facts=(
            ("Feature A", "15%", "latency", "Feature B"),
            ("observational study", "causation"),
        ),
    ),
)


class BoundedDeepSeekJsonClient:
    def __init__(self, max_attempts: int = 3):
        settings = get_settings()
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url.rstrip("/")
        self.model = settings.deepseek_model
        self.temperature = settings.deepseek_temperature
        self.max_tokens = settings.deepseek_max_tokens
        self.timeout = settings.request_timeout
        self.max_attempts = max_attempts
        self.calls: list[dict] = []

    def complete_json(self, system: str, prompt: str) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "thinking": {"type": "disabled"},
                        "response_format": {"type": "json_object"},
                    },
                    timeout=self.timeout,
                )
                if response.status_code in {401, 403}:
                    raise RuntimeError(
                        f"DeepSeek authentication failed with status {response.status_code}"
                    )
                response.raise_for_status()
                body = response.json()
                payload = json.loads(body["choices"][0]["message"]["content"])
                if not isinstance(payload, dict):
                    raise ValueError("DeepSeek JSON response is not an object")
                usage = body.get("usage", {})
                self.calls.append(
                    {
                        "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "attempt": attempt,
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "prompt_cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens", 0)),
                        "prompt_cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens", 0)),
                        "payload": payload,
                    }
                )
                return payload
            except RuntimeError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    time.sleep(2 ** (attempt - 1))
        raise RuntimeError(
            f"DeepSeek request failed after {self.max_attempts} attempts: {last_error}"
        )


PROTOCOLS = {
    LEGACY_EXTRACTION_PROTOCOL: (
        LEGACY_ENTITY_SYSTEM_PROMPT,
        LEGACY_FACT_SYSTEM_PROMPT,
    ),
    OPEN_DOMAIN_EXTRACTION_PROTOCOL: (
        ENTITY_SYSTEM_PROMPT,
        FACT_SYSTEM_PROMPT,
    ),
}


def evaluate_protocol(
    protocol: str,
    entity_prompt: str,
    fact_prompt: str,
    client: BoundedDeepSeekJsonClient,
) -> dict:
    case_results = []
    for index, case in enumerate(CASES, 1):
        print(f"[{protocol}] {index}/{len(CASES)} {case.case_id}", flush=True)
        chunk = Chunk(
            chunk_id=f"chunk_{case.case_id}",
            document_id=f"doc_{case.case_id}",
            section=case.domain,
            text=case.text,
            start_char=0,
            end_char=len(case.text),
        )
        call_start = len(client.calls)
        entities = canonicalize_entities(
            extract_entities_with_llm(
                [chunk],
                client,
                system_prompt=entity_prompt,
            )
        )
        entity_call = client.calls[-1]
        facts = extract_facts_with_llm(
            [chunk],
            entities,
            client,
            system_prompt=fact_prompt,
        )
        fact_call = client.calls[-1]
        case_results.append(
            _score_case(
                case,
                entities,
                facts,
                entity_call["payload"],
                fact_call["payload"],
                sum(call["latency_ms"] for call in client.calls[call_start:]),
                sum(call["prompt_tokens"] for call in client.calls[call_start:]),
                sum(call["completion_tokens"] for call in client.calls[call_start:]),
                sum(call["prompt_cache_hit_tokens"] for call in client.calls[call_start:]),
                sum(call["prompt_cache_miss_tokens"] for call in client.calls[call_start:]),
            )
        )
    return {
        "protocol": protocol,
        "entity_prompt_sha256": hashlib.sha256(entity_prompt.encode()).hexdigest(),
        "fact_prompt_sha256": hashlib.sha256(fact_prompt.encode()).hexdigest(),
        "summary": _summarize(case_results),
        "cases": case_results,
    }


def _score_case(
    case: ComparisonCase,
    entities: list[Entity],
    facts: list[EvidenceHyperedge],
    raw_entity_payload: dict,
    raw_fact_payload: dict,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cache_hit_tokens: int,
    prompt_cache_miss_tokens: int,
) -> dict:
    entity_coverage = [
        expected
        for expected in case.expected_entities
        if any(_entity_matches(entity, expected) for entity in entities)
    ]
    fact_coverage = [
        list(expected)
        for expected in case.expected_facts
        if _fact_is_covered(expected, entities, facts)
    ]
    raw_entities = raw_entity_payload.get("entities", [])
    raw_entities = raw_entities if isinstance(raw_entities, list) else []
    raw_facts = raw_fact_payload.get("facts", [])
    raw_facts = raw_facts if isinstance(raw_facts, list) else []
    known_ids = {entity.entity_id for entity in entities}
    grounded_raw_facts = sum(
        _raw_evidence_is_grounded(raw, case.text) for raw in raw_facts if isinstance(raw, dict)
    )
    raw_unknown_argument_count = sum(
        1
        for raw in raw_facts
        if isinstance(raw, dict)
        for argument in raw.get("arguments", [])
        if isinstance(argument, dict)
        and str(argument.get("entity_id", "")).strip() not in known_ids
    )
    return {
        "case": asdict(case),
        "metrics": {
            "entity_count": len(entities),
            "expected_entity_count": len(case.expected_entities),
            "covered_entity_count": len(entity_coverage),
            "raw_entity_count": len(raw_entities),
            "fact_count": len(facts),
            "expected_fact_count": len(case.expected_facts),
            "covered_fact_count": len(fact_coverage),
            "raw_fact_count": len(raw_facts),
            "grounded_raw_fact_count": grounded_raw_facts,
            "raw_unknown_argument_count": raw_unknown_argument_count,
            "nary_fact_count": sum(len(fact.arguments) >= 3 for fact in facts),
            "total_fact_arguments": sum(len(fact.arguments) for fact in facts),
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
        },
        "covered_entities": entity_coverage,
        "covered_facts": fact_coverage,
        "entities": [entity.model_dump(mode="json") for entity in entities],
        "facts": [fact.model_dump(mode="json") for fact in facts],
        "raw_entity_payload": raw_entity_payload,
        "raw_fact_payload": raw_fact_payload,
    }


def _summarize(case_results: list[dict]) -> dict:
    totals = {
        key: sum(result["metrics"][key] for result in case_results)
        for key in (
            "entity_count",
            "expected_entity_count",
            "covered_entity_count",
            "raw_entity_count",
            "fact_count",
            "expected_fact_count",
            "covered_fact_count",
            "raw_fact_count",
            "grounded_raw_fact_count",
            "raw_unknown_argument_count",
            "nary_fact_count",
            "total_fact_arguments",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        )
    }
    return {
        **totals,
        "entity_coverage": _ratio(totals["covered_entity_count"], totals["expected_entity_count"]),
        "fact_coverage": _ratio(totals["covered_fact_count"], totals["expected_fact_count"]),
        "entity_retention": _ratio(totals["entity_count"], totals["raw_entity_count"]),
        "fact_retention": _ratio(totals["fact_count"], totals["raw_fact_count"]),
        "raw_evidence_grounding": _ratio(
            totals["grounded_raw_fact_count"], totals["raw_fact_count"]
        ),
        "nary_fact_rate": _ratio(totals["nary_fact_count"], totals["fact_count"]),
        "mean_fact_arity": _ratio(totals["total_fact_arguments"], totals["fact_count"]),
    }


def _entity_matches(entity: Entity, expected: str) -> bool:
    expected_name = normalize_name(expected)
    surfaces = [entity.canonical_name, *entity.aliases]
    surfaces.extend(
        mention.split(":", 2)[2] for mention in entity.source_mentions if mention.count(":") >= 2
    )
    normalized_surfaces = [normalize_name(surface) for surface in surfaces]
    return any(
        expected_name == surface or expected_name in surface or surface in expected_name
        for surface in normalized_surfaces
        if surface
    )


def _fact_is_covered(
    expected: tuple[str, ...],
    entities: list[Entity],
    facts: list[EvidenceHyperedge],
) -> bool:
    matching_ids = [
        {entity.entity_id for entity in entities if _entity_matches(entity, expected_entity)}
        for expected_entity in expected
    ]
    if any(not ids for ids in matching_ids):
        return False
    return any(
        all(ids & {argument.entity_id for argument in fact.arguments} for ids in matching_ids)
        for fact in facts
    )


def _raw_evidence_is_grounded(raw: dict, text: str) -> bool:
    evidence = raw.get("evidence_sentence")
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    normalized_evidence = " ".join(evidence.split())
    normalized_text = " ".join(text.split())
    return normalized_evidence in normalized_text


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _write_report(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    old = result["protocols"][LEGACY_EXTRACTION_PROTOCOL]["summary"]
    new = result["protocols"][OPEN_DOMAIN_EXTRACTION_PROTOCOL]["summary"]
    rows = [
        ("Entity coverage", old["entity_coverage"], new["entity_coverage"], "percent"),
        ("Fact coverage", old["fact_coverage"], new["fact_coverage"], "percent"),
        ("Retained entities", old["entity_count"], new["entity_count"], "number"),
        ("Retained facts", old["fact_count"], new["fact_count"], "number"),
        ("Entity retention", old["entity_retention"], new["entity_retention"], "percent"),
        ("Fact retention", old["fact_retention"], new["fact_retention"], "percent"),
        ("N-ary fact rate", old["nary_fact_rate"], new["nary_fact_rate"], "percent"),
        ("Mean fact arity", old["mean_fact_arity"], new["mean_fact_arity"], "number"),
        (
            "Raw evidence grounding",
            old["raw_evidence_grounding"],
            new["raw_evidence_grounding"],
            "percent",
        ),
        (
            "Unknown raw arguments",
            old["raw_unknown_argument_count"],
            new["raw_unknown_argument_count"],
            "number",
        ),
        ("Prompt tokens", old["prompt_tokens"], new["prompt_tokens"], "number"),
        (
            "Completion tokens",
            old["completion_tokens"],
            new["completion_tokens"],
            "number",
        ),
        (
            "Prompt cache hit tokens",
            old["prompt_cache_hit_tokens"],
            new["prompt_cache_hit_tokens"],
            "number",
        ),
        ("Latency", old["latency_ms"], new["latency_ms"], "milliseconds"),
    ]
    lines = [
        "# Extraction Protocol Small-Scale A/B",
        "",
        f"- Model: `{result['model']}`",
        f"- Temperature: `{result['temperature']}`",
        f"- Cases: `{len(CASES)}`",
        "- Design: old and new prompts use the same current grounding and ID validator.",
        "",
        "| Metric | Legacy v1 | OpenIE n-ary v2 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, old_value, new_value, kind in rows:
        if kind == "percent":
            cells = (
                f"{old_value:.1%}",
                f"{new_value:.1%}",
                f"{new_value - old_value:+.1%}",
            )
        elif kind == "milliseconds":
            cells = (
                f"{old_value:.0f} ms",
                f"{new_value:.0f} ms",
                f"{new_value - old_value:+.0f} ms",
            )
        else:
            cells = (
                f"{old_value:.2f}",
                f"{new_value:.2f}",
                f"{new_value - old_value:+.2f}",
            )
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend(
        [
            "",
            "## Per Case",
            "",
            "| Case | Legacy entity | V2 entity | Legacy fact | V2 fact |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    old_cases = {
        case["case"]["case_id"]: case
        for case in result["protocols"][LEGACY_EXTRACTION_PROTOCOL]["cases"]
    }
    new_cases = {
        case["case"]["case_id"]: case
        for case in result["protocols"][OPEN_DOMAIN_EXTRACTION_PROTOCOL]["cases"]
    }
    for case in CASES:
        old_metrics = old_cases[case.case_id]["metrics"]
        new_metrics = new_cases[case.case_id]["metrics"]
        lines.append(
            f"| {case.case_id} | "
            f"{old_metrics['covered_entity_count']}/{old_metrics['expected_entity_count']} | "
            f"{new_metrics['covered_entity_count']}/{new_metrics['expected_entity_count']} | "
            f"{old_metrics['covered_fact_count']}/{old_metrics['expected_fact_count']} | "
            f"{new_metrics['covered_fact_count']}/{new_metrics['expected_fact_count']} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This is a six-case directional smoke test, not a statistically powered benchmark.",
            "Coverage uses hand-authored expected entity surfaces and relation argument sets; it does",
            "not use an LLM judge. Inspect `comparison.json` for every raw and retained extraction.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/extraction_ab"))
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")

    client = BoundedDeepSeekJsonClient(max_attempts=args.max_attempts)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    protocols = {
        protocol: evaluate_protocol(protocol, prompts[0], prompts[1], client)
        for protocol, prompts in PROTOCOLS.items()
    }
    result = {
        "run_id": run_id,
        "model": client.model,
        "temperature": client.temperature,
        "max_tokens": client.max_tokens,
        "comparison_design": "shared_current_grounding_validator",
        "protocols": protocols,
    }
    _write_report(result, output_dir)
    print(f"report={output_dir / 'report.md'}")
    print(f"details={output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()

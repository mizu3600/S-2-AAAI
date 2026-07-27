from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field as dataclass_field, replace
from pathlib import Path

from s2rag.benchmarks.schemas import BenchmarkExample, BenchmarkSuite
from s2rag.generation.generator import EvidenceGenerator
from s2rag.evaluation.metrics import extract_citation_ids


@dataclass(frozen=True)
class SystemCapability:
    supports_passage_ranking: bool
    supports_fact_ranking: bool
    supports_answer_generation: bool
    citation_capability: str
    generation_protocol: str | None

    def __post_init__(self) -> None:
        if self.citation_capability not in {"none", "passage", "fact", "both"}:
            raise ValueError(f"invalid citation capability: {self.citation_capability}")


@dataclass(frozen=True)
class ExternalBaselineSpec:
    key: str
    display_name: str
    repository: str
    ranking_contract: str
    capability: SystemCapability


@dataclass(frozen=True)
class ExternalBaselineResult:
    system: str
    example_id: str
    capability: SystemCapability
    status: str
    document_ranking: list[str]
    document_ranking_declared: bool
    fact_ranking: list[str]
    fact_ranking_declared: bool
    answer: str
    citations: list[str]
    citations_declared: bool
    citation_source: str
    citation_level: str
    citation_mapping_complete: bool
    citation_parse_failed: bool
    received_citation_count: int
    unmapped_citation_ids: list[str]
    indexing_seconds: float | None
    retrieval_seconds: float | None
    total_seconds: float | None
    error: str | None
    ranking_origin: str
    generation_protocol: str | None
    generation_trace: dict
    generation_protocol_matched: bool
    mapping_coverage: float
    unmapped_ranking_ids: list[str]
    raw: dict
    shared_model_trace: dict = dataclass_field(default_factory=dict)
    shared_model_protocol_matched: bool = True


_NATIVE_PASSAGE_CAPABILITY = SystemCapability(
    supports_passage_ranking=True,
    supports_fact_ranking=False,
    supports_answer_generation=True,
    citation_capability="none",
    generation_protocol="unified_concise_deepseek_v1",
)


BASELINE_SPECS = {
    "graphrag": ExternalBaselineSpec(
        "graphrag",
        "Microsoft GraphRAG",
        "https://github.com/microsoft/graphrag",
        "ranked community or chunk identifiers",
        _NATIVE_PASSAGE_CAPABILITY,
    ),
    "lightrag": ExternalBaselineSpec(
        "lightrag",
        "LightRAG",
        "https://github.com/HKUDS/LightRAG",
        "ranked chunk or entity identifiers",
        _NATIVE_PASSAGE_CAPABILITY,
    ),
    "pathrag": ExternalBaselineSpec(
        "pathrag",
        "PathRAG",
        "https://github.com/BUPT-GAMMA/PathRAG",
        "ranked path-context identifiers",
        _NATIVE_PASSAGE_CAPABILITY,
    ),
    "hypergraphrag": ExternalBaselineSpec(
        "hypergraphrag",
        "HyperGraphRAG",
        "https://github.com/LHRLAB/HyperGraphRAG",
        "ranked hyperedge or passage identifiers",
        _NATIVE_PASSAGE_CAPABILITY,
    ),
    "hipporag2": ExternalBaselineSpec(
        "hipporag2",
        "HippoRAG 2",
        "https://github.com/OSU-NLP-Group/HippoRAG",
        "ranked passage identifiers from native PPR",
        _NATIVE_PASSAGE_CAPABILITY,
    ),
    "cograg": ExternalBaselineSpec(
        "cograg",
        "Cog-RAG",
        "https://github.com/haoohu/Cog-RAG",
        "ranked passage identifiers from dual-hypergraph retrieval",
        _NATIVE_PASSAGE_CAPABILITY,
    ),
    "hgrag": ExternalBaselineSpec(
        "hgrag",
        "HGRAG",
        "https://github.com/MF-AIR/HGRAG",
        "ranked passage identifiers from hypergraph diffusion",
        _NATIVE_PASSAGE_CAPABILITY,
    ),
    "hyperrag": ExternalBaselineSpec(
        "hyperrag",
        "Hyper-RAG",
        "https://github.com/iMoonLab/Hyper-RAG",
        "ranked hypergraph passage identifiers",
        _NATIVE_PASSAGE_CAPABILITY,
    ),
}


class ExternalBaselineAdapter:
    spec: ExternalBaselineSpec
    ranking_fields = ("document_ranking", "ranked_ids", "ranking", "retrieved_ids", "contexts")

    def __init__(self, spec: ExternalBaselineSpec):
        self.spec = spec

    def normalize(self, row: dict, example: BenchmarkExample) -> ExternalBaselineResult:
        (
            document_ranking,
            document_ranking_declared,
            mapping_coverage,
            unmapped_ranking_ids,
        ) = self._normalize_document_ranking(row, example)
        fact_ranking = _as_list(row.get("fact_ranking"))
        (
            citations,
            citations_declared,
            citation_source,
            citation_mapping_complete,
            citation_parse_failed,
            received_citation_count,
            unmapped_citation_ids,
        ) = self._normalize_citations(row, example)
        generation_trace = row.get("generation_trace", {})
        if not isinstance(generation_trace, dict):
            generation_trace = {}
        status = str(row.get("status", "success"))
        error = row.get("error")
        if error and status == "success":
            status = "failed"
        elif not document_ranking_declared:
            status = "unscorable"
        elif unmapped_ranking_ids and mapping_coverage == 0.0:
            status = "unscorable"
        return ExternalBaselineResult(
            system=f"external:{self.spec.key}",
            example_id=example.example_id,
            capability=self.spec.capability,
            status=status,
            document_ranking=document_ranking,
            document_ranking_declared=document_ranking_declared,
            fact_ranking=[str(item) for item in fact_ranking],
            fact_ranking_declared="fact_ranking" in row,
            answer=str(row.get("answer", row.get("response", "")) or ""),
            citations=citations,
            citations_declared=citations_declared,
            citation_source=citation_source,
            citation_level=self.spec.capability.citation_capability,
            citation_mapping_complete=citation_mapping_complete,
            citation_parse_failed=citation_parse_failed,
            received_citation_count=received_citation_count,
            unmapped_citation_ids=unmapped_citation_ids,
            indexing_seconds=_number(row.get("indexing_seconds", row.get("index_seconds"))),
            retrieval_seconds=_number(row.get("retrieval_seconds", row.get("latency_seconds"))),
            total_seconds=_number(row.get("total_seconds", row.get("elapsed_seconds"))),
            error=error,
            ranking_origin=row.get("ranking_origin", f"{self.spec.key}_native_result"),
            generation_protocol=row.get("generation_protocol"),
            generation_trace=generation_trace,
            generation_protocol_matched=False,
            mapping_coverage=mapping_coverage,
            unmapped_ranking_ids=unmapped_ranking_ids,
            raw=row,
            shared_model_trace=(
                row.get("shared_model_trace", {})
                if isinstance(row.get("shared_model_trace", {}), dict)
                else {}
            ),
        )

    def missing(self, example: BenchmarkExample) -> ExternalBaselineResult:
        return ExternalBaselineResult(
            system=f"external:{self.spec.key}",
            example_id=example.example_id,
            capability=self.spec.capability,
            status="missing",
            document_ranking=[],
            document_ranking_declared=False,
            fact_ranking=[],
            fact_ranking_declared=False,
            answer="",
            citations=[],
            citations_declared=False,
            citation_source="missing",
            citation_level=self.spec.capability.citation_capability,
            citation_mapping_complete=True,
            citation_parse_failed=False,
            received_citation_count=0,
            unmapped_citation_ids=[],
            indexing_seconds=None,
            retrieval_seconds=None,
            total_seconds=None,
            error="missing result row",
            ranking_origin=f"{self.spec.key}_native_result",
            generation_protocol=None,
            generation_trace={},
            generation_protocol_matched=False,
            mapping_coverage=0.0,
            unmapped_ranking_ids=[],
            raw={},
            shared_model_trace={},
        )

    def _normalize_document_ranking(
        self, row: dict, example: BenchmarkExample
    ) -> tuple[list[str], bool, float, list[str]]:
        raw_ranking = []
        ranking_declared = False
        for field in self.ranking_fields:
            if field in row:
                ranking_declared = True
                raw_ranking = _as_list(row[field])
                break
        source_map = row.get("source_id_map", row.get("id_map", {}))
        mapped, unmapped = [], []
        mapped_candidates = 0
        for candidate in raw_ranking:
            candidate_mapping = _map_candidate(candidate, example, source_map)
            if candidate_mapping:
                mapped_candidates += 1
                mapped.extend(candidate_mapping)
            else:
                label = _candidate_label(candidate)
                unmapped.append(label)
                mapped.append(f"__unmapped__:{label}")
        coverage = mapped_candidates / len(raw_ranking) if raw_ranking else 0.0
        return list(dict.fromkeys(mapped)), ranking_declared, coverage, unmapped

    def _normalize_citations(
        self, row: dict, example: BenchmarkExample
    ) -> tuple[list[str], bool, str, bool, bool, int, list[str]]:
        explicit_field = next(
            (field for field in ("citations", "citation_ids") if field in row),
            None,
        )
        if explicit_field is not None:
            citations = _as_list(row[explicit_field])
            citation_source = "explicit"
            citations_declared = True
            parse_failed = False
        else:
            answer = str(row.get("answer", row.get("response", "")) or "")
            passage_labels = {
                label
                for passage in example.passages
                for label in (passage.passage_id, passage.title)
            }
            citations = extract_citation_ids(answer, allowed_ids=passage_labels)
            if self.spec.capability.citation_capability in {"fact", "both"}:
                citations.extend(extract_citation_ids(answer))
            citations = list(dict.fromkeys(citations))
            citation_source = "answer"
            citations_declared = False
            bracket_values = [item.strip() for item in re.findall(r"\[([^\[\]]+)\]", answer)]
            parse_failed = (
                bool(bracket_values)
                and not citations
                and any(not item.isdigit() for item in bracket_values)
            )
        received_count = len(citations)
        capability = self.spec.capability.citation_capability
        if capability == "none" or not citations:
            return (
                [],
                citations_declared,
                citation_source,
                True,
                parse_failed,
                received_count,
                [],
            )

        source_map = row.get("source_id_map", row.get("id_map", {}))
        normalized: list[str] = []
        unmapped: list[str] = []
        for citation in citations:
            mapped = _map_candidate(citation, example, source_map)
            if mapped and capability in {"passage", "both"}:
                normalized.extend(mapped)
            elif capability in {"fact", "both"} and str(citation).startswith("fact_"):
                normalized.append(str(citation))
            else:
                unmapped.append(_candidate_label(citation))
        return (
            list(dict.fromkeys(normalized)),
            citations_declared,
            citation_source,
            not unmapped,
            parse_failed,
            received_count,
            unmapped,
        )


class GraphRAGAdapter(ExternalBaselineAdapter):
    ranking_fields = (
        "community_ranking",
        "chunk_ranking",
        "document_ranking",
        "ranking",
        "contexts",
    )


class LightRAGAdapter(ExternalBaselineAdapter):
    ranking_fields = ("chunk_ranking", "entity_ranking", "document_ranking", "ranking", "contexts")


class PathRAGAdapter(ExternalBaselineAdapter):
    ranking_fields = ("path_context_ranking", "document_ranking", "ranking", "contexts")


class HyperGraphRAGAdapter(ExternalBaselineAdapter):
    ranking_fields = ("hyperedge_ranking", "document_ranking", "ranking", "contexts")


class HippoRAG2Adapter(ExternalBaselineAdapter):
    ranking_fields = ("ppr_ranking", "document_ranking", "ranked_ids", "ranking")


class CogRAGAdapter(ExternalBaselineAdapter):
    ranking_fields = ("dual_hypergraph_ranking", "document_ranking", "ranking", "contexts")


class HGRAGAdapter(ExternalBaselineAdapter):
    ranking_fields = ("diffusion_ranking", "document_ranking", "ranking", "contexts")


class HyperRAGAdapter(ExternalBaselineAdapter):
    ranking_fields = ("hypergraph_ranking", "document_ranking", "ranking", "contexts")


ADAPTERS = {
    "graphrag": GraphRAGAdapter(BASELINE_SPECS["graphrag"]),
    "lightrag": LightRAGAdapter(BASELINE_SPECS["lightrag"]),
    "pathrag": PathRAGAdapter(BASELINE_SPECS["pathrag"]),
    "hypergraphrag": HyperGraphRAGAdapter(BASELINE_SPECS["hypergraphrag"]),
    "hipporag2": HippoRAG2Adapter(BASELINE_SPECS["hipporag2"]),
    "cograg": CogRAGAdapter(BASELINE_SPECS["cograg"]),
    "hgrag": HGRAGAdapter(BASELINE_SPECS["hgrag"]),
    "hyperrag": HyperRAGAdapter(BASELINE_SPECS["hyperrag"]),
}


def load_external_results(
    baseline: str,
    path: str | Path,
    suite: BenchmarkSuite,
    *,
    reject_unknown_examples: bool = True,
    expected_generation_trace: dict | None = None,
    expected_shared_model_trace: dict | None = None,
) -> list[ExternalBaselineResult]:
    try:
        adapter = ADAPTERS[baseline.casefold()]
    except KeyError as exc:
        raise ValueError(f"unsupported external baseline: {baseline}") from exc
    examples = {item.example_id: item for item in suite.examples}
    rows_by_example: dict[str, dict] = {}
    unknown_examples = []
    rows, submitted_manifest = _read_result_payload(path)
    for row in rows:
        example_id = str(row.get("example_id", row.get("question_id", row.get("id", ""))))
        if example_id not in examples:
            unknown_examples.append(example_id)
            continue
        if example_id in rows_by_example:
            raise ValueError(f"duplicate external result for example_id={example_id}")
        rows_by_example[example_id] = row
    if reject_unknown_examples and unknown_examples:
        raise ValueError(
            f"external results contain unknown example IDs: {sorted(unknown_examples)[:10]}"
        )
    results = [
        (
            adapter.normalize(rows_by_example[example.example_id], example)
            if example.example_id in rows_by_example
            else adapter.missing(example)
        )
        for example in suite.examples
    ]
    expected = expected_generation_trace or expected_shared_generation_trace()
    if submitted_manifest is not None:
        _validate_submitted_example_manifest(submitted_manifest, examples)
        protocol_matched = _generation_contract_matches(
            submitted_manifest.get("generation_protocol"),
            submitted_manifest.get("generation_trace", {}),
            adapter.spec.capability,
            expected,
        )
    else:
        received = [result for result in results if result.status != "missing"]
        protocol_matched = bool(received) and all(
            _generation_contract_matches(
                result.generation_protocol,
                result.generation_trace,
                adapter.spec.capability,
                expected,
            )
            for result in received
        )
    submitted_model_trace = (
        submitted_manifest.get("shared_model_trace", {}) if submitted_manifest is not None else None
    )
    return [
        replace(
            result,
            generation_protocol_matched=protocol_matched,
            generation_protocol=(
                submitted_manifest.get("generation_protocol")
                if submitted_manifest is not None
                else result.generation_protocol
            ),
            generation_trace=(
                submitted_manifest.get("generation_trace", {})
                if submitted_manifest is not None
                else result.generation_trace
            ),
            shared_model_trace=(
                submitted_model_trace
                if submitted_model_trace is not None
                else result.shared_model_trace
            ),
            shared_model_protocol_matched=(
                expected_shared_model_trace is None
                or _model_contract_matches(
                    (
                        submitted_model_trace
                        if submitted_model_trace is not None
                        else result.shared_model_trace
                    ),
                    expected_shared_model_trace,
                )
            ),
        )
        for result in results
    ]


def expected_shared_generation_trace(context_budget: int = 12) -> dict:
    manifest = EvidenceGenerator().manifest()
    return {
        key: manifest[key]
        for key in (
            "model_id",
            "prompt_sha256",
            "temperature",
            "max_tokens",
            "retry_policy",
            "retry_initial_seconds",
            "retry_max_seconds",
            "max_attempts",
        )
    } | {"context_budget": context_budget}


def _generation_contract_matches(
    generation_protocol,
    generation_trace,
    capability: SystemCapability,
    expected_trace: dict,
) -> bool:
    if not capability.supports_answer_generation:
        return False
    if generation_protocol != capability.generation_protocol:
        return False
    if not isinstance(generation_trace, dict):
        return False
    return all(generation_trace.get(key) == value for key, value in expected_trace.items())


def _model_contract_matches(submitted: dict, expected: dict) -> bool:
    if not isinstance(submitted, dict):
        return False
    return submitted == expected


def _validate_submitted_example_manifest(
    manifest: dict,
    examples: dict[str, BenchmarkExample],
) -> None:
    expected_count = len(examples)
    expected_hash = hashlib.sha256("\n".join(sorted(examples)).encode()).hexdigest()
    if manifest.get("expected_examples") != expected_count:
        raise ValueError("external manifest expected_examples does not match the evaluation suite")
    if manifest.get("expected_example_ids_sha256") != expected_hash:
        raise ValueError("external manifest example ID hash does not match the evaluation suite")


def _read_result_payload(path: str | Path) -> tuple[list[dict], dict | None]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()], None
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload, None
    for key in ("records", "results", "data"):
        if isinstance(payload.get(key), list):
            manifest = payload.get("manifest")
            if manifest is not None and not isinstance(manifest, dict):
                raise ValueError("external result manifest must be an object")
            return payload[key], manifest
    raise ValueError(f"cannot find result records in {source}")


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _map_document_id(value, example: BenchmarkExample) -> str | None:
    if isinstance(value, dict):
        for key in ("passage_id", "document_id", "id", "title", "index"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, int) and 0 <= value < len(example.passages):
        return example.passages[value].passage_id
    text = str(value)
    for passage in example.passages:
        if text == passage.passage_id or text == passage.title:
            return passage.passage_id
    return None


def _map_candidate(value, example: BenchmarkExample, source_map) -> list[str]:
    references = _candidate_references(value, source_map)
    return list(
        dict.fromkeys(
            mapped
            for reference in references
            if (mapped := _map_document_id(reference, example)) is not None
        )
    )


def _candidate_references(value, source_map) -> list:
    if isinstance(value, (str, int)):
        mapped = source_map.get(str(value)) if isinstance(source_map, dict) else None
        return _as_list(mapped) if mapped is not None else [value]
    if not isinstance(value, dict):
        return [value]

    candidate_id = value.get("id", value.get("community_id", value.get("entity_id")))
    if candidate_id is not None and isinstance(source_map, dict):
        mapped = source_map.get(str(candidate_id))
        if mapped is not None:
            return _as_list(mapped)

    direct = []
    for key in ("passage_id", "document_id", "id", "title", "index"):
        if key in value:
            direct.append({key: value[key]})
    if direct:
        return direct

    nested = []
    for key in (
        "source_passage_ids",
        "source_document_ids",
        "source_titles",
        "source_indices",
        "source_ids",
        "sources",
        "passages",
        "documents",
        "chunks",
        "text_units",
    ):
        if key in value:
            nested.extend(_as_list(value[key]))
    return [reference for item in nested for reference in _candidate_references(item, source_map)]


def _candidate_label(value) -> str:
    if isinstance(value, dict):
        for key in ("id", "community_id", "entity_id", "title", "passage_id"):
            if key in value:
                return str(value[key])
    return str(value)


def _number(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None

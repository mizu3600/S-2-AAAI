from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qmshe.benchmarks.schemas import BenchmarkExample, BenchmarkSuite


@dataclass(frozen=True)
class ExternalBaselineSpec:
    key: str
    display_name: str
    repository: str
    ranking_contract: str


@dataclass(frozen=True)
class ExternalBaselineResult:
    system: str
    example_id: str
    status: str
    document_ranking: list[str]
    document_ranking_declared: bool
    fact_ranking: list[str]
    fact_ranking_declared: bool
    answer: str
    citations: list[str]
    citation_level: str
    citation_mapping_complete: bool
    indexing_seconds: float | None
    retrieval_seconds: float | None
    total_seconds: float | None
    error: str | None
    ranking_origin: str
    generation_protocol: str | None
    mapping_coverage: float
    unmapped_ranking_ids: list[str]
    raw: dict


BASELINE_SPECS = {
    "graphrag": ExternalBaselineSpec(
        "graphrag", "Microsoft GraphRAG", "https://github.com/microsoft/graphrag",
        "ranked community or chunk identifiers",
    ),
    "lightrag": ExternalBaselineSpec(
        "lightrag", "LightRAG", "https://github.com/HKUDS/LightRAG",
        "ranked chunk or entity identifiers",
    ),
    "pathrag": ExternalBaselineSpec(
        "pathrag", "PathRAG", "https://github.com/BUPT-GAMMA/PathRAG",
        "ranked path-context identifiers",
    ),
    "hypergraphrag": ExternalBaselineSpec(
        "hypergraphrag", "HyperGraphRAG", "https://github.com/LHRLAB/HyperGraphRAG",
        "ranked hyperedge or passage identifiers",
    ),
    "hipporag2": ExternalBaselineSpec(
        "hipporag2", "HippoRAG 2", "https://github.com/OSU-NLP-Group/HippoRAG",
        "ranked passage identifiers from native PPR",
    ),
    "cograg": ExternalBaselineSpec(
        "cograg", "Cog-RAG", "https://github.com/haoohu/Cog-RAG",
        "ranked passage identifiers from dual-hypergraph retrieval",
    ),
    "hgrag": ExternalBaselineSpec(
        "hgrag", "HGRAG", "https://github.com/MF-AIR/HGRAG",
        "ranked passage identifiers from hypergraph diffusion",
    ),
    "hyperrag": ExternalBaselineSpec(
        "hyperrag", "Hyper-RAG", "https://github.com/iMoonLab/Hyper-RAG",
        "ranked hypergraph passage identifiers",
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
        citations, citation_level, citation_mapping_complete = self._normalize_citations(
            row, example
        )
        status = str(row.get("status", "success"))
        error = row.get("error")
        if error and status == "success":
            status = "failed"
        elif not document_ranking_declared:
            status = "unscorable"
        elif unmapped_ranking_ids:
            status = "unscorable"
        return ExternalBaselineResult(
            system=f"external:{self.spec.key}",
            example_id=example.example_id,
            status=status,
            document_ranking=document_ranking,
            document_ranking_declared=document_ranking_declared,
            fact_ranking=[str(item) for item in fact_ranking],
            fact_ranking_declared="fact_ranking" in row,
            answer=str(row.get("answer", row.get("response", "")) or ""),
            citations=citations,
            citation_level=citation_level,
            citation_mapping_complete=citation_mapping_complete,
            indexing_seconds=_number(
                row.get("indexing_seconds", row.get("index_seconds"))
            ),
            retrieval_seconds=_number(row.get("retrieval_seconds", row.get("latency_seconds"))),
            total_seconds=_number(row.get("total_seconds", row.get("elapsed_seconds"))),
            error=error,
            ranking_origin=row.get("ranking_origin", f"{self.spec.key}_native_result"),
            generation_protocol=row.get("generation_protocol"),
            mapping_coverage=mapping_coverage,
            unmapped_ranking_ids=unmapped_ranking_ids,
            raw=row,
        )

    def missing(self, example: BenchmarkExample) -> ExternalBaselineResult:
        return ExternalBaselineResult(
            system=f"external:{self.spec.key}",
            example_id=example.example_id,
            status="missing",
            document_ranking=[],
            document_ranking_declared=False,
            fact_ranking=[],
            fact_ranking_declared=False,
            answer="",
            citations=[],
            citation_level="none",
            citation_mapping_complete=False,
            indexing_seconds=None,
            retrieval_seconds=None,
            total_seconds=None,
            error="missing result row",
            ranking_origin=f"{self.spec.key}_native_result",
            generation_protocol=None,
            mapping_coverage=0.0,
            unmapped_ranking_ids=[],
            raw={},
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
                unmapped.append(_candidate_label(candidate))
        coverage = (
            mapped_candidates / len(raw_ranking)
            if raw_ranking
            else float(ranking_declared)
        )
        return list(dict.fromkeys(mapped)), ranking_declared, coverage, unmapped

    def _normalize_citations(
        self, row: dict, example: BenchmarkExample
    ) -> tuple[list[str], str, bool]:
        citations = _as_list(row.get("citations", row.get("citation_ids", [])))
        if not citations:
            return [], "none", False
        source_map = row.get("source_id_map", row.get("id_map", {}))
        citation_mappings = [
            _map_candidate(citation, example, source_map) for citation in citations
        ]
        document_ids = [item for mapped in citation_mappings for item in mapped]
        if document_ids:
            return (
                list(dict.fromkeys(document_ids)),
                "document",
                all(citation_mappings),
            )
        fact_ids = [str(item) for item in citations]
        return fact_ids, "fact", all(item.startswith("fact_") for item in fact_ids)


class GraphRAGAdapter(ExternalBaselineAdapter):
    ranking_fields = ("community_ranking", "chunk_ranking", "document_ranking", "ranking", "contexts")


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
) -> list[ExternalBaselineResult]:
    try:
        adapter = ADAPTERS[baseline.casefold()]
    except KeyError as exc:
        raise ValueError(f"unsupported external baseline: {baseline}") from exc
    examples = {item.example_id: item for item in suite.examples}
    rows_by_example: dict[str, dict] = {}
    unknown_examples = []
    for row in _read_rows(path):
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
    return [
        (
            adapter.normalize(rows_by_example[example.example_id], example)
            if example.example_id in rows_by_example
            else adapter.missing(example)
        )
        for example in suite.examples
    ]


def _read_rows(path: str | Path) -> list[dict]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    for key in ("records", "results", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
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
    return [
        reference
        for item in nested
        for reference in _candidate_references(item, source_map)
    ]


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

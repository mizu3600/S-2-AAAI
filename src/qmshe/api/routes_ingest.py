from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from qmshe.api.dependencies import (
    ensure_runtime_mode_enabled,
    set_graph_pipeline,
    set_pipeline,
)
from qmshe.extraction.canonicalizer import canonicalize_entities
from qmshe.extraction.entity_extractor import extract_entities_rule_based
from qmshe.extraction.fact_extractor import extract_facts_rule_based, extract_facts_with_llm
from qmshe.ingest.chunker import chunk_document
from qmshe.ingest.schemas import Corpus
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.pipeline import QMSHEPipeline, load_corpus, save_corpus
from qmshe.providers import DeepSeekClient, ProviderError

router = APIRouter(prefix="/v1", tags=["index"])


class IngestRequest(BaseModel):
    source_uri: str
    domain: str = "PSC"
    rebuild_embeddings: bool = True


class BuildRequest(BaseModel):
    corpus_version: str = "psc-v1"
    corpus_path: str = "data/processed/corpus.json"
    build_evidence_graph: bool = True
    build_semantic_graph: bool = True
    build_spectral_embeddings: bool = True
    mode: Literal["hypergraph", "graph", "both"] = "graph"
    graph_profile: GraphProfile = GraphProfile.REIFIED_FACT
    graph_index_strategy: Literal["single", "multi", "hybrid"] = "hybrid"


class IncrementalRequest(BaseModel):
    corpus_path: str
    mode: Literal["hypergraph", "graph"] = "graph"
    graph_profile: GraphProfile = GraphProfile.REIFIED_FACT


@router.post("/documents/ingest")
def ingest(request: IngestRequest) -> dict:
    from qmshe.ingest.pdf_parser import parse_document

    path = Path(request.source_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="source file not found")
    parsed = parse_document(path, request.domain)
    chunks = chunk_document(parsed)
    entities = canonicalize_entities(extract_entities_rule_based(chunks))
    try:
        facts = extract_facts_with_llm(chunks, entities, DeepSeekClient())
    except ProviderError:
        facts = extract_facts_rule_based(chunks, entities)
    corpus = Corpus(documents=[parsed.document], chunks=chunks, entities=entities, evidence_hyperedges=facts)
    output = Path("data/processed/corpus.json")
    save_corpus(corpus, output)
    return {"document_id": parsed.document.document_id, "chunks": len(chunks), "entities": len(entities),
            "facts": len(facts), "corpus_path": str(output)}


@router.post("/index/build")
def build_index(request: BuildRequest) -> dict:
    _ = (request.build_evidence_graph, request.build_semantic_graph, request.build_spectral_embeddings)
    corpus = load_corpus(request.corpus_path)
    built = []
    if request.mode in {"hypergraph", "both"}:
        ensure_runtime_mode_enabled("hypergraph")
        set_pipeline(QMSHEPipeline(corpus))
        built.append("hypergraph")
    if request.mode in {"graph", "both"}:
        ensure_runtime_mode_enabled("graph", request.graph_profile)
        set_graph_pipeline(QMSGEGraphPipeline(
            corpus, profile=request.graph_profile, index_strategy=request.graph_index_strategy
        ))
        built.append(f"graph:{request.graph_profile.value}")
    return {"status": "ready", "corpus_version": request.corpus_version,
            "modes": built, "entities": len(corpus.entities),
            "hyperedges": len(corpus.evidence_hyperedges)}


@router.post("/index/incremental")
def incremental_index(request: IncrementalRequest) -> dict:
    from qmshe.api.dependencies import get_graph_pipeline, get_pipeline

    updated = load_corpus(request.corpus_path)
    if request.mode == "graph":
        return get_graph_pipeline(request.graph_profile).incremental_update(updated)
    pipeline = get_pipeline()
    return pipeline.incremental_update(updated)

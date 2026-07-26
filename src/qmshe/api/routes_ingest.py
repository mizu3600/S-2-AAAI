from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from qmshe.api.dependencies import set_pipeline
from qmshe.ingest.service import ingest_document
from qmshe.pipeline import QMSHERAGPipeline, load_corpus, save_corpus


router = APIRouter(prefix="/v1", tags=["pipeline"])


class IngestRequest(BaseModel):
    source_uri: str
    domain: str = "PSC"
    output_path: str = "data/processed/corpus.json"


class BuildRequest(BaseModel):
    corpus_path: str = "data/processed/corpus.json"


@router.post("/documents/ingest")
def ingest(request: IngestRequest) -> dict:
    path = Path(request.source_uri)
    if not path.exists():
        raise HTTPException(status_code=404, detail="source file not found")
    corpus = ingest_document(path, request.domain)
    save_corpus(corpus, request.output_path)
    return {
        "document_id": corpus.documents[0].document_id,
        "chunks": len(corpus.chunks),
        "entities": len(corpus.entities),
        "facts": len(corpus.evidence_hyperedges),
        "corpus_path": request.output_path,
    }


@router.post("/index/build")
def build_index(request: BuildRequest) -> dict:
    corpus = load_corpus(request.corpus_path)
    pipeline = QMSHERAGPipeline(corpus)
    set_pipeline(pipeline)
    return {
        "status": "ready",
        "pipeline": "reified_fact_hybrid",
        "nodes": len(pipeline.node_ids),
        "facts": len(corpus.evidence_hyperedges),
    }

import json
from pathlib import Path

import typer

from qmshe.ingest.service import ingest_document
from qmshe.pipeline import QMSHERAGPipeline, load_corpus, save_corpus


app = typer.Typer(help="Reified-Fact spectral graph RAG")


@app.command()
def ingest(
    source: Path,
    output: Path = Path("data/processed/corpus.json"),
    domain: str = "PSC",
) -> None:
    corpus = ingest_document(source, domain)
    save_corpus(corpus, output)
    typer.echo(
        f"saved {len(corpus.chunks)} chunks, {len(corpus.entities)} entities and "
        f"{len(corpus.evidence_hyperedges)} facts to {output}"
    )


@app.command()
def build(
    corpus_path: Path = typer.Argument(Path("data/processed/corpus.json"))
) -> None:
    pipeline = QMSHERAGPipeline(load_corpus(corpus_path))
    typer.echo(
        f"built reified-fact hybrid index with {len(pipeline.node_ids)} nodes"
    )


@app.command()
def query(
    question: str,
    corpus_path: Path = Path("data/processed/corpus.json"),
    top_k: int = 12,
    debug: bool = False,
) -> None:
    result = QMSHERAGPipeline(load_corpus(corpus_path)).query(
        question, top_k=top_k, return_debug=debug
    )
    typer.echo(json.dumps(result.__dict__, ensure_ascii=False, indent=2))

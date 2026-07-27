from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from s2rag import __version__
from s2rag.api.routes_ingest import router as ingest_router
from s2rag.api.routes_query import router as query_router

app = FastAPI(title="S²-RAG", version=__version__)
app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "pipeline": "reified_fact_hybrid",
    }


@app.exception_handler(RuntimeError)
def runtime_error_handler(_: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})

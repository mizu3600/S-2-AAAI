from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from qmshe import __version__
from qmshe.api.routes_evaluate import router as evaluate_router
from qmshe.api.routes_ingest import router as ingest_router
from qmshe.api.routes_query import router as query_router

app = FastAPI(title="QMSHE-RAG", version=__version__)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(evaluate_router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "active_mode": "graph",
        "active_profile": "reified_fact",
    }


@app.exception_handler(RuntimeError)
def runtime_error_handler(_: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})

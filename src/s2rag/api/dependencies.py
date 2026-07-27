from s2rag.pipeline import S2RAGPipeline


_pipeline: S2RAGPipeline | None = None


def set_pipeline(pipeline: S2RAGPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> S2RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("index is not built; call POST /v1/index/build first")
    return _pipeline

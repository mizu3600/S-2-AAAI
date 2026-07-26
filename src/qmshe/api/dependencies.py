from qmshe.pipeline import QMSHERAGPipeline


_pipeline: QMSHERAGPipeline | None = None


def set_pipeline(pipeline: QMSHERAGPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> QMSHERAGPipeline:
    if _pipeline is None:
        raise RuntimeError("index is not built; call POST /v1/index/build first")
    return _pipeline

from qmshe.pipeline import QMSHEPipeline
from qmshe.graph.ordinary import GraphProfile
from qmshe.graph_pipeline import QMSGEGraphPipeline
from qmshe.settings import get_settings

_pipeline: QMSHEPipeline | None = None
_graph_pipelines: dict[GraphProfile, QMSGEGraphPipeline] = {}


def set_pipeline(pipeline: QMSHEPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> QMSHEPipeline:
    ensure_runtime_mode_enabled("hypergraph")
    if _pipeline is None:
        raise RuntimeError("index is not built; call POST /v1/index/build first")
    return _pipeline


def set_graph_pipeline(pipeline: QMSGEGraphPipeline) -> None:
    _graph_pipelines[pipeline.profile] = pipeline


def get_graph_pipeline(
    profile: GraphProfile | str = GraphProfile.REIFIED_FACT,
) -> QMSGEGraphPipeline:
    selected = GraphProfile(profile)
    ensure_runtime_mode_enabled("graph", selected)
    if selected not in _graph_pipelines:
        raise RuntimeError(
            f"ordinary graph index ({selected.value}) is not built; "
            "call POST /v1/index/build with mode=graph first"
        )
    return _graph_pipelines[selected]


def get_active_pipeline() -> QMSGEGraphPipeline:
    """Return the sole production-default Reified-Fact pipeline."""
    return get_graph_pipeline(GraphProfile.REIFIED_FACT)


def ensure_runtime_mode_enabled(
    mode: str, profile: GraphProfile | str = GraphProfile.REIFIED_FACT
) -> None:
    settings = get_settings()
    if mode == "hypergraph":
        if not settings.qmshe_enable_hypergraph:
            raise RuntimeError(
                "hypergraph runtime is disabled; set QMSHE_ENABLE_HYPERGRAPH=true to opt in"
            )
        return
    if mode != "graph":
        raise RuntimeError("mode must be hypergraph or graph")
    selected = GraphProfile(profile)
    if selected is GraphProfile.ENTITY_RELATION:
        enabled = settings.qmshe_enable_entity_relation
        variable = "QMSHE_ENABLE_ENTITY_RELATION"
    else:
        enabled = settings.qmshe_enable_reified_fact
        variable = "QMSHE_ENABLE_REIFIED_FACT"
    if not enabled:
        raise RuntimeError(
            f"ordinary graph profile ({selected.value}) is disabled; set {variable}=true to opt in"
        )

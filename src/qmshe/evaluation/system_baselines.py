from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalSystemBaseline:
    name: str
    repository: str
    adapter_contract: str
    isolation: str = "separate environment"


SYSTEM_BASELINES = {
    "vanilla_dense_rag": ExternalSystemBaseline(
        "Vanilla Dense RAG", "internal", "ranked chunk IDs + answer + citations"
    ),
    "graphrag": ExternalSystemBaseline(
        "Microsoft GraphRAG",
        "https://github.com/microsoft/graphrag",
        "ranked community/chunk IDs",
    ),
    "lightrag": ExternalSystemBaseline(
        "LightRAG",
        "https://github.com/HKUDS/LightRAG",
        "ranked chunk/entity IDs",
    ),
    "pathrag": ExternalSystemBaseline(
        "PathRAG",
        "https://github.com/BUPT-GAMMA/PathRAG",
        "ranked path context IDs",
    ),
    "hypergraphrag": ExternalSystemBaseline(
        "HyperGraphRAG",
        "https://github.com/LHRLAB/HyperGraphRAG",
        "ranked hyperedge IDs",
    ),
    "hipporag2": ExternalSystemBaseline(
        "HippoRAG 2",
        "https://github.com/OSU-NLP-Group/HippoRAG",
        "ranked passage IDs via PPR",
    ),
    "cograg": ExternalSystemBaseline(
        "Cog-RAG",
        "https://github.com/haoohu/Cog-RAG",
        "ranked dual-hypergraph theme+entity IDs",
    ),
    "hgrag": ExternalSystemBaseline(
        "HGRAG",
        "https://github.com/MF-AIR/HGRAG",
        "ranked hypergraph-diffusion passage IDs",
    ),
    "hyperrag": ExternalSystemBaseline(
        "Hyper-RAG",
        "https://github.com/iMoonLab/Hyper-RAG",
        "ranked hypergraph passage IDs",
    ),
}


def validate_external_result(result: dict) -> None:
    required = {"question_id", "ranked_ids", "latency_ms"}
    if not required <= result:
        raise ValueError(f"external baseline result missing {sorted(required - set(result))}")
    if not isinstance(result["ranked_ids"], list):
        raise TypeError("ranked_ids must be a list")

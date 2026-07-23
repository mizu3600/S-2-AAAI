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
    "lightrag": ExternalSystemBaseline(
        "LightRAG", "https://github.com/HKUDS/LightRAG", "ranked chunk/entity IDs"
    ),
    "hipporag2": ExternalSystemBaseline(
        "HippoRAG 2", "https://github.com/OSU-NLP-Group/HippoRAG", "ranked passage IDs"
    ),
    "ehrag": ExternalSystemBaseline(
        "EHRAG", "external implementation", "ranked evidence IDs"
    ),
    "hgrag": ExternalSystemBaseline(
        "HGRAG/HyperRAG", "external implementation", "ranked hyperedge IDs"
    ),
}


def validate_external_result(result: dict) -> None:
    required = {"question_id", "ranked_ids", "latency_ms"}
    if not required <= result:
        raise ValueError(f"external baseline result missing {sorted(required - set(result))}")
    if not isinstance(result["ranked_ids"], list):
        raise TypeError("ranked_ids must be a list")


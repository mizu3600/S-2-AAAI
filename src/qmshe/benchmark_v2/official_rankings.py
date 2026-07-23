from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Callable


@dataclass
class CapturedInternalRanking:
    framework: str
    items: list[dict]
    origin: str

    def source_ids(self) -> list[str]:
        output = []
        for item in self.items:
            value = item.get("source_id", item.get("id"))
            if value is not None:
                output.append(str(value))
        return list(dict.fromkeys(output))


@asynccontextmanager
async def capture_ranked_text_units(
    framework: str,
    operate_module: ModuleType,
    function_name: str,
):
    """Capture PathRAG/HyperGraphRAG's ranked text-unit objects in memory.

    This wraps the official internal selection function for one query and does
    not parse the rendered ``-----Sources-----`` context. The original function
    is restored even when the official query raises.
    """
    original: Callable = getattr(operate_module, function_name)
    capture = CapturedInternalRanking(
        framework=framework,
        items=[],
        origin=f"official_internal:{operate_module.__name__}.{function_name}",
    )

    async def wrapper(*args, **kwargs):
        result = await original(*args, **kwargs)
        capture.items = [dict(item) for item in result]
        return result

    setattr(operate_module, function_name, wrapper)
    try:
        yield capture
    finally:
        setattr(operate_module, function_name, original)


def graphrag_sources_ranking(context_data: dict) -> CapturedInternalRanking:
    """Read GraphRAG's structured Sources table, never its generated prose."""
    sources = context_data.get("sources", [])
    if hasattr(sources, "to_dict"):
        sources = sources.to_dict(orient="records")
    return CapturedInternalRanking("graphrag", list(sources), "official_internal:context_data.sources")


def lightrag_chunks_ranking(query_data: dict) -> CapturedInternalRanking:
    """Read LightRAG's structured chunk list returned by ``aquery_data``."""
    chunks = query_data.get("data", query_data).get("chunks", [])
    return CapturedInternalRanking("lightrag", list(chunks), "official_internal:aquery_data.chunks")

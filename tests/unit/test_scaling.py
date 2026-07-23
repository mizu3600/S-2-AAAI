import asyncio

import networkx as nx
import numpy as np

from qmshe.ingest.schemas import Argument, Chunk, Entity, EvidenceHyperedge
from qmshe.pipeline import QMSHEPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.scaling.cache import VersionedQueryCache
from qmshe.scaling.incremental_index import plan_incremental_update
from qmshe.scaling.load_test import run_load_test
from qmshe.scaling.partitioning import build_coarse_graph, partition_graph
from qmshe.scaling.versioning import ArtifactVersion, VersionManifest
from qmshe.synthetic import make_synthetic_corpus


class LocalEncoder:
    def encode(self, texts):
        return DeterministicEmbedder(64).embed(texts)


def test_versions_cache_and_partition(tmp_path):
    version = ArtifactVersion.create()
    manifest = VersionManifest(tmp_path / "version.json")
    manifest.save(version)
    manifest.assert_compatible(version)
    cache = VersionedQueryCache(max_items=1)
    key = cache.key("question", version)
    cache.put(key, {"answer": 1})
    assert cache.get(key) == {"answer": 1}
    graph = nx.path_graph(["a", "b", "c", "d"])
    partitions = partition_graph(graph, max_partition_size=2)
    coarse, features = build_coarse_graph(graph, partitions, {node: np.ones(3) for node in graph})
    assert all(len(part.node_ids) <= 2 for part in partitions)
    assert coarse.number_of_nodes() == len(partitions)
    assert features


def test_incremental_approximate_index_and_load_test():
    old = make_synthetic_corpus()
    updated = old.model_copy(deep=True)
    updated.entities.append(Entity(
        entity_id="ent_new", canonical_name="new additive", entity_type="additive", description="New additive"
    ))
    updated.chunks.append(Chunk(
        chunk_id="chunk_new", document_id="doc_synthetic", section="Update", text="A new additive affects Voc.",
        start_char=351, end_char=379,
    ))
    updated.evidence_hyperedges.append(EvidenceHyperedge(
        hyperedge_id="fact_new", predicate="improves_device_performance",
        arguments=[Argument(role="material", entity_id="ent_new"), Argument(role="result", entity_id="ent_voc")],
        evidence_chunk_ids=["chunk_new"], evidence_sentence="A new additive affects Voc.", confidence=0.8,
    ))
    plan = plan_incremental_update(old, updated)
    assert plan.spectral_status == "approximate"
    pipeline = QMSHEPipeline(old, text_encoder=LocalEncoder())
    pipeline.generator.client = None
    result = pipeline.incremental_update(updated)
    assert result["indexed_objects"] == 2
    assert "fact_new" in pipeline.object_ids
    load = asyncio.run(run_load_test(pipeline, ["What affects Voc?"], requests=4, concurrency=2))
    assert load.success_rate == 1.0


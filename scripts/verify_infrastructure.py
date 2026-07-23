"""Smoke-test Qdrant, Neo4j and PostgreSQL with the synthetic corpus."""

from qmshe.pipeline import QMSHEPipeline
from qmshe.providers import DeterministicEmbedder
from qmshe.storage.neo4j_store import Neo4jEvidenceStore
from qmshe.storage.postgres_store import PostgresMetadataStore
from qmshe.storage.qdrant_store import QdrantMultiVectorStore
from qmshe.synthetic import make_synthetic_corpus


class LocalEncoder:
    def encode(self, texts):
        return DeterministicEmbedder(64).embed(texts)


def main() -> None:
    corpus = make_synthetic_corpus()
    pipeline = QMSHEPipeline(corpus, text_encoder=LocalEncoder())
    vectors = {name: value.detach().numpy() for name, value in pipeline.node_bands.items()}
    qdrant = QdrantMultiVectorStore(collection="qmshe_smoke")
    qdrant.create_collection({name: matrix.shape[1] for name, matrix in vectors.items()}, recreate=True)
    payloads = [
        {
            "object_type": "entity" if object_id.startswith("ent_") else "hyperedge",
            "graph_version": corpus.graph_version,
            "encoder_version": "deterministic-smoke-v1",
            "spectral_version": "spec-smoke-v1",
            "index_version": "index-smoke-v1",
            "evidence_only": object_id.startswith("fact_"),
        }
        for object_id in pipeline.object_ids
    ]
    qdrant.upsert(pipeline.object_ids, vectors, payloads)
    qdrant_hits = qdrant.search("full", vectors["full"][0], top_k=1)

    neo4j = Neo4jEvidenceStore()
    try:
        neo4j.write_corpus(corpus)
    finally:
        neo4j.close()

    postgres = PostgresMetadataStore()
    postgres.initialize()
    postgres.save_corpus_version(corpus, "deterministic-smoke-v1", "spec-smoke-v1", "index-smoke-v1")
    print({"qdrant_hits": len(qdrant_hits), "neo4j": "written", "postgres": "written"})


if __name__ == "__main__":
    main()


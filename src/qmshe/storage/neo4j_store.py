from qmshe.ingest.schemas import Corpus
from qmshe.graph.ordinary import OrdinaryGraphArtifacts
from qmshe.settings import Settings, get_settings


class Neo4jEvidenceStore:
    def __init__(self, settings: Settings | None = None, auth: tuple[str, str] = ("neo4j", "qmshe-local-only")):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("install the 'infra' extra to use Neo4j") from exc
        self.settings = settings or get_settings()
        self.driver = GraphDatabase.driver(self.settings.neo4j_uri, auth=auth)

    def close(self) -> None:
        self.driver.close()

    def write_corpus(self, corpus: Corpus) -> None:
        with self.driver.session() as session:
            for entity in corpus.entities:
                session.run(
                    "MERGE (e:Entity {id: $id}) SET e.name=$name, e.type=$type, e.description=$description",
                    id=entity.entity_id, name=entity.canonical_name, type=entity.entity_type,
                    description=entity.description,
                )
            for fact in corpus.evidence_hyperedges:
                session.run(
                    "MERGE (f:EvidenceFact {id: $id}) SET f.predicate=$predicate, "
                    "f.confidence=$confidence, f.chunk_ids=$chunk_ids",
                    id=fact.hyperedge_id, predicate=fact.predicate, confidence=fact.confidence,
                    chunk_ids=fact.evidence_chunk_ids,
                )
                for argument in fact.arguments:
                    session.run(
                        "MATCH (e:Entity {id:$entity_id}), (f:EvidenceFact {id:$fact_id}) "
                        "MERGE (e)-[r:PARTICIPATES_IN {role:$role}]->(f)",
                        entity_id=argument.entity_id, fact_id=fact.hyperedge_id, role=argument.role,
                    )
            for edge in corpus.semantic_hyperedges:
                session.run(
                    "MERGE (s:SemanticEdge {id:$id}) SET s.status='retrieval_only', "
                    "s.topic=$topic, s.confidence=$confidence",
                    id=edge.semantic_edge_id, topic=edge.topic, confidence=edge.confidence,
                )

    def write_ordinary_graph(self, artifacts: OrdinaryGraphArtifacts) -> None:
        """Persist the graph branch with labels isolated from the hypergraph projection."""
        with self.driver.session() as session:
            for node_id, attributes in artifacts.graph.nodes(data=True):
                session.run(
                    "MERGE (n:QMSGEGraphNode {id:$id, profile:$profile}) "
                    "SET n.kind=$kind, n.text=$text",
                    id=node_id, profile=artifacts.profile.value,
                    kind=attributes.get("kind", "entity"), text=attributes.get("text", ""),
                )
            for left, right, attributes in artifacts.graph.edges(data=True):
                session.run(
                    "MATCH (a:QMSGEGraphNode {id:$left, profile:$profile}), "
                    "(b:QMSGEGraphNode {id:$right, profile:$profile}) "
                    "MERGE (a)-[r:QMSGE_LINK {edge_key:$edge_key}]->(b) "
                    "SET r.weight=$weight, r.role=$role, r.fact_ids=$fact_ids",
                    left=left, right=right, profile=artifacts.profile.value,
                    edge_key=f"{left}|{right}", weight=float(attributes.get("weight", 1.0)),
                    role=attributes.get("role"),
                    fact_ids=attributes.get("fact_ids", [attributes.get("fact_id")]),
                )

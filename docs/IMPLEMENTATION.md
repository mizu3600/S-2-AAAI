# QMSHE-RAG implementation notes

## Implemented scope

The repository implements the engineering path across design phases 0–5. Publication-scale results
still require full datasets, valid API credentials, domain papers and expert gold annotations.

1. PDF, TXT and Markdown parsing with document/section/page/offset provenance.
2. Boundary-aware token chunking and conservative entity canonicalization.
3. DeepSeek JSON fact extraction plus deterministic keyless test extraction.
4. Separate evidence hypergraph and retrieval-only semantic graph.
5. Sparse incidence matrices, role matrices, Zhou and joint bipartite Laplacians.
6. Learnable low/mid/high Chebyshev filters with raw semantic residual.
7. Query seeds, query-conditioned four-way gate, entity/hyperedge joint embedding.
8. Dense, QMSHE and BM25 retrieval, RRF and SiliconFlow reranking. The legacy
   connectivity-based graph reranker remains available as an explicit opt-in,
   but is disabled by default because controlled attribution found no benefit
   for Entity-Relation and significant regressions for Reified-Fact and Hypergraph.
9. Evidence verification, source-preserving context and citation-constrained generation.
10. Training losses, dual hard negatives, evaluation metrics and baseline adapters.
11. FastAPI endpoints, Docker Compose services and incremental-update policy.
12. Qdrant named multi-vector, Neo4j evidence-path and PostgreSQL version metadata adapters.
13. Public benchmark adapters, Stage A training, MLflow/DVC experiment tracking and ablation registry.
14. PSC ontology, measurement normalization, quality audit and provenance-validated annotation store.
15. Incremental approximate indexing, version compatibility, partition/coarse graph, cache and load test.

The production API and CLI default to `graph:reified_fact`. Hypergraph and Entity-Relation remain
available to training/evaluation code but are runtime-disabled unless their explicit environment
opt-ins are set.

## Evidence boundary

`EvidenceHyperedge` always names one or more source chunks. `SemanticHyperedge` has the
literal status `retrieval_only`; the verifier rejects it before context assembly. This invariant
is covered by an automated regression test.

## Spectral implementation

The Zhou operator is

`I - Dv^-1/2 H W De^-1 H^T Dv^-1/2`.

All graph matrices remain CSR/COO. The Chebyshev recurrence operates with sparse-dense matrix
multiplication. Tests compare the sparse Zhou implementation to the dense formula and the
Chebyshev polynomial response to direct eigendecomposition with tolerance `1e-5`.

## Baselines

Runnable components are provided for BM25, dense, BM25+dense, Node2Vec random-walk
co-occurrence, Laplacian Eigenmaps, semantic+LapPE, semantic+PPR, GCN, GraphSAGE,
HypergraphConv and QMSHE. External system baselines (LightRAG, HippoRAG 2, EHRAG and
HGRAG/HyperRAG) intentionally remain isolated adapters so they can use the same frozen corpus,
encoder, top-k, reranker, generator and token budget without contaminating the core environment.

## Version and incremental policy

New objects enter through their raw embedding and a neighbor-weighted approximate band embedding,
marked approximate by the caller. A full rebuild is requested at 3% node or 5% hyperedge growth.
Production payloads must carry graph, encoder, spectral and index versions; incompatible versions
must never share a collection.

## Security

API keys are read from `.env`, which is ignored by Git. Do not place keys in YAML, source, test
fixtures, notebooks, Docker images or experiment logs.

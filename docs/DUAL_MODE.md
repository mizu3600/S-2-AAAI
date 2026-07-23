# QMSxE-RAG dual graph modes

The repository contains two independent spectral retrieval branches over one source corpus. The
production runtime now exposes only QMSGE ordinary graph with the `reified_fact` profile by
default. QMSHE Hypergraph and the Entity-Relation profile remain intact for research and historical
reproduction, but require explicit environment-variable opt-in.

## Module boundary

| Concern | Hypergraph branch | Ordinary-graph branch |
|---|---|---|
| Evidence structure | n-ary `EvidenceHyperedge` | binary relations or reified fact nodes |
| Operator | Zhou/joint hypergraph Laplacian | normalized adjacency `D^-1/2(A+I)D^-1/2` |
| Spectral bands | learned Chebyshev filters | `S²X`, `SX-S²X`, `X-SX` |
| Query selection | band gate + role gate | independent band gate |
| Index objects | entities + hyperedges | graph nodes |
| Incremental scope | hyperedge approximation | ordinary-graph two-hop neighborhood |

The branches share parsing, canonical entities, extracted evidence facts, text encoder, reranker,
generator and citations. They do not share topology matrices, spectral encoders, indexes, gates,
checkpoints or caches.

## Ordinary graph profiles

`entity_relation` keeps only entity nodes. Each ordinary edge retains predicate, endpoint roles,
source fact IDs and confidence. Retrieval over entities is mapped back to the original evidence
facts before citation verification.

`reified_fact` represents every evidence fact as a normal graph node. Entity-to-fact edges retain
the argument role, source ID and confidence. This profile preserves n-ary condition binding while
still using an ordinary graph operator and ordinary graph retrieval stack.

The ordinary branch supports `single`, `multi` and `hybrid` index strategies. `single` searches the
concatenated full vector. `multi` searches raw/low/mid/high indexes separately and fuses their ranks
with the query gate weights. `hybrid` (default) combines both candidates before lexical fusion and
graph-constrained reranking.

## API

Build the production-default Reified-Fact profile:

```bash
curl -X POST http://127.0.0.1:8000/v1/index/build \
  -H 'content-type: application/json' \
  -d '{"corpus_path":"data/processed/corpus.json","mode":"graph","graph_profile":"reified_fact","graph_index_strategy":"hybrid"}'
```

Query the ordinary graph explicitly:

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H 'content-type: application/json' \
  -d '{"question":"...","mode":"graph","graph_profile":"reified_fact","return_debug":true}'
```

Omitting `mode` queries `graph:reified_fact`. Requests for Hypergraph or Entity-Relation return 409
while disabled. To reproduce historical experiments without changing source code, set
`QMSHE_ENABLE_HYPERGRAPH=true` or `QMSHE_ENABLE_ENTITY_RELATION=true` before starting the process.
The pipelines may still coexist in the process registry after opt-in.

## Training and fair comparison

```bash
uv run python scripts/train_graph_mode.py \
  --dataset hotpotqa --input-path data/benchmarks/hotpotqa_sample.json \
  --profile reified_fact --limit 5 --epochs 10

uv run python scripts/run_dual_mode_experiment.py \
  --dataset hotpotqa --input-path data/benchmarks/hotpotqa_sample.json --limit 5
```

The comparison runner builds all three candidates from the same benchmark example and fixes the
encoder, corpus, top-k and answer-generation policy. It reports the hypergraph branch, entity
relation graph and reified fact graph separately.

## Storage and versions

Use a separate Qdrant collection such as `qmsge_graph_objects` for the graph branch. Store named
vectors `raw`, `low`, `mid`, `high` and `full`; include `mode`, `profile`, graph version, encoder
version and spectral version in each payload. `Neo4jEvidenceStore.write_ordinary_graph` uses the
isolated `QMSGEGraphNode` label and profile key, preventing collisions with the existing evidence
hypergraph projection.

For small updates, only new/changed nodes and their two-hop neighborhood receive recalculated graph
bands; unchanged existing vectors are retained. Growth of at least 3% of nodes or 5% of edges
requests a full graph rebuild. Every update invalidates the profile-specific query cache.

Hybrid routing is intentionally not enabled. Production has one active route,
`graph:reified_fact`; comparison scripts instantiate the preserved branches directly and therefore
do not depend on runtime enablement flags.

# Runtime retrieval ablation matrix

| System | Variant | Fact R@5 | Fact R@10 | Fact R@20 | Fact R@40 | Fact MRR | Passage R@5 | Passage MRR | Path F1 | Raw | Low | Mid | High | s/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph:entity_relation | dense_only | 0.7516 | 0.8394 | 0.9272 | 0.9352 | 0.8732 | 0.8968 | 0.9470 | 0.7407 | 0.2414 | 0.5472 | 0.0826 | 0.1288 | 0.1619 |
| graph:entity_relation | full | 0.7472 | 0.8588 | 0.9453 | 0.9929 | 0.8846 | 0.9259 | 0.9555 | 0.7540 | 0.2414 | 0.5472 | 0.0826 | 0.1288 | 0.2488 |
| graph:entity_relation | index_hybrid | 0.7472 | 0.8588 | 0.9453 | 0.9929 | 0.8846 | 0.9259 | 0.9555 | 0.7540 | 0.2414 | 0.5472 | 0.0826 | 0.1288 | 0.2439 |
| graph:entity_relation | index_multi | 0.7489 | 0.8579 | 0.9330 | 0.9691 | 0.8774 | 0.9127 | 0.9440 | 0.7434 | 0.2414 | 0.5472 | 0.0826 | 0.1288 | 0.2146 |
| graph:entity_relation | index_single | 0.7428 | 0.8465 | 0.9312 | 0.9753 | 0.8774 | 0.9153 | 0.9481 | 0.7487 | 0.2414 | 0.5472 | 0.0826 | 0.1288 | 0.2284 |
| graph:entity_relation | no_bm25 | 0.7472 | 0.8562 | 0.9427 | 0.9903 | 0.8846 | 0.9233 | 0.9555 | 0.7540 | 0.2414 | 0.5472 | 0.0826 | 0.1288 | 0.2414 |
| graph:entity_relation | no_graph_rerank | 0.7472 | 0.8588 | 0.9453 | 0.9929 | 0.8846 | 0.9259 | 0.9555 | 0.7540 | 0.2414 | 0.5472 | 0.0826 | 0.1288 | 0.2386 |
| graph:reified_fact | dense_only | 0.7516 | 0.8394 | 0.9087 | 0.9167 | 0.8732 | 0.8968 | 0.9474 | 0.7407 | 0.8143 | 0.1354 | 0.0131 | 0.0372 | 0.1533 |
| graph:reified_fact | full | 0.7489 | 0.8619 | 0.9405 | 0.9881 | 0.8846 | 0.9286 | 0.9555 | 0.7540 | 0.8143 | 0.1354 | 0.0131 | 0.0372 | 0.2259 |
| graph:reified_fact | index_hybrid | 0.7489 | 0.8619 | 0.9405 | 0.9881 | 0.8846 | 0.9286 | 0.9555 | 0.7540 | 0.8143 | 0.1354 | 0.0131 | 0.0372 | 0.2285 |
| graph:reified_fact | index_multi | 0.7516 | 0.8632 | 0.9409 | 0.9868 | 0.8824 | 0.9259 | 0.9539 | 0.7540 | 0.8143 | 0.1354 | 0.0131 | 0.0372 | 0.2186 |
| graph:reified_fact | index_single | 0.7516 | 0.8646 | 0.9440 | 0.9802 | 0.8941 | 0.9206 | 0.9568 | 0.7540 | 0.8143 | 0.1354 | 0.0131 | 0.0372 | 0.1992 |
| graph:reified_fact | no_bm25 | 0.7569 | 0.8593 | 0.9422 | 0.9881 | 0.8899 | 0.9286 | 0.9555 | 0.7566 | 0.8143 | 0.1354 | 0.0131 | 0.0372 | 0.2324 |
| graph:reified_fact | no_graph_rerank | 0.7489 | 0.8619 | 0.9405 | 0.9881 | 0.8846 | 0.9286 | 0.9555 | 0.7540 | 0.8143 | 0.1354 | 0.0131 | 0.0372 | 0.2244 |
| hypergraph:evidence_hypergraph | dense_only | 0.6386 | 0.6558 | 0.6770 | 0.6770 | 0.8663 | 0.7222 | 0.9132 | 0.6878 | 0.9956 | 0.0015 | 0.0014 | 0.0015 | 0.0749 |
| hypergraph:evidence_hypergraph | full | 0.6888 | 0.7704 | 0.8193 | 0.8193 | 0.8714 | 0.8889 | 0.9409 | 0.7011 | 0.9956 | 0.0015 | 0.0014 | 0.0015 | 0.1359 |
| hypergraph:evidence_hypergraph | no_bm25 | 0.5304 | 0.5833 | 0.5957 | 0.5957 | 0.6861 | 0.7222 | 0.8230 | 0.5979 | 0.9956 | 0.0015 | 0.0014 | 0.0015 | 0.0990 |
| hypergraph:evidence_hypergraph | no_graph_rerank | 0.6888 | 0.7704 | 0.8193 | 0.8193 | 0.8714 | 0.8889 | 0.9409 | 0.7011 | 0.9956 | 0.0015 | 0.0014 | 0.0015 | 0.1323 |

## Paired effects on Recall@20

| System | Variant | Baseline | Delta | 95% CI | p |
|---|---|---|---:|---:|---:|
| graph:entity_relation | dense_only | full | -0.0181 | [-0.0445, +0.0075] | 0.1868 |
| graph:entity_relation | index_multi | index_hybrid | -0.0123 | [-0.0317, +0.0053] | 0.2416 |
| graph:entity_relation | index_single | index_hybrid | -0.0141 | [-0.0309, +0.0009] | 0.0938 |
| graph:entity_relation | no_bm25 | full | -0.0026 | [-0.0079, +0.0000] | 1.0000 |
| graph:entity_relation | no_graph_rerank | full | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| graph:reified_fact | dense_only | full | -0.0317 | [-0.0600, -0.0049] | 0.0265 |
| graph:reified_fact | index_multi | index_hybrid | +0.0004 | [-0.0119, +0.0119] | 1.0000 |
| graph:reified_fact | index_single | index_hybrid | +0.0035 | [-0.0159, +0.0220] | 0.7835 |
| graph:reified_fact | no_bm25 | full | +0.0018 | [-0.0079, +0.0115] | 0.6846 |
| graph:reified_fact | no_graph_rerank | full | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| hypergraph:evidence_hypergraph | dense_only | full | -0.1423 | [-0.1929, -0.0922] | 0.0001 |
| hypergraph:evidence_hypergraph | no_bm25 | full | -0.2236 | [-0.2661, -0.1831] | 0.0001 |
| hypergraph:evidence_hypergraph | no_graph_rerank | full | +0.0000 | [+0.0000, +0.0000] | 1.0000 |

## Manifest

```json
{
  "track": "end_to_end_retrieval_runtime_ablation",
  "held_out_examples": 63,
  "seeds": [
    13,
    42,
    73
  ],
  "fixed_tuned_reranker": true,
  "generator_disabled": true,
  "candidate_budget": 60,
  "top_k": 40,
  "timing_cache_policy": "query_embedding_and_reranker_caches_cleared_per_condition",
  "cuda_synchronized": true
}
```

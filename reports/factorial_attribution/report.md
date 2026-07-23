# Exact factorial attribution

Raw dense retrieval is always present. Binary factors are attributed with exact Shapley values; LoRA is the contextual increment over the base reranker.

## Anchor configurations

| System | Condition | Fact R@20 | MRR | Passage R@5 | Path F1 |
|---|---|---:|---:|---:|---:|
| graph:entity_relation | full_base | 0.9242 | 0.8353 | 0.8915 | 0.6746 |
| graph:entity_relation | full_lora | 0.9409 | 0.8772 | 0.9206 | 0.7540 |
| graph:entity_relation | full_no_neural_reranker | 0.7942 | 0.3789 | 0.6376 | 0.3889 |
| graph:entity_relation | raw_minimal | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| graph:reified_fact | full_base | 0.9312 | 0.8380 | 0.8836 | 0.6667 |
| graph:reified_fact | full_lora | 0.9489 | 0.8877 | 0.9259 | 0.7513 |
| graph:reified_fact | full_no_neural_reranker | 0.7164 | 0.4100 | 0.5952 | 0.3730 |
| graph:reified_fact | raw_minimal | 0.6981 | 0.7459 | 0.7381 | 0.5899 |
| hypergraph:evidence_hypergraph | full_base | 0.8767 | 0.8437 | 0.8598 | 0.6508 |
| hypergraph:evidence_hypergraph | full_lora | 0.8794 | 0.8926 | 0.8942 | 0.7381 |
| hypergraph:evidence_hypergraph | full_no_neural_reranker | 0.7471 | 0.4918 | 0.6640 | 0.4101 |
| hypergraph:evidence_hypergraph | raw_minimal | 0.6981 | 0.7459 | 0.7381 | 0.5899 |

## Exact Shapley attribution

| System | Factor | Fact R@20 | 95% CI | p | MRR | Path F1 |
|---|---|---:|---:|---:|---:|---:|
| graph:entity_relation | base_reranker | +0.0689 | [+0.0550, +0.0830] | 0.0001 | +0.2652 | +0.1635 |
| graph:entity_relation | bm25 | +0.2630 | [+0.2388, +0.2867] | 0.0001 | +0.2241 | +0.1805 |
| graph:entity_relation | entity_expansion | +0.5987 | [+0.5708, +0.6263] | 0.0001 | +0.3853 | +0.3577 |
| graph:entity_relation | graph_rerank | -0.0019 | [-0.0045, +0.0005] | 0.1390 | -0.0074 | -0.0063 |
| graph:entity_relation | spectral | -0.0045 | [-0.0154, +0.0066] | 0.4379 | -0.0319 | -0.0208 |
| graph:reified_fact | base_reranker | +0.0966 | [+0.0806, +0.1131] | 0.0001 | +0.2694 | +0.1935 |
| graph:reified_fact | bm25 | +0.0303 | [+0.0193, +0.0416] | 0.0001 | +0.0081 | -0.0159 |
| graph:reified_fact | entity_expansion | +0.0878 | [+0.0662, +0.1100] | 0.0001 | -0.0483 | -0.0196 |
| graph:reified_fact | graph_rerank | -0.0179 | [-0.0249, -0.0106] | 0.0001 | -0.1496 | -0.0682 |
| graph:reified_fact | spectral | +0.0362 | [+0.0158, +0.0568] | 0.0005 | +0.0124 | -0.0130 |
| hypergraph:evidence_hypergraph | base_reranker | +0.0503 | [+0.0387, +0.0628] | 0.0001 | +0.2249 | +0.1581 |
| hypergraph:evidence_hypergraph | bm25 | +0.1199 | [+0.0930, +0.1475] | 0.0001 | -0.0006 | -0.0342 |
| hypergraph:evidence_hypergraph | graph_rerank | -0.0083 | [-0.0136, -0.0032] | 0.0013 | -0.1327 | -0.0575 |
| hypergraph:evidence_hypergraph | spectral | +0.0166 | [-0.0108, +0.0447] | 0.2373 | +0.0062 | -0.0055 |

## LoRA increment over base reranker

| System | Fact R@20 | 95% CI | p | MRR | Path F1 |
|---|---:|---:|---:|---:|---:|
| graph:entity_relation | +0.0104 | [+0.0074, +0.0134] | 0.0001 | +0.0258 | +0.0463 |
| graph:reified_fact | +0.0153 | [+0.0118, +0.0188] | 0.0001 | +0.0496 | +0.0747 |
| hypergraph:evidence_hypergraph | +0.0079 | [+0.0045, +0.0116] | 0.0001 | +0.0489 | +0.0608 |

## Largest pairwise interactions on Fact Recall@20

| System | Pair | Mean second difference |
|---|---|---:|
| graph:entity_relation | entity_expansion×bm25 | -0.5799 |
| graph:reified_fact | entity_expansion×bm25 | -0.1218 |
| graph:entity_relation | entity_expansion×spectral | +0.0859 |
| hypergraph:evidence_hypergraph | bm25×spectral | +0.0832 |
| hypergraph:evidence_hypergraph | bm25×base_reranker | +0.0743 |
| graph:reified_fact | entity_expansion×base_reranker | +0.0727 |
| graph:reified_fact | bm25×base_reranker | +0.0614 |
| graph:reified_fact | graph_rerank×base_reranker | +0.0575 |
| graph:entity_relation | entity_expansion×base_reranker | +0.0567 |
| graph:entity_relation | bm25×spectral | -0.0427 |
| graph:reified_fact | entity_expansion×graph_rerank | -0.0381 |
| graph:entity_relation | bm25×base_reranker | +0.0378 |
| graph:reified_fact | entity_expansion×spectral | -0.0338 |
| graph:reified_fact | spectral×graph_rerank | -0.0334 |
| hypergraph:evidence_hypergraph | spectral×graph_rerank | -0.0289 |

## Manifest

```json
{
  "track": "exact_factorial_retrieval_attribution",
  "held_out_examples": 63,
  "seeds": [
    13,
    42,
    73
  ],
  "graph_factors": [
    "entity_expansion",
    "bm25",
    "spectral",
    "graph_rerank",
    "base_reranker"
  ],
  "hypergraph_factors": [
    "bm25",
    "spectral",
    "graph_rerank",
    "base_reranker"
  ],
  "raw_dense_source_always_present": true,
  "fusion": "reciprocal_rank_fusion",
  "lora_effect": "conditional Base BGE reranker to LoRA reranker increment",
  "candidate_budget": 60,
  "top_k": 40,
  "generator_disabled": true
}
```

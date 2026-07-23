# Embedding/Reranker fine-tuning × random-seed ablation

Held-out examples: 63; seeds: 13, 42, 73

All models use the same fixed ID-based train/validation/test partition. Mean±SD is across seeds.

| Embedding | Reranker | System | R@20 | MRR | nDCG@10 |
|---|---|---|---:|---:|---:|
| base | base | vanilla_rag/dense_bge_compatible | 0.7061±0.0000 | 0.8069±0.0000 | 0.6443±0.0000 |
| base | base | hypergraph/evidence_hypergraph | 0.6851±0.0252 | 0.8104±0.0142 | 0.6337±0.0019 |
| base | base | graph/entity_relation | 0.8619±0.0092 | 0.8162±0.0049 | 0.6732±0.0128 |
| base | base | graph/reified_fact | 0.9052±0.0265 | 0.8351±0.0014 | 0.7051±0.0032 |
| base | tuned | vanilla_rag/dense_bge_compatible | 0.7061±0.0000 | 0.8651±0.0092 | 0.6786±0.0042 |
| base | tuned | hypergraph/evidence_hypergraph | 0.6851±0.0252 | 0.8573±0.0117 | 0.6642±0.0025 |
| base | tuned | graph/entity_relation | 0.8743±0.0130 | 0.8658±0.0141 | 0.7262±0.0051 |
| base | tuned | graph/reified_fact | 0.9140±0.0247 | 0.8877±0.0021 | 0.7597±0.0108 |
| tuned | base | vanilla_rag/dense_bge_compatible | 0.7299±0.0000 | 0.8201±0.0000 | 0.6595±0.0000 |
| tuned | base | hypergraph/evidence_hypergraph | 0.7190±0.0109 | 0.8250±0.0119 | 0.6554±0.0196 |
| tuned | base | graph/entity_relation | 0.8522±0.0035 | 0.8133±0.0191 | 0.6665±0.0061 |
| tuned | base | graph/reified_fact | 0.9021±0.0317 | 0.8351±0.0014 | 0.7033±0.0065 |
| tuned | tuned | vanilla_rag/dense_bge_compatible | 0.7299±0.0000 | 0.8796±0.0092 | 0.6925±0.0042 |
| tuned | tuned | hypergraph/evidence_hypergraph | 0.7190±0.0109 | 0.8633±0.0170 | 0.6838±0.0224 |
| tuned | tuned | graph/entity_relation | 0.8672±0.0048 | 0.8629±0.0235 | 0.7215±0.0049 |
| tuned | tuned | graph/reified_fact | 0.9083±0.0345 | 0.8877±0.0020 | 0.7587±0.0126 |

## Recall@20 ablation effects

| System | Embedding only | Reranker only | Joint | Interaction |
|---|---:|---:|---:|---:|
| graph:reified_fact | -0.0031 | +0.0088 | +0.0031 | -0.0026 |
| hypergraph:evidence_hypergraph | +0.0340 | +0.0000 | +0.0340 | +0.0000 |
| vanilla_rag:dense_bge_compatible | +0.0238 | +0.0000 | +0.0238 | +0.0000 |
| graph:entity_relation | -0.0097 | +0.0123 | +0.0053 | +0.0026 |

## Paired randomization tests on Recall@20

| Condition | Difference | 95% bootstrap CI | p |
|---|---:|---:|---:|
| base:tuned:entity_relation_minus_hypergraph | 0.1892 | [0.1208, 0.2575] | 0.0001 |
| base:tuned:reified_fact_minus_hypergraph | 0.2289 | [0.1708, 0.2903] | 0.0001 |
| base:base:entity_relation_minus_hypergraph | 0.1768 | [0.1102, 0.2443] | 0.0001 |
| base:base:reified_fact_minus_hypergraph | 0.2201 | [0.1633, 0.2796] | 0.0001 |
| tuned:tuned:entity_relation_minus_hypergraph | 0.1481 | [0.0825, 0.2152] | 0.0001 |
| tuned:tuned:reified_fact_minus_hypergraph | 0.1892 | [0.1359, 0.2466] | 0.0001 |
| tuned:base:entity_relation_minus_hypergraph | 0.1332 | [0.0697, 0.1980] | 0.0003 |
| tuned:base:reified_fact_minus_hypergraph | 0.1831 | [0.1325, 0.2372] | 0.0001 |

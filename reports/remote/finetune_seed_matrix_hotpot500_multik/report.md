# Embedding/Reranker fine-tuning × random-seed ablation

Held-out examples: 63; seeds: 13, 42, 73

All models use the same fixed ID-based train/validation/test partition. Mean±SD is across seeds.

| Embedding | Reranker | System | R@20 | MRR | nDCG@10 |
|---|---|---|---:|---:|---:|
| base | base | vanilla_rag/dense_bge_compatible | 0.7061±0.0000 | 0.8069±0.0000 | 0.6443±0.0000 |
| base | base | hypergraph/evidence_hypergraph | 0.6851±0.0252 | 0.8104±0.0142 | 0.6337±0.0019 |
| base | base | graph/entity_relation | 0.8619±0.0092 | 0.8165±0.0045 | 0.6732±0.0128 |
| base | base | graph/reified_fact | 0.9052±0.0265 | 0.8354±0.0011 | 0.7051±0.0032 |
| base | tuned | vanilla_rag/dense_bge_compatible | 0.7061±0.0000 | 0.8651±0.0092 | 0.6786±0.0042 |
| base | tuned | hypergraph/evidence_hypergraph | 0.6851±0.0252 | 0.8573±0.0117 | 0.6642±0.0025 |
| base | tuned | graph/entity_relation | 0.8743±0.0130 | 0.8659±0.0140 | 0.7262±0.0051 |
| base | tuned | graph/reified_fact | 0.9140±0.0247 | 0.8878±0.0018 | 0.7597±0.0108 |
| tuned | base | vanilla_rag/dense_bge_compatible | 0.7299±0.0000 | 0.8201±0.0000 | 0.6595±0.0000 |
| tuned | base | hypergraph/evidence_hypergraph | 0.7190±0.0109 | 0.8250±0.0119 | 0.6554±0.0196 |
| tuned | base | graph/entity_relation | 0.8522±0.0035 | 0.8137±0.0187 | 0.6665±0.0061 |
| tuned | base | graph/reified_fact | 0.9021±0.0317 | 0.8353±0.0012 | 0.7033±0.0065 |
| tuned | tuned | vanilla_rag/dense_bge_compatible | 0.7299±0.0000 | 0.8796±0.0092 | 0.6925±0.0042 |
| tuned | tuned | hypergraph/evidence_hypergraph | 0.7190±0.0109 | 0.8633±0.0170 | 0.6838±0.0224 |
| tuned | tuned | graph/entity_relation | 0.8672±0.0048 | 0.8632±0.0232 | 0.7215±0.0049 |
| tuned | tuned | graph/reified_fact | 0.9083±0.0345 | 0.8878±0.0018 | 0.7587±0.0126 |

## Multi-k retrieval metrics

| Embedding | Reranker | System | R@5 | R@10 | R@20 | R@30 | R@40 | P@10 | P@20 | Hit@10 | Complete@20 | Complete@40 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | base | vanilla_rag/dense_bge_compatible | 0.6095 | 0.6849 | 0.7061 | 0.7061 | 0.7061 | 0.1556 | 0.0802 | 0.9524 | 0.4603 | 0.4603 |
| base | base | hypergraph/evidence_hypergraph | 0.6040 | 0.6639 | 0.6851 | 0.6851 | 0.6851 | 0.1508 | 0.0778 | 0.9312 | 0.4127 | 0.4127 |
| base | base | graph/entity_relation | 0.6254 | 0.7529 | 0.8619 | 0.8751 | 0.8857 | 0.1746 | 0.1008 | 0.9735 | 0.7143 | 0.7619 |
| base | base | graph/reified_fact | 0.6527 | 0.7975 | 0.9052 | 0.9264 | 0.9449 | 0.1831 | 0.1056 | 0.9947 | 0.7989 | 0.8783 |
| base | tuned | vanilla_rag/dense_bge_compatible | 0.6611 | 0.6849 | 0.7061 | 0.7061 | 0.7061 | 0.1556 | 0.0802 | 0.9524 | 0.4603 | 0.4603 |
| base | tuned | hypergraph/evidence_hypergraph | 0.6406 | 0.6728 | 0.6851 | 0.6851 | 0.6851 | 0.1529 | 0.0778 | 0.9418 | 0.4127 | 0.4127 |
| base | tuned | graph/entity_relation | 0.7139 | 0.7937 | 0.8743 | 0.8804 | 0.8857 | 0.1836 | 0.1021 | 0.9788 | 0.7460 | 0.7619 |
| base | tuned | graph/reified_fact | 0.7398 | 0.8327 | 0.9140 | 0.9396 | 0.9449 | 0.1915 | 0.1061 | 0.9894 | 0.8201 | 0.8783 |
| tuned | base | vanilla_rag/dense_bge_compatible | 0.6280 | 0.7087 | 0.7299 | 0.7299 | 0.7299 | 0.1619 | 0.0833 | 0.9683 | 0.4762 | 0.4762 |
| tuned | base | hypergraph/evidence_hypergraph | 0.6256 | 0.6996 | 0.7190 | 0.7190 | 0.7190 | 0.1593 | 0.0817 | 0.9524 | 0.4656 | 0.4656 |
| tuned | base | graph/entity_relation | 0.6175 | 0.7437 | 0.8522 | 0.8707 | 0.8787 | 0.1725 | 0.0997 | 0.9735 | 0.6931 | 0.7407 |
| tuned | base | graph/reified_fact | 0.6527 | 0.7931 | 0.9021 | 0.9206 | 0.9392 | 0.1820 | 0.1050 | 0.9894 | 0.7884 | 0.8624 |
| tuned | tuned | vanilla_rag/dense_bge_compatible | 0.6743 | 0.7087 | 0.7299 | 0.7299 | 0.7299 | 0.1619 | 0.0833 | 0.9683 | 0.4762 | 0.4762 |
| tuned | tuned | hypergraph/evidence_hypergraph | 0.6622 | 0.7067 | 0.7190 | 0.7190 | 0.7190 | 0.1608 | 0.0817 | 0.9524 | 0.4656 | 0.4656 |
| tuned | tuned | graph/entity_relation | 0.7139 | 0.7884 | 0.8672 | 0.8734 | 0.8787 | 0.1825 | 0.1013 | 0.9735 | 0.7302 | 0.7407 |
| tuned | tuned | graph/reified_fact | 0.7398 | 0.8310 | 0.9083 | 0.9339 | 0.9392 | 0.1910 | 0.1053 | 0.9894 | 0.8042 | 0.8624 |

## Recall@20 ablation effects

| System | Embedding only | Reranker only | Joint | Interaction |
|---|---:|---:|---:|---:|
| graph:entity_relation | -0.0097 | +0.0123 | +0.0053 | +0.0026 |
| hypergraph:evidence_hypergraph | +0.0340 | +0.0000 | +0.0340 | +0.0000 |
| vanilla_rag:dense_bge_compatible | +0.0238 | +0.0000 | +0.0238 | +0.0000 |
| graph:reified_fact | -0.0031 | +0.0088 | +0.0031 | -0.0026 |

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

# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6280 | 0.7087 | 0.7299 | 0.7299 | 0.7299 | 0.9683 | 0.4762 | 0.8201 |
| hypergraph:evidence_hypergraph | 0.6392 | 0.7079 | 0.7159 | 0.7159 | 0.7159 | 0.9524 | 0.4603 | 0.8325 |
| graph:entity_relation | 0.6148 | 0.7556 | 0.8561 | 0.8799 | 0.8852 | 0.9524 | 0.7302 | 0.7959 |
| graph:reified_fact | 0.6519 | 0.8053 | 0.9143 | 0.9381 | 0.9593 | 1.0000 | 0.8095 | 0.8366 |

# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6280 | 0.7087 | 0.7299 | 0.7299 | 0.7299 | 0.9683 | 0.4762 | 0.8201 |
| hypergraph:evidence_hypergraph | 0.5937 | 0.6730 | 0.7101 | 0.7101 | 0.7101 | 0.9365 | 0.4603 | 0.8114 |
| graph:entity_relation | 0.6280 | 0.7423 | 0.8513 | 0.8672 | 0.8725 | 1.0000 | 0.6508 | 0.8332 |
| graph:reified_fact | 0.6492 | 0.7762 | 0.8661 | 0.8820 | 0.8952 | 0.9841 | 0.6984 | 0.8346 |

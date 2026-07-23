# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6095 | 0.6849 | 0.7061 | 0.7061 | 0.7061 | 0.9524 | 0.4603 | 0.8069 |
| hypergraph:evidence_hypergraph | 0.6056 | 0.6717 | 0.7140 | 0.7140 | 0.7140 | 0.9206 | 0.4444 | 0.7962 |
| graph:entity_relation | 0.6228 | 0.7450 | 0.8540 | 0.8619 | 0.8751 | 0.9841 | 0.6508 | 0.8199 |
| graph:reified_fact | 0.6492 | 0.7894 | 0.8754 | 0.8913 | 0.9045 | 1.0000 | 0.7302 | 0.8348 |

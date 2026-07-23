# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6095 | 0.6849 | 0.7061 | 0.7061 | 0.7061 | 0.9524 | 0.4603 | 0.8069 |
| hypergraph:evidence_hypergraph | 0.5995 | 0.6603 | 0.6683 | 0.6683 | 0.6683 | 0.9365 | 0.4127 | 0.8246 |
| graph:entity_relation | 0.6386 | 0.7754 | 0.8720 | 0.8878 | 0.8931 | 0.9683 | 0.7619 | 0.8181 |
| graph:reified_fact | 0.6519 | 0.8053 | 0.9143 | 0.9460 | 0.9672 | 1.0000 | 0.8095 | 0.8366 |

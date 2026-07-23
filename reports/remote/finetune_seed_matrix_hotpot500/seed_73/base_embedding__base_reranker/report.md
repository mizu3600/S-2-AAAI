# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6849 | 0.7061 | 0.8069 | 0.6443 | 0.1429 | 88.57 |
| hypergraph:evidence_hypergraph | 0.6603 | 0.6683 | 0.8246 | 0.6352 | 0.0635 | 45.75 |
| graph:entity_relation | 0.7754 | 0.8720 | 0.8181 | 0.6874 | 0.0794 | 104.53 |
| graph:reified_fact | 0.8053 | 0.9143 | 0.8366 | 0.7069 | 0.0000 | 41.51 |

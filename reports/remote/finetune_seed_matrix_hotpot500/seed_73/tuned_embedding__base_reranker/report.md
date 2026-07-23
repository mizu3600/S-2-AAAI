# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.7087 | 0.7299 | 0.8201 | 0.6595 | 0.1746 | 101.49 |
| hypergraph:evidence_hypergraph | 0.7079 | 0.7159 | 0.8325 | 0.6649 | 0.0476 | 36.07 |
| graph:entity_relation | 0.7556 | 0.8561 | 0.7952 | 0.6668 | 0.0952 | 96.55 |
| graph:reified_fact | 0.8053 | 0.9143 | 0.8366 | 0.7069 | 0.0000 | 45.18 |

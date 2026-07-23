# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.7087 | 0.7299 | 0.8743 | 0.6913 | 0.1746 | 119.99 |
| hypergraph:evidence_hypergraph | 0.7079 | 0.7159 | 0.8704 | 0.6918 | 0.0476 | 59.54 |
| graph:entity_relation | 0.7913 | 0.8720 | 0.8416 | 0.7177 | 0.0952 | 165.26 |
| graph:reified_fact | 0.8442 | 0.9328 | 0.8881 | 0.7647 | 0.0000 | 68.28 |

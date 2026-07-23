# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6849 | 0.7061 | 0.8069 | 0.6443 | 0.1429 | 76.30 |
| hypergraph:evidence_hypergraph | 0.6598 | 0.6730 | 0.8105 | 0.6316 | 0.1111 | 39.10 |
| graph:entity_relation | 0.7384 | 0.8598 | 0.8107 | 0.6626 | 0.0635 | 113.95 |
| graph:reified_fact | 0.7979 | 0.9259 | 0.8340 | 0.7071 | 0.2222 | 37.45 |

# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6849 | 0.7061 | 0.8069 | 0.6443 | 0.1429 | 81.01 |
| hypergraph:evidence_hypergraph | 0.6717 | 0.7140 | 0.7962 | 0.6343 | 0.0635 | 58.65 |
| graph:entity_relation | 0.7450 | 0.8540 | 0.8199 | 0.6695 | 0.1111 | 97.98 |
| graph:reified_fact | 0.7894 | 0.8754 | 0.8348 | 0.7014 | 0.1429 | 29.14 |

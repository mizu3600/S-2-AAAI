# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6849 | 0.7061 | 0.8598 | 0.6752 | 0.1429 | 79.75 |
| hypergraph:evidence_hypergraph | 0.6929 | 0.7140 | 0.8439 | 0.6655 | 0.0635 | 73.07 |
| graph:entity_relation | 0.7860 | 0.8619 | 0.8819 | 0.7242 | 0.1111 | 129.65 |
| graph:reified_fact | 0.8093 | 0.8860 | 0.8895 | 0.7474 | 0.1429 | 35.09 |

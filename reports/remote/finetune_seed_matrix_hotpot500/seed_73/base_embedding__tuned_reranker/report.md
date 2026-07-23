# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6849 | 0.7061 | 0.8598 | 0.6774 | 0.1429 | 82.66 |
| hypergraph:evidence_hypergraph | 0.6603 | 0.6683 | 0.8624 | 0.6613 | 0.0635 | 59.94 |
| graph:entity_relation | 0.8019 | 0.8878 | 0.8563 | 0.7320 | 0.0794 | 136.15 |
| graph:reified_fact | 0.8442 | 0.9328 | 0.8881 | 0.7645 | 0.0000 | 52.33 |

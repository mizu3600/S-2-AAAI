# hotpotqa Graph/Hypergraph fair comparison

Examples: 50

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.2367 | 0.2533 | 0.2423 | 0.1826 | 0.2000 | 0.40 |
| hypergraph:evidence_hypergraph | 0.3273 | 0.4307 | 0.2817 | 0.2216 | 0.1400 | 9.26 |
| graph:entity_relation | 0.3390 | 0.5707 | 0.2723 | 0.2236 | 0.0600 | 9.60 |
| graph:reified_fact | 0.3633 | 0.6267 | 0.2497 | 0.2344 | 0.1800 | 8.97 |

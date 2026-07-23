# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.7087 | 0.7299 | 0.8743 | 0.6890 | 0.1746 | 104.26 |
| hypergraph:evidence_hypergraph | 0.6889 | 0.7101 | 0.8439 | 0.6585 | 0.0794 | 77.08 |
| graph:entity_relation | 0.7860 | 0.8672 | 0.8881 | 0.7270 | 0.1111 | 136.52 |
| graph:reified_fact | 0.8040 | 0.8688 | 0.8894 | 0.7443 | 0.1587 | 39.84 |

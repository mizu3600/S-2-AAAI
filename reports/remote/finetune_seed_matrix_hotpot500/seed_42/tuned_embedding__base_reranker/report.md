# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.7087 | 0.7299 | 0.8201 | 0.6595 | 0.1746 | 100.19 |
| hypergraph:evidence_hypergraph | 0.6730 | 0.7101 | 0.8114 | 0.6328 | 0.0794 | 43.01 |
| graph:entity_relation | 0.7423 | 0.8513 | 0.8332 | 0.6725 | 0.1111 | 91.41 |
| graph:reified_fact | 0.7762 | 0.8661 | 0.8346 | 0.6958 | 0.1587 | 32.06 |

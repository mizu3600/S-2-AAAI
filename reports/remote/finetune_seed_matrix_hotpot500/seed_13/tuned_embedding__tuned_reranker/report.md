# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.7087 | 0.7299 | 0.8902 | 0.6971 | 0.1746 | 101.17 |
| hypergraph:evidence_hypergraph | 0.7233 | 0.7312 | 0.8757 | 0.7011 | 0.1270 | 37.87 |
| graph:entity_relation | 0.7878 | 0.8624 | 0.8590 | 0.7197 | 0.0476 | 164.75 |
| graph:reified_fact | 0.8447 | 0.9233 | 0.8854 | 0.7673 | 0.1746 | 53.69 |

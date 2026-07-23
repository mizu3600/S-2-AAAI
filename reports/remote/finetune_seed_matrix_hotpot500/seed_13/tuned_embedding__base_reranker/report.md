# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.7087 | 0.7299 | 0.8201 | 0.6595 | 0.1746 | 93.41 |
| hypergraph:evidence_hypergraph | 0.7180 | 0.7312 | 0.8312 | 0.6685 | 0.1270 | 27.90 |
| graph:entity_relation | 0.7331 | 0.8492 | 0.8115 | 0.6603 | 0.0476 | 110.95 |
| graph:reified_fact | 0.7979 | 0.9259 | 0.8340 | 0.7071 | 0.1746 | 42.97 |

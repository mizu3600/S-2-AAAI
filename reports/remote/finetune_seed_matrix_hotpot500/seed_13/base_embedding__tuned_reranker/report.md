# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@10 | R@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6849 | 0.7061 | 0.8757 | 0.6833 | 0.1429 | 81.59 |
| hypergraph:evidence_hypergraph | 0.6651 | 0.6730 | 0.8656 | 0.6659 | 0.1111 | 47.36 |
| graph:entity_relation | 0.7931 | 0.8730 | 0.8590 | 0.7224 | 0.0635 | 158.00 |
| graph:reified_fact | 0.8447 | 0.9233 | 0.8854 | 0.7673 | 0.2222 | 41.71 |

# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6743 | 0.7087 | 0.7299 | 0.7299 | 0.7299 | 0.9683 | 0.4762 | 0.8902 |
| hypergraph:evidence_hypergraph | 0.6836 | 0.7233 | 0.7312 | 0.7312 | 0.7312 | 0.9683 | 0.4762 | 0.8757 |
| graph:entity_relation | 0.7146 | 0.7878 | 0.8624 | 0.8704 | 0.8783 | 0.9683 | 0.7302 | 0.8594 |
| graph:reified_fact | 0.7437 | 0.8447 | 0.9233 | 0.9550 | 0.9630 | 0.9841 | 0.8571 | 0.8859 |

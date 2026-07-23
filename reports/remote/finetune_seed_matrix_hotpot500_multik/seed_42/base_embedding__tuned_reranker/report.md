# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6611 | 0.6849 | 0.7061 | 0.7061 | 0.7061 | 0.9524 | 0.4603 | 0.8598 |
| hypergraph:evidence_hypergraph | 0.6466 | 0.6929 | 0.7140 | 0.7140 | 0.7140 | 0.9365 | 0.4444 | 0.8439 |
| graph:entity_relation | 0.7013 | 0.7860 | 0.8619 | 0.8672 | 0.8751 | 1.0000 | 0.6825 | 0.8819 |
| graph:reified_fact | 0.7325 | 0.8093 | 0.8860 | 0.9045 | 0.9045 | 0.9841 | 0.7619 | 0.8895 |

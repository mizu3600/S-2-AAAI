# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6095 | 0.6849 | 0.7061 | 0.7061 | 0.7061 | 0.9524 | 0.4603 | 0.8069 |
| hypergraph:evidence_hypergraph | 0.6069 | 0.6598 | 0.6730 | 0.6730 | 0.6730 | 0.9365 | 0.3810 | 0.8105 |
| graph:entity_relation | 0.6148 | 0.7384 | 0.8598 | 0.8757 | 0.8889 | 0.9683 | 0.7302 | 0.8114 |
| graph:reified_fact | 0.6571 | 0.7979 | 0.9259 | 0.9418 | 0.9630 | 0.9841 | 0.8571 | 0.8347 |

# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6611 | 0.6849 | 0.7061 | 0.7061 | 0.7061 | 0.9524 | 0.4603 | 0.8757 |
| hypergraph:evidence_hypergraph | 0.6399 | 0.6651 | 0.6730 | 0.6730 | 0.6730 | 0.9524 | 0.3810 | 0.8656 |
| graph:entity_relation | 0.7040 | 0.7931 | 0.8730 | 0.8810 | 0.8889 | 0.9683 | 0.7619 | 0.8594 |
| graph:reified_fact | 0.7437 | 0.8447 | 0.9233 | 0.9550 | 0.9630 | 0.9841 | 0.8571 | 0.8859 |

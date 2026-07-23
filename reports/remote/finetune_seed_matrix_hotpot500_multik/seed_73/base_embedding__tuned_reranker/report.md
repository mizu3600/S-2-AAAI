# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6611 | 0.6849 | 0.7061 | 0.7061 | 0.7061 | 0.9524 | 0.4603 | 0.8598 |
| hypergraph:evidence_hypergraph | 0.6352 | 0.6603 | 0.6683 | 0.6683 | 0.6683 | 0.9365 | 0.4127 | 0.8624 |
| graph:entity_relation | 0.7365 | 0.8019 | 0.8878 | 0.8931 | 0.8931 | 0.9683 | 0.7937 | 0.8563 |
| graph:reified_fact | 0.7431 | 0.8442 | 0.9328 | 0.9593 | 0.9672 | 1.0000 | 0.8413 | 0.8881 |

# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6743 | 0.7087 | 0.7299 | 0.7299 | 0.7299 | 0.9683 | 0.4762 | 0.8743 |
| hypergraph:evidence_hypergraph | 0.6696 | 0.7079 | 0.7159 | 0.7159 | 0.7159 | 0.9524 | 0.4603 | 0.8704 |
| graph:entity_relation | 0.7206 | 0.7913 | 0.8720 | 0.8772 | 0.8852 | 0.9524 | 0.7619 | 0.8421 |
| graph:reified_fact | 0.7431 | 0.8442 | 0.9328 | 0.9513 | 0.9593 | 1.0000 | 0.8413 | 0.8881 |

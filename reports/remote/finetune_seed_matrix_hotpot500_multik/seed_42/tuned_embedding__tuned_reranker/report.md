# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6743 | 0.7087 | 0.7299 | 0.7299 | 0.7299 | 0.9683 | 0.4762 | 0.8743 |
| hypergraph:evidence_hypergraph | 0.6333 | 0.6889 | 0.7101 | 0.7101 | 0.7101 | 0.9365 | 0.4603 | 0.8439 |
| graph:entity_relation | 0.7066 | 0.7860 | 0.8672 | 0.8725 | 0.8725 | 1.0000 | 0.6984 | 0.8881 |
| graph:reified_fact | 0.7325 | 0.8040 | 0.8688 | 0.8952 | 0.8952 | 0.9841 | 0.7143 | 0.8894 |

# hotpotqa Graph/Hypergraph fair comparison

Examples: 63

| Mode | R@5 | R@10 | R@20 | R@30 | R@40 | Hit@10 | Complete@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_rag:dense_bge_compatible | 0.6280 | 0.7087 | 0.7299 | 0.7299 | 0.7299 | 0.9683 | 0.4762 | 0.8201 |
| hypergraph:evidence_hypergraph | 0.6439 | 0.7180 | 0.7312 | 0.7312 | 0.7312 | 0.9683 | 0.4762 | 0.8312 |
| graph:entity_relation | 0.6095 | 0.7331 | 0.8492 | 0.8651 | 0.8783 | 0.9683 | 0.6984 | 0.8122 |
| graph:reified_fact | 0.6571 | 0.7979 | 0.9259 | 0.9418 | 0.9630 | 0.9841 | 0.8571 | 0.8346 |

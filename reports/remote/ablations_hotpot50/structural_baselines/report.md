# hotpotqa experiment

Examples: 10

| Method | Recall@10 | Recall@20 | MRR | nDCG@10 | Bridge@20 | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| semantic+lap_pe | 0.3250 | 0.8667 | 0.2396 | 0.2068 | 0.3000 | 7.59 |
| laplacian_eigenmaps | 0.5000 | 0.7833 | 0.2734 | 0.2876 | 0.1000 | 7.01 |

## By hop count

### 2

| Method | Recall@20 | MRR | Bridge@20 |
|---|---:|---:|---:|
| semantic+lap_pe | 0.8667 | 0.2396 | 0.3000 |
| laplacian_eigenmaps | 0.7833 | 0.2734 | 0.1000 |


## By query type

### bridge

| Method | Recall@20 | MRR | Bridge@20 |
|---|---:|---:|---:|
| semantic+lap_pe | 0.8095 | 0.2505 | 0.4286 |
| laplacian_eigenmaps | 0.6905 | 0.1797 | 0.1429 |

### comparison

| Method | Recall@20 | MRR | Bridge@20 |
|---|---:|---:|---:|
| semantic+lap_pe | 1.0000 | 0.2143 | 0.0000 |
| laplacian_eigenmaps | 1.0000 | 0.4921 | 0.0000 |


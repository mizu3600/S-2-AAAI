# Stage A training-effect 2×2 control

All cells rank Facts directly before BM25, graph reranking and the neural reranker.

| Condition | R@5 | R@10 | R@20 mean±SD | R@40 | Hit@10 | Complete@20 | MRR | nDCG@10 | ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_bge_identity | 0.5989 | 0.7336 | 0.8664±0.0000 | 0.9656 | 0.9841 | 0.6825 | 0.7588 | 0.6239 | 0.87 |
| untrained_raw | 0.1085 | 0.2248 | 0.4810±0.0342 | 0.9167 | 0.4497 | 0.2275 | 0.1874 | 0.1378 | 10.27 |
| trained_raw | 0.3780 | 0.5033 | 0.7601±0.0204 | 0.9683 | 0.7672 | 0.5344 | 0.4066 | 0.3547 | 9.56 |
| untrained_full | 0.1001 | 0.1954 | 0.4775±0.0338 | 0.9071 | 0.4021 | 0.2328 | 0.1787 | 0.1234 | 9.41 |
| trained_full | 0.3880 | 0.5057 | 0.7693±0.0105 | 0.9683 | 0.7672 | 0.5608 | 0.4133 | 0.3590 | 9.40 |

## Paired effects on Recall@20

| Question | Treatment − control | Delta | 95% CI | p |
|---|---|---:|---:|---:|
| raw_training_effect | trained_raw − untrained_raw | +0.2790 | [+0.2145, +0.3444] | 0.0001 |
| full_training_effect | trained_full − untrained_full | +0.2918 | [+0.2247, +0.3586] | 0.0001 |
| untrained_band_effect | untrained_full − untrained_raw | -0.0035 | [-0.0317, +0.0238] | 0.8239 |
| trained_band_effect | trained_full − trained_raw | +0.0093 | [-0.0066, +0.0251] | 0.3430 |
| trained_full_vs_dense_bge | trained_full − dense_bge_identity | -0.0971 | [-0.1421, -0.0503] | 0.0001 |

## Manifest

```json
{
  "track": "stage_a_training_effect_2x2",
  "held_out_examples": 63,
  "seeds": [
    13,
    42,
    73
  ],
  "ks": [
    1,
    2,
    5,
    10,
    20,
    30,
    40
  ],
  "architecture_matched_initialization": true,
  "same_initialization_within_seed": true,
  "dense_bge_identity_is_reference_not_a_2x2_cell": true,
  "no_bm25_graph_rerank_or_neural_reranker": true
}
```

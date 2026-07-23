# Core Stage A ablation matrix

This is the intrinsic spectral-ranking track before BM25, graph reranking and neural reranking.

| System | Variant | R@5 | R@10 | R@20 | R@40 | Hit@10 | Complete@20 | MRR | nDCG@10 | ms/query |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph:entity_relation | full | 0.0945 | 0.2085 | 0.4372 | 0.8177 | 0.4074 | 0.1852 | 0.1639 | 0.1220 | 2.03 |
| graph:reified_fact | full | 0.2681 | 0.3898 | 0.6142 | 0.9471 | 0.6508 | 0.3810 | 0.3817 | 0.2965 | 2.10 |
| hypergraph:evidence_hypergraph | fixed_gate | 0.3449 | 0.4831 | 0.7415 | 0.9762 | 0.7460 | 0.5185 | 0.4003 | 0.3425 | 9.09 |
| hypergraph:evidence_hypergraph | full | 0.3880 | 0.5057 | 0.7693 | 0.9683 | 0.7672 | 0.5608 | 0.4133 | 0.3590 | 9.42 |
| hypergraph:evidence_hypergraph | no_bridge_loss | 0.3470 | 0.4987 | 0.7515 | 0.9797 | 0.7672 | 0.5344 | 0.4305 | 0.3688 | 8.83 |
| hypergraph:evidence_hypergraph | no_hard_negatives | 0.3214 | 0.4779 | 0.7287 | 0.9630 | 0.7460 | 0.4868 | 0.3948 | 0.3349 | 8.76 |
| hypergraph:evidence_hypergraph | no_high | 0.3763 | 0.4969 | 0.7640 | 0.9683 | 0.7619 | 0.5397 | 0.4090 | 0.3536 | 8.79 |
| hypergraph:evidence_hypergraph | no_low | 0.3776 | 0.4969 | 0.7640 | 0.9683 | 0.7619 | 0.5397 | 0.4091 | 0.3537 | 8.95 |
| hypergraph:evidence_hypergraph | no_mid | 0.3776 | 0.5015 | 0.7587 | 0.9683 | 0.7672 | 0.5397 | 0.4065 | 0.3542 | 8.83 |
| hypergraph:evidence_hypergraph | no_role_gate | 0.3776 | 0.4962 | 0.7640 | 0.9683 | 0.7619 | 0.5450 | 0.4063 | 0.3524 | 3.83 |
| hypergraph:evidence_hypergraph | no_semantic_graph | 0.3763 | 0.5015 | 0.7640 | 0.9683 | 0.7672 | 0.5450 | 0.4061 | 0.3541 | 8.82 |
| hypergraph:evidence_hypergraph | raw_only | 0.3780 | 0.5033 | 0.7601 | 0.9683 | 0.7672 | 0.5344 | 0.4066 | 0.3547 | 8.90 |

## Paired effects versus Full on Recall@20

| Variant | Delta | 95% CI | p |
|---|---:|---:|---:|
| raw_only | -0.0093 | [-0.0251, +0.0066] | 0.3368 |
| no_low | -0.0053 | [-0.0212, +0.0079] | 0.7255 |
| no_mid | -0.0106 | [-0.0265, +0.0026] | 0.2812 |
| no_high | -0.0053 | [-0.0212, +0.0079] | 0.7259 |
| fixed_gate | -0.0278 | [-0.0569, -0.0009] | 0.0559 |
| no_role_gate | -0.0053 | [-0.0185, +0.0079] | 0.6940 |
| no_semantic_graph | -0.0053 | [-0.0185, +0.0079] | 0.6947 |
| no_bridge_loss | -0.0178 | [-0.0543, +0.0195] | 0.3525 |
| no_hard_negatives | -0.0406 | [-0.0635, -0.0203] | 0.0002 |

## Manifest

```json
{
  "track": "stage_a_intrinsic_pre_fusion",
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
  "fixed_encoder": "/models/bge-m3",
  "no_bm25_or_reranker_in_this_track": true,
  "note": "Training variants are independently trained; runtime fusion ablations are reported separately."
}
```

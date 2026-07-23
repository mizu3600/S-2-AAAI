# QMSxE-RAG reproducible delivery report

Dataset: hotpotqa; examples: 5; source: `data/benchmarks/hotpotqa_sample.json`.

## Deliverables

| Deliverable | Artifact |
|---|---|
| Mathematical correctness | [Spectral report](math/spectral_validation.md) |
| Embedding baseline main table | [Baseline report](embedding_baselines/report.md) |
| Hop and query-type tables | [Grouped baseline report](embedding_baselines/report.md) |
| Graph/Hypergraph comparison | [Dual-mode report](dual_mode/report.md) |
| Ablations | [Ablation results](ablations.json) |
| Rebuilt/retrained ablations | [Trained ablations](trained_ablations/trained_ablations.md) |
| Efficiency | [Efficiency metrics](efficiency.json) |
| Band-gate visualization | [Gate weights](gate_weights.svg) |
| Success and failure cases | [Cases](cases.md) |

## Efficiency summary

| Run | Requests | Success rate | P50 ms | P95 ms |
|---|---:|---:|---:|---:|
| Cold | 100 | 1.0000 | 169.645 | 385.760 |
| Warm | 100 | 1.0000 | 0.142 | 179.228 |

## Reproduction

```bash
python scripts/reproduce_delivery.py --dataset hotpotqa --input-path data/benchmarks/hotpotqa_sample.json --limit 5
```

System-level LightRAG/PathRAG/GraphRAG results are imported separately from the isolated Unified-RAG-Evaluation run; smoke results must not be described as publication claims.

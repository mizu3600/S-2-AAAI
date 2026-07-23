# QMSxE-RAG Benchmark V2 Results

This report is produced by the independent protocol-first chain. Existing production pipelines are unchanged.

| suite | system | candidate_count | examples | fact_hit_at_1 | fact_mrr | fact_recall_at_10 | fact_recall_at_40 | passage_recall_at_10 | path_f1 | answer_em | answer_f1 | citation_f1 | joint_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hotpotqa_dev | trained_parity:graph:entity_relation:seed_42 | 10 | 288 | 0.7153 | 0.8332 | 0.8709 | 0.9964 | 0.9983 | 0.7240 | 0.0000 | 0.0295 | 0.5404 | 0.0171 |
| hotpotqa_dev | trained_parity:graph:reified_fact:seed_42 | 10 | 288 | 0.7153 | 0.8332 | 0.8709 | 0.9981 | 1.0000 | 0.7240 | 0.0000 | 0.0295 | 0.5404 | 0.0171 |
| hotpotqa_dev | trained_parity:hypergraph:evidence_hypergraph:seed_42 | 10 | 288 | 0.7118 | 0.8299 | 0.8681 | 0.9935 | 0.9965 | 0.7205 | 0.0000 | 0.0294 | 0.5390 | 0.0171 |

## Protocol manifest

```json
{
  "protocol": "benchmark_v2_trained_parity",
  "examples": 288,
  "seed": 42,
  "fact_budget": 60,
  "reranker_inputs": 60,
  "structured_track": false,
  "note": "Uses existing trained checkpoints and legacy extraction; structured-role track requires retraining."
}
```

# QMSxE-RAG Benchmark V2 Results

This report is produced by the independent protocol-first chain. Existing production pipelines are unchanged.

| suite | system | candidate_count | examples | fact_hit_at_1 | fact_mrr | fact_recall_at_10 | fact_recall_at_40 | passage_recall_at_10 | path_f1 | answer_em | answer_f1 | citation_f1 | joint_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hotpotqa_dev | controlled_entity_relation | 10 | 7405 | 0.6548 | 0.7671 | 0.7469 | 0.9825 | 0.9980 | 0.5708 | 0.0451 | 0.0908 | 0.4487 | 0.0513 |
| hotpotqa_dev | controlled_entity_relation | 100 | 1000 | 0.6390 | 0.7515 | 0.7033 | 0.8778 | 0.9110 | 0.5380 | 0.0510 | 0.1058 | 0.4300 | 0.0619 |
| hotpotqa_dev | controlled_entity_relation | 1000 | 200 | 0.6150 | 0.7295 | 0.6615 | 0.8057 | 0.8450 | 0.5150 | 0.0600 | 0.1138 | 0.4088 | 0.0706 |
| hotpotqa_dev | controlled_hypergraph | 10 | 7405 | 0.6550 | 0.7672 | 0.7467 | 0.9827 | 0.9979 | 0.5707 | 0.0451 | 0.0909 | 0.4487 | 0.0513 |
| hotpotqa_dev | controlled_hypergraph | 100 | 1000 | 0.6390 | 0.7514 | 0.7031 | 0.8868 | 0.9095 | 0.5370 | 0.0510 | 0.1058 | 0.4291 | 0.0619 |
| hotpotqa_dev | controlled_hypergraph | 1000 | 200 | 0.6150 | 0.7293 | 0.6565 | 0.8053 | 0.8400 | 0.5125 | 0.0600 | 0.1138 | 0.4063 | 0.0706 |
| hotpotqa_dev | controlled_reified_fact | 10 | 7405 | 0.6548 | 0.7671 | 0.7468 | 0.9831 | 0.9983 | 0.5709 | 0.0451 | 0.0909 | 0.4489 | 0.0513 |
| hotpotqa_dev | controlled_reified_fact | 100 | 1000 | 0.6390 | 0.7514 | 0.7025 | 0.8659 | 0.9105 | 0.5375 | 0.0510 | 0.1058 | 0.4294 | 0.0619 |
| hotpotqa_dev | controlled_reified_fact | 1000 | 200 | 0.6150 | 0.7285 | 0.6632 | 0.8053 | 0.8425 | 0.5175 | 0.0600 | 0.1138 | 0.4088 | 0.0706 |
| nary_stress | controlled_entity_relation | 5 | 500 | 0.0000 | 0.2479 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.2780 | 0.2820 | 0.0000 | 0.0000 |
| nary_stress | controlled_hypergraph | 5 | 500 | 0.0000 | 0.2479 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.2780 | 0.2820 | 0.0000 | 0.0000 |
| nary_stress | controlled_reified_fact | 5 | 500 | 0.0000 | 0.2479 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.2780 | 0.2820 | 0.0000 | 0.0000 |

## Protocol manifest

```json
{
  "protocol": "benchmark_v2",
  "dataset": "HotpotQA distractor dev v1",
  "dataset_sha256": "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
  "dataset_examples": 7405,
  "evaluation_split": "official dev (distractor)",
  "training_split_used": false,
  "query_types": {
    "bridge": 5918,
    "comparison": 1487
  },
  "candidate_matrix": [
    {
      "candidate_count": 10,
      "examples": 7405
    },
    {
      "candidate_count": 100,
      "examples": 1000
    },
    {
      "candidate_count": 1000,
      "examples": 200
    }
  ],
  "nary_examples": 500,
  "shared_retrieval_budget": 60,
  "shared_reranker_inputs": 60,
  "shared_output_facts": 40,
  "ranking_source": "internal fact scores",
  "structured_extractor": "rule-roles-v2",
  "expanded_candidate_source": "label-free deterministic sampling from other dev passages",
  "elapsed_seconds": 198.79195895799967
}
```

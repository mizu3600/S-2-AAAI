# Unified Native RAG Benchmark

## Passage retrieval

| Method | MRR | R@1 | Hit/Acc@1 | Complete@1 | R@2 | Hit/Acc@2 | Complete@2 | R@5 | Hit/Acc@5 | Complete@5 | R@10 | Hit/Acc@10 | Complete@10 | R@20 | Hit/Acc@20 | Complete@20 | R@30 | Hit/Acc@30 | Complete@30 | R@40 | Hit/Acc@40 | Complete@40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline:bm25 | 0.8077 | 0.3490 | 0.6979 | 0.0000 | 0.5087 | 0.8299 | 0.1875 | 0.7500 | 0.9583 | 0.5417 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| baseline:dense_bge_m3 | 0.9538 | 0.4601 | 0.9201 | 0.0000 | 0.7535 | 0.9722 | 0.5347 | 0.9201 | 0.9965 | 0.8438 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| official:graphrag | 0.6554 | 0.2969 | 0.5938 | 0.0000 | 0.4740 | 0.6667 | 0.2812 | 0.5799 | 0.7292 | 0.4306 | 0.6701 | 0.7778 | 0.5625 | 0.6701 | 0.7778 | 0.5625 | 0.6701 | 0.7778 | 0.5625 | 0.6701 | 0.7778 | 0.5625 |
| official:hypergraphrag | 0.9011 | 0.4167 | 0.8333 | 0.0000 | 0.6684 | 0.9340 | 0.4028 | 0.8767 | 0.9965 | 0.7569 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| official:lightrag | 0.8421 | 0.3733 | 0.7465 | 0.0000 | 0.5920 | 0.8750 | 0.3090 | 0.8264 | 0.9722 | 0.6806 | 0.9444 | 0.9965 | 0.8924 | 0.9444 | 0.9965 | 0.8924 | 0.9444 | 0.9965 | 0.8924 | 0.9444 | 0.9965 | 0.8924 |
| official:pathrag | 0.4675 | 0.1128 | 0.2257 | 0.0000 | 0.2465 | 0.4514 | 0.0417 | 0.5938 | 0.8333 | 0.3542 | 0.9931 | 0.9931 | 0.9931 | 0.9931 | 0.9931 | 0.9931 | 0.9931 | 0.9931 | 0.9931 | 0.9931 | 0.9931 | 0.9931 |
| qmsxe:graph:entity_relation | 0.9272 | 0.4375 | 0.8750 | 0.0000 | 0.7240 | 0.9549 | 0.4931 | 0.9028 | 0.9965 | 0.8090 | 0.9983 | 1.0000 | 0.9965 | 0.9983 | 1.0000 | 0.9965 | 0.9983 | 1.0000 | 0.9965 | 0.9983 | 1.0000 | 0.9965 |
| qmsxe:graph:reified_fact | 0.9272 | 0.4375 | 0.8750 | 0.0000 | 0.7240 | 0.9549 | 0.4931 | 0.9028 | 0.9965 | 0.8090 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| qmsxe:hypergraph:evidence_hypergraph | 0.9272 | 0.4375 | 0.8750 | 0.0000 | 0.7205 | 0.9549 | 0.4861 | 0.8993 | 0.9965 | 0.8021 | 0.9965 | 1.0000 | 0.9931 | 0.9965 | 1.0000 | 0.9931 | 0.9965 | 1.0000 | 0.9931 | 0.9965 | 1.0000 | 0.9931 |

## Fact retrieval

| Method | MRR | R@1 | Hit/Acc@1 | Complete@1 | R@2 | Hit/Acc@2 | Complete@2 | R@5 | Hit/Acc@5 | Complete@5 | R@10 | Hit/Acc@10 | Complete@10 | R@20 | Hit/Acc@20 | Complete@20 | R@30 | Hit/Acc@30 | Complete@30 | R@40 | Hit/Acc@40 | Complete@40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline:bm25 | 0.5894 | 0.1815 | 0.4514 | 0.0000 | 0.2692 | 0.5729 | 0.0000 | 0.4639 | 0.7674 | 0.1354 | 0.6117 | 0.8819 | 0.3090 | 0.7578 | 0.9514 | 0.5417 | 0.8652 | 0.9722 | 0.7431 | 0.9329 | 0.9896 | 0.8715 |
| baseline:dense_bge_m3 | 0.7646 | 0.2702 | 0.6389 | 0.0000 | 0.3963 | 0.7847 | 0.0521 | 0.6641 | 0.9410 | 0.3507 | 0.8319 | 0.9792 | 0.6701 | 0.9278 | 0.9896 | 0.8646 | 0.9575 | 0.9965 | 0.9132 | 0.9844 | 1.0000 | 0.9688 |
| official:graphrag | 0.4904 | 0.1605 | 0.3819 | 0.0000 | 0.2308 | 0.4931 | 0.0139 | 0.4046 | 0.6389 | 0.1667 | 0.5095 | 0.6910 | 0.3264 | 0.5966 | 0.7500 | 0.4410 | 0.6415 | 0.7674 | 0.5104 | 0.6632 | 0.7778 | 0.5486 |
| official:hypergraphrag | 0.6924 | 0.2392 | 0.5625 | 0.0000 | 0.3451 | 0.6875 | 0.0451 | 0.5877 | 0.8785 | 0.2812 | 0.7576 | 0.9479 | 0.5625 | 0.8984 | 0.9931 | 0.7917 | 0.9472 | 0.9965 | 0.8924 | 0.9698 | 1.0000 | 0.9375 |
| official:lightrag | 0.6154 | 0.1999 | 0.4792 | 0.0000 | 0.2939 | 0.6042 | 0.0174 | 0.5006 | 0.7917 | 0.1944 | 0.6754 | 0.8993 | 0.4410 | 0.8318 | 0.9688 | 0.6840 | 0.9069 | 0.9931 | 0.8160 | 0.9378 | 0.9965 | 0.8750 |
| official:pathrag | 0.2485 | 0.0619 | 0.1493 | 0.0000 | 0.0816 | 0.1771 | 0.0000 | 0.1569 | 0.2882 | 0.0278 | 0.2976 | 0.4931 | 0.0972 | 0.5681 | 0.8021 | 0.3333 | 0.7634 | 0.9167 | 0.6042 | 0.9106 | 0.9688 | 0.8542 |
| qmsxe:graph:entity_relation | 0.8332 | 0.3130 | 0.7153 | 0.0000 | 0.5116 | 0.8993 | 0.1701 | 0.7490 | 0.9792 | 0.4931 | 0.8709 | 1.0000 | 0.7118 | 0.9605 | 1.0000 | 0.9028 | 0.9866 | 1.0000 | 0.9688 | 0.9964 | 1.0000 | 0.9896 |
| qmsxe:graph:reified_fact | 0.8332 | 0.3130 | 0.7153 | 0.0000 | 0.5116 | 0.8993 | 0.1701 | 0.7490 | 0.9792 | 0.4931 | 0.8709 | 1.0000 | 0.7118 | 0.9623 | 1.0000 | 0.9062 | 0.9883 | 1.0000 | 0.9722 | 0.9981 | 1.0000 | 0.9931 |
| qmsxe:hypergraph:evidence_hypergraph | 0.8299 | 0.3119 | 0.7118 | 0.0000 | 0.5104 | 0.8958 | 0.1701 | 0.7461 | 0.9757 | 0.4896 | 0.8681 | 0.9965 | 0.7083 | 0.9588 | 1.0000 | 0.8993 | 0.9837 | 1.0000 | 0.9618 | 0.9935 | 1.0000 | 0.9826 |

## Path, answer, citation and joint

| Method | Path EM | Path P | Path R | Path F1 | Answer EM | Answer P | Answer R | Answer F1 | Citation EM | Citation P | Citation R | Citation F1 | Joint EM | Joint P | Joint R | Joint F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline:bm25 | 0.1875 | 0.5087 | 0.5087 | 0.5087 | 0.4583 | 0.6404 | 0.6616 | 0.6249 | 0.2257 | 0.8482 | 0.5764 | 0.6633 | 0.1146 | 0.5999 | 0.4178 | 0.4579 |
| baseline:dense_bge_m3 | 0.5347 | 0.7535 | 0.7535 | 0.7535 | 0.4861 | 0.6719 | 0.7010 | 0.6575 | 0.3854 | 0.9421 | 0.6823 | 0.7678 | 0.1840 | 0.6440 | 0.4817 | 0.5137 |
| official:graphrag | 0.2812 | 0.4826 | 0.4740 | 0.4769 | 0.3750 | 0.5254 | 0.5558 | 0.5147 | 0.1632 | 0.6451 | 0.4288 | 0.4991 | 0.0729 | 0.4360 | 0.3072 | 0.3347 |
| official:hypergraphrag | 0.4028 | 0.6684 | 0.6684 | 0.6684 | 0.4653 | 0.6618 | 0.6999 | 0.6479 | 0.3785 | 0.9074 | 0.6684 | 0.7464 | 0.1667 | 0.6164 | 0.4789 | 0.4955 |
| official:lightrag | 0.3090 | 0.5920 | 0.5920 | 0.5920 | 0.4757 | 0.6427 | 0.6514 | 0.6210 | 0.2917 | 0.8773 | 0.6076 | 0.6966 | 0.1562 | 0.5868 | 0.4125 | 0.4530 |
| official:pathrag | 0.0417 | 0.2465 | 0.2465 | 0.2465 | 0.4028 | 0.5594 | 0.5782 | 0.5425 | 0.1076 | 0.6940 | 0.4271 | 0.5144 | 0.0694 | 0.4767 | 0.2982 | 0.3439 |
| qmsxe:graph:entity_relation | 0.4931 | 0.7240 | 0.7240 | 0.7240 | 0.4826 | 0.6635 | 0.6856 | 0.6471 | 0.3542 | 0.9207 | 0.6597 | 0.7449 | 0.1771 | 0.6345 | 0.4695 | 0.5038 |
| qmsxe:graph:reified_fact | 0.4931 | 0.7240 | 0.7240 | 0.7240 | 0.4826 | 0.6637 | 0.6868 | 0.6474 | 0.3576 | 0.9230 | 0.6649 | 0.7488 | 0.1771 | 0.6333 | 0.4712 | 0.5041 |
| qmsxe:hypergraph:evidence_hypergraph | 0.4861 | 0.7205 | 0.7205 | 0.7205 | 0.4792 | 0.6618 | 0.6845 | 0.6446 | 0.3368 | 0.9213 | 0.6562 | 0.7420 | 0.1562 | 0.6292 | 0.4621 | 0.4957 |

## Efficiency, usage and reliability

| Method | Success | Total s/query | Index s | Retrieval s | Generation s | LLM calls | Embedding calls | Reranker calls | Observed tokens | Prompt tokens | Completion tokens | Embedding tokens | API cost USD/query | Timing scope | Token scope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline:bm25 | 1.0000 | 0.9256 | N/A | 0.0002 | 0.9253 | 1.0000 | 0.0000 | 0.0000 | 845.1736 | 807.8854 | 37.2882 | 0.0000 | 0.000022 | query | measured |
| baseline:dense_bge_m3 | 1.0000 | 1.1273 | N/A | 0.2116 | 0.9157 | 1.0000 | 1.0000 | 0.0000 | 2210.5833 | 757.2708 | 38.8924 | 1414.4201 | 0.000023 | query | measured |
| official:graphrag | 1.0000 | 31.0369 | N/A | N/A | 0.9149 | 1.0000 | 0.0000 | 0.0000 | 669.1215 | 632.7188 | 36.4028 | N/A | 0.000021 | index_plus_query | generation_measured_index_unavailable |
| official:hypergraphrag | 1.0000 | 2.9380 | N/A | 1.9853 | 0.9527 | 2.0000 | 2.0000 | 0.0000 | 5184.6493 | 4911.4236 | 215.3333 | 57.8924 | 0.000662 | query_only_existing_index | generation_measured_index_unavailable |
| official:lightrag | 1.0000 | 18.9510 | N/A | N/A | 0.8931 | 1.0000 | 0.0000 | 0.0000 | 817.2326 | 778.4340 | 38.7986 | N/A | 0.000022 | index_plus_query | generation_measured_index_unavailable |
| official:pathrag | 0.9931 | 2.3403 | N/A | 1.4353 | 0.9113 | 1.9931 | 2.0000 | 0.0000 | 1422.6354 | 1314.2014 | 81.4861 | 26.9479 | 0.000100 | query_only_existing_index | generation_measured_index_unavailable, query_measured_index_unavailable |
| qmsxe:graph:entity_relation | 1.0000 | 1.5795 | N/A | N/A | 0.9458 | 1.0000 | 0.0000 | 0.0000 | 803.4097 | 764.9097 | 38.5000 | 0.0000 | 0.000023 | index_plus_query | measured |
| qmsxe:graph:reified_fact | 1.0000 | 1.4758 | N/A | N/A | 0.9737 | 1.0000 | 0.0000 | 0.0000 | 802.4965 | 763.6944 | 38.8021 | 0.0000 | 0.000023 | index_plus_query | measured |
| qmsxe:hypergraph:evidence_hypergraph | 1.0000 | 1.3928 | N/A | N/A | 0.8676 | 1.0000 | 0.0000 | 0.0000 | 806.4826 | 767.5174 | 38.9653 | 0.0000 | 0.000024 | index_plus_query | measured |

## Manifest

```json
{
  "protocol": "unified_native_benchmark_v1",
  "examples": 288,
  "candidate_documents_per_example": {
    "2": 1,
    "4": 1,
    "6": 1,
    "8": 1,
    "10": 284
  },
  "dataset_sha256": "81dfe1546ec9051c51f1c33ad40c3d5db32936ec8483b3a6037f3fa49a5a3f1a",
  "systems": [
    "baseline:bm25",
    "baseline:dense_bge_m3",
    "official:graphrag",
    "official:hypergraphrag",
    "official:lightrag",
    "official:pathrag",
    "qmsxe:graph:entity_relation",
    "qmsxe:graph:reified_fact",
    "qmsxe:hypergraph:evidence_hypergraph"
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
  "shared_answer_generator": "deepseek-chat temperature=0",
  "api_cost_scope": "DeepSeek calls only; SiliconFlow embedding price and unavailable historical index-build calls are excluded.",
  "citation_level": "document",
  "fact_fallback": "document-induced for systems without canonical text-unit IDs",
  "path_fallback": "top-2 induced document path",
  "limitations": [
    "Official index-build token usage is unavailable; fresh PathRAG/HyperGraphRAG query tokens and shared generation tokens are measured.",
    "API cost is partial where SiliconFlow embeddings or historical official index-build calls are involved; provider-reported tokens remain visible.",
    "This run is the HotpotQA distractor track (normally 10 candidate passages); passage cutoffs above the candidate count are reported for schema completeness but saturate."
  ]
}
```

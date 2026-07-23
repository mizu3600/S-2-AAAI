# Remote experiment summary

The remote engineering run used 50 HotpotQA validation examples and the deterministic keyless
encoder. These results validate execution and comparison wiring; they are not BGE-M3 publication
results.

## Graph/Hypergraph system comparison

| Mode | Recall@10 | Recall@20 | MRR | nDCG@10 | Bridge@20 |
|---|---:|---:|---:|---:|---:|
| Vanilla dense RAG | 0.2367 | 0.2533 | 0.2423 | 0.1826 | 0.2000 |
| QMSHE hypergraph | 0.3273 | 0.4307 | 0.2817 | 0.2216 | 0.1400 |
| Entity-relation graph | 0.3390 | 0.5707 | 0.2723 | 0.2236 | 0.0600 |
| Reified-fact graph | 0.3633 | 0.6267 | 0.2497 | 0.2344 | 0.1800 |

On this smoke configuration, the reified-fact ordinary graph has the strongest Recall@20. The
current untrained QMSHE frequency gate does not beat the stronger structural baselines, so the
results do not yet support a claim that the hypergraph model is superior.

## Selected embedding baselines

| Method | Recall@20 | MRR | Bridge@20 |
|---|---:|---:|---:|
| BM25 + Dense | 0.5567 | 0.3740 | 0.2000 |
| Node2Vec | 0.6767 | 0.3316 | 0.2400 |
| Laplacian Eigenmaps | 0.6540 | 0.3839 | 0.1800 |
| Semantic + PPR | 0.6750 | 0.4359 | 0.3400 |
| HypergraphConv | 0.6083 | 0.3562 | 0.1400 |
| QMSHE | 0.4307 | 0.2817 | 0.1400 |

Full tables, hop/query-type grouping and records are under `reports/remote/public_hotpot50/` and
`reports/remote/dual_hotpot50_v2/`.

## Query LoRA held-out result

| Metric | Frozen BGE-M3 | Query LoRA | Change |
|---|---:|---:|---:|
| Recall@20 | 0.7842 | 0.7950 | +0.0108 |
| MRR | 0.7176 | 0.7747 | +0.0571 |

The selected adapter is epoch 2 of 3. It was trained on 400 examples and evaluated on 100 held-out
examples against 20,795 frozen fact embeddings.

## Efficiency

| Run | Requests | Success | Throughput | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| Cold cache | 100 | 100% | 228.43 qps | 33.53 ms | 41.91 ms |
| Warm cache | 100 | 100% | 2047.88 qps | 1.04 ms | 28.67 ms |

External LightRAG/PathRAG results are not merged into these tables while the server's pre-existing
Unified-RAG-Evaluation jobs are still running. Mixing their partial outputs or different corpora
would violate the project's fairness rule.

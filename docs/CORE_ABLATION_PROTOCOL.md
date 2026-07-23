# Core algorithm ablation protocol

The ablation suite is separate from the production API and has two tracks so
that a downstream reranker cannot hide a weak spectral representation.

## Track A: independently trained Stage A representations

The fixed HotpotQA hash split contains 362 train, 75 validation and 63 test
examples when `limit=500`. Every cell is trained independently for seeds 13,
42 and 73. The BGE-M3 encoder, optimizer, epochs and validation selection rule
remain fixed.

| Variant | Change | Retrained |
|---|---|---:|
| full | Raw + Low + Mid + High with learned gates | yes, existing control |
| raw_only | Disable Low, Mid and High during training and evaluation | yes |
| no_low | Disable global/smooth band | yes |
| no_mid | Disable bridge/transition band | yes |
| no_high | Disable local-detail band | yes |
| fixed_gate | Replace learned band gate with uniform 0.25 weights | yes |
| no_role_gate | Use the shared hypergraph operator without query role mixing | yes |
| no_semantic_graph | Set evidence and role semantic-adjacency weights to zero | yes |
| no_bridge_loss | Set bridge supervision weight to zero | yes |
| no_hard_negatives | Replace all-node hardest negatives with seeded random negatives | yes |

This track ranks canonical Facts directly from Stage A spectral scores. It does
not use BM25, graph reranking or the neural reranker. It reports all Recall,
Hit/Accuracy and Complete cutoffs at 1/2/5/10/20/30/40, MRR, nDCG@10,
latency, mean±sample SD, paired bootstrap confidence intervals and paired
randomization tests.

## Track B: runtime retrieval components

Track B loads the same Full Stage A checkpoints and tuned reranker, then changes
only inference-time components:

- `no_graph_rerank`;
- `dense_only` (raw dense retrieval, same neural reranker);
- `no_bm25`;
- ordinary-graph `single`, `multi` and `hybrid` index strategies;
- Entity-Relation, Reified-Fact and Evidence-Hypergraph structures.

This track reports the same retrieval metrics, latency and paired effects. No
model is retrained for these switches. Before timing each condition, the local
query-embedding cache and cross-encoder score cache are cleared and CUDA is
synchronized. This makes every cell pay for its own query encoding and reranker
inference instead of rewarding conditions that happen to execute later.

## Reproduction

```bash
PYTHONPATH=src python scripts/train_core_ablation_matrix.py \
  --base-model /models/bge-m3 \
  --input-path data/benchmarks/hotpot_dev_distractor_v1.json \
  --output-root data/models/core_ablation \
  --limit 500 --epochs 3 --seeds 13,42,73

PYTHONPATH=src python scripts/evaluate_core_ablation_matrix.py \
  --base-model /models/bge-m3 \
  --checkpoint-root data/models/core_ablation \
  --full-checkpoint-root data/models/stage_ab \
  --output-dir reports/core_ablation_matrix \
  --limit 500 --seeds 13,42,73

PYTHONPATH=src python scripts/run_runtime_ablation_matrix.py \
  --embedding-model /models/bge-m3 \
  --reranker-model /models/bge-reranker-v2-m3 \
  --reranker-adapters \
  '13=data/models/bge_reranker_lora_seed_13,42=data/models/bge_reranker_lora_seed_42,73=data/models/bge_reranker_lora_seed_73' \
  --checkpoint-root data/models/stage_ab \
  --output-dir reports/runtime_ablation_matrix \
  --limit 500 --seeds 13,42,73
```

The reports must not merge the intrinsic and end-to-end tracks into one table:
they answer different questions. Runtime flags default to the original Full
pipeline, so the production behavior is unchanged.

## Stage A training-effect 2×2 control

The component ablations above do not constitute an untrained control because
`raw_only` is itself independently trained. The separate 2×2 experiment holds
the architecture and within-seed initialization fixed and compares:

- `trained_raw` versus `untrained_raw` for semantic alignment training;
- `trained_full` versus `untrained_full` for total Stage A training;
- `trained_full` versus `trained_raw` for the trained spectral-band increment;
- `untrained_full` versus `untrained_raw` as an initialization sanity check.

It also reports `dense_bge_identity`, which is not a 2×2 cell: it is the
practical no-Stage-A reference using cosine similarity in the original BGE-M3
space instead of an untrained random projection.

```bash
PYTHONPATH=src python scripts/evaluate_stage_a_training_effect.py \
  --base-model /models/bge-m3 \
  --raw-checkpoint-root data/models/core_ablation \
  --full-checkpoint-root data/models/stage_ab \
  --output-dir reports/stage_a_training_effect \
  --limit 500 --seeds 13,42,73
```

## Exact factorial attribution

One-at-a-time deletion does not separate source effects from interactions. The
factorial track keeps raw dense retrieval as the common reference and evaluates
every subset of these binary factors:

- ordinary graph: Entity→Fact expansion, BM25, spectral retrieval, structural
  graph reranking and the base neural reranker (`2^5` cells);
- hypergraph: BM25, spectral retrieval, structural graph reranking and the base
  neural reranker (`2^4` cells).

Every non-reranker context is additionally evaluated with the LoRA reranker, so
the adapter is measured as a conditional increment over the base cross-encoder.
The report uses exact Shapley values for main attribution and mean second
differences for pair interactions. BM25 and spectral source Shapley values plus
their interaction quantify multi-source fusion without inventing an arbitrary
"fusion off" concatenation order. All configurations have the same 60-Fact
candidate cap and 40-Fact output cap.

```bash
PYTHONPATH=src python scripts/run_factorial_attribution.py \
  --embedding-model /models/bge-m3 \
  --reranker-model /models/bge-reranker-v2-m3 \
  --reranker-adapters \
  '13=data/models/bge_reranker_lora_seed_13,42=data/models/bge_reranker_lora_seed_42,73=data/models/bge_reranker_lora_seed_73' \
  --checkpoint-root data/models/stage_ab \
  --output-dir reports/factorial_attribution \
  --limit 500 --seeds 13,42,73
```

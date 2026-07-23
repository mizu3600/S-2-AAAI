# Phase 3–5 implementation and verification

## Phase 3: public experiments

Implemented adapters normalize HotpotQA, 2WikiMultiHopQA, MuSiQue, QASPER and MetaQA into one
schema containing answer, passages, supporting sentences, bridge entities, path, hop count and query
type. Each benchmark context becomes a source-preserving evidence hypergraph. The experiment runner
holds corpus, encoder, top-k and evaluation budget fixed across BM25, dense, hybrid and QMSHE.

The repository includes a downloader for small rows from the hosted HotpotQA and QASPER mirrors,
an experiment report generator, DVC stages and MLflow logging. Full 2Wiki and MuSiQue files must be
provided from their official distributions because their public download/viewer endpoints are not
stable enough to silently substitute a third-party copy.

Stage A training freezes text embeddings and trains semantic projections, Chebyshev coefficients,
band gate and relation gate on a merged benchmark graph. Checkpoints record graph version, dataset,
roles and loss history.

### Measured smoke results

The initial five-example HotpotQA run used the deterministic test encoder, not the paid BGE service
and not a trained publication checkpoint. It is a pipeline verification, not a scientific claim.

| Method | Recall@10 | Recall@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.4667 | 0.6333 | 0.3200 | 0.3312 |
| Dense | 0.1000 | 0.1000 | 0.0400 | 0.0474 |
| BM25 + Dense | 0.4667 | 0.6333 | 0.2233 | 0.2724 |
| node2vec | 0.4000 | 0.4667 | 0.1554 | 0.2298 |
| Laplacian Eigenmaps | 0.3000 | 0.4067 | 0.3300 | 0.2613 |
| Semantic + LapPE | 0.3000 | 0.5333 | 0.2333 | 0.1915 |
| Semantic + PPR | 0.3000 | 0.5333 | 0.4182 | 0.3226 |
| GCN | 0.3067 | 0.4867 | 0.4650 | 0.3030 |
| GraphSAGE | 0.2000 | 0.5267 | 0.2776 | 0.1754 |
| HypergraphConv | 0.4400 | 0.5600 | 0.2700 | 0.3038 |
| QMSHE | 0.3733 | 0.4400 | 0.2905 | 0.2323 |

QASPER was also run on two real questions. Recall remained near zero, correctly showing that the
keyless lexical test encoder and automatic paragraph-to-sentence evidence mapping are insufficient
for a scientific result. Use BGE-M3, the full training split and manually verified evidence before
reporting PSC/QASPER performance.

The ablation registry contains every design variant. Runtime-safe variants (low-only, fixed gate,
no-high, no-raw and full) run from one checkpoint. Operator variants are marked `requires_rebuild`;
loss/negative variants are marked `requires_retrain`, preventing invalid post-hoc masking from being
reported as a trained ablation.

## Phase 4: PSC domain

The PSC module defines entity and predicate vocabularies, conservative measurement normalization,
qualifier conflict detection, batch corpus construction and quality auditing. Benchmark annotations
validate every document, chunk, hyperedge, bridge entity and path reference. Splits are made by
source paper to prevent leakage.

The local PSC demo produced one document, two chunks and one evidence fact with 100% provenance and
argument validity, zero duplicate chunks and zero validation errors. It is a format/quality smoke
test; a publication benchmark still requires the user's PSC PDFs and expert labels.

## Phase 5: incremental and scale

New nodes and hyperedges receive raw embeddings immediately and neighbor-weighted approximate
low/mid/high and role-band vectors. The update plan requests a full rebuild at 3% node growth or 5%
hyperedge growth. Query caches are keyed by graph, encoder, spectral and index versions.

Greedy community partitioning, bounded partition splitting and coarse-graph feature aggregation are
provided for million-node routing. The in-memory exact index remains the research gold standard;
Qdrant provides named raw/low/mid/high/full production vectors.

On the synthetic graph, 100 requests at concurrency 8 achieved 100% success. Cold-cache P95 was
approximately 7.5 ms and warm-cache P95 approximately 5.4 ms, excluding external model network time.
These numbers are not representative of a million-node PSC corpus.

Runtime metrics expose query count, cache hit rate, no-evidence rate, citation count, P50/P95,
average band weights and average relation weights at `GET /v1/metrics`.

## Reproducibility boundaries

- Never compare methods with different extraction, encoder, reranker, generator or context budgets.
- Never cite a semantic hyperedge as evidence.
- Never mix incompatible graph/encoder/spectral versions in one index.
- Do not describe smoke-test results as publication results.
- Full system baselines run in isolated environments and must return the normalized ranked-ID
  contract documented in `SYSTEM_BASELINES.md`.

# External baseline protocol

The pinned repository URLs and commits are stored in
`official_baselines.lock.json`. Install the sources into an isolated directory:

```bash
uv run python scripts/install_official_baselines.py --baseline all
```

The strict external protocol currently covers HotpotQA passage retrieval only.
Export the immutable canonical bundle first:

```bash
uv run python scripts/export_official_baseline_bundle.py \
  --input-path data/benchmarks/hotpot_dev_distractor_v1.json
```

The bundle uses the benchmark `passage_id` as each native `document_id` and writes
a SHA-256 manifest. Each upstream project keeps its own environment, configuration
and native runner.
This repository does not replace an upstream ranking algorithm with a lexical or
PPR proxy. Instead, it imports the native result file through a framework-specific
adapter:

```bash
uv run python scripts/import_external_baseline_results.py \
  --baseline graphrag \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa.json \
  --result-path /path/to/graphrag-results.jsonl
```

Every native result record must include `example_id` and a ranked result field.
The adapters recognize framework-specific ranking fields, then map passage ID,

| Baseline | Native ranking fields |
|---|---|
| GraphRAG | `community_ranking`, `chunk_ranking` |
| LightRAG | `chunk_ranking`, `entity_ranking` |
| PathRAG | `path_context_ranking` |
| HyperGraphRAG | `hyperedge_ranking` |
| HippoRAG2 | `ppr_ranking` |
| Cog-RAG | `dual_hypergraph_ranking` |
| HGRAG | `diffusion_ranking` |
| Hyper-RAG | `hypergraph_ranking` |

Native graph IDs must be accompanied by a per-example `source_id_map`:

```json
{
  "example_id": "hotpot-example-id",
  "community_ranking": [{"id": "community-7"}],
  "source_id_map": {"community-7": ["canonical_passage_id"]},
  "indexing_seconds": 1.2,
  "retrieval_seconds": 0.08
}
```

Missing rows, failed rows and unmapped native IDs are retained in the denominator.
The report records `mapping_coverage`, missing/failed counts and `N/A` for metrics
whose ranking level is unavailable. A passage ranking is never expanded into all
sentences to manufacture a fact ranking.

Optional `answer`, `citations`, `indexing_seconds`, `retrieval_seconds`,
`total_seconds`, `status` and `error` fields are preserved. Native answer metrics
are `N/A` unless the trace declares
`generation_protocol=hotpotqa_shared_generation_v1`. Retrieval evidence citations
and citations actually emitted in the answer are separate metrics.

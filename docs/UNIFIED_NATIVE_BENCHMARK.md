# Unified native benchmark

This is an independent evaluation layer. It does not replace or modify the
retrieval, graph construction, training, or generation implementation of any
QMSxE or third-party system.

## Fairness contract

Each method runs its native indexing/query path and emits a `StandardTrace`.
The adapter boundary translates native document/text-unit IDs to the fixed
HotpotQA canonical IDs. The evaluator never changes a method's retrieval score.

All systems share:

- the same 288-example frozen dataset and SHA-256 manifest;
- the same candidate documents and gold supporting facts;
- the same evaluation cutoffs: 1, 2, 5, 10, 20, 30, and 40;
- the same DeepSeek answer generator, prompt, temperature, and citation parser;
- the same metric implementation and report generator.

The trace explicitly records whether a ranking is native, mapped from an
official text unit, or induced from a document ranking. This prevents an
estimated Fact score from being mistaken for a native Fact score.

## Metrics

- Document and Fact: Recall@K, Hit/Accuracy@K, Complete@K, and MRR.
- Path: EM, precision, recall, and F1. When a framework does not expose a stable
  path ID, the report labels the top-two-document path as induced.
- Answer: normalized exact match, token precision, recall, and F1.
- Citation: document citation EM, precision, recall, and F1.
- Joint: Hotpot-style answer/citation joint EM, precision, recall, and F1.
- Operations: success rate, retrieval/generation/total wall time, LLM,
  embedding and reranker calls, provider-reported token counts, and estimated
  API cost.

`Hit/Accuracy@K` is retrieval top-K accuracy: one if at least one gold item is
present, otherwise zero. It is not a binary-classification accuracy. Answer
accuracy is reported as answer EM and F1.

## Commands

Fresh native PathRAG and HyperGraphRAG captures reuse their official existing
indexes but call the original query functions:

```bash
PYTHONPATH=src third_party/.venvs/pathhyper/bin/python scripts/requery_path_hyper_internal.py \
  --framework pathrag \
  --official-repo third_party/official_baselines/PathRAG \
  --input data/benchmarks/hotpotqa_official_baselines_288.json \
  --work-parent data/index/official_baselines \
  --output reports/unified_native_benchmark/pathrag_internal.jsonl

PYTHONPATH=src third_party/.venvs/pathhyper/bin/python scripts/requery_path_hyper_internal.py \
  --framework hypergraphrag \
  --official-repo third_party/official_baselines/HyperGraphRAG \
  --input data/benchmarks/hotpotqa_official_baselines_288.json \
  --work-parent data/index/official_baselines \
  --output reports/unified_native_benchmark/hypergraphrag_internal.jsonl
```

Build the consolidated benchmark (credentials are read from environment
variables and never written to the report):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_unified_native_benchmark.py
```

Outputs are written to `reports/unified_native_benchmark`: `manifest.json`,
`summary.json`, `records.json`, `traces.json`, and `report.md`. The JSON trace is
the audit source; the Markdown file is only a presentation layer.

## Scope notes

Official GraphRAG and LightRAG historical records expose their structured
document ranking but not complete index-build token accounting. Fresh PathRAG
and HyperGraphRAG runs measure query tokens exactly while reusing already built
indexes. Local QMSxE retrieval has zero API-model tokens; its shared answer
generation is measured separately. Missing values stay `N/A` and are never
silently converted to zero.

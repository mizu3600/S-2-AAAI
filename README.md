# QMSHE-RAG

仓库只保留一条可执行链：

```text
PDF/TXT/MD
  -> 分块
  -> 实体与证据事实抽取
  -> Reified-Fact 普通图
  -> raw/low/mid/high 谱表示
  -> Full + Multi-band + Dense + BM25 混合检索
  -> 图重排
  -> 证据校验与引用上下文
  -> DeepSeek 生成（无密钥时使用证据抽取式回答）
```

提取阶段不是用 LLM 抽实体：实体由 PSC 规则和 canonicalization 生成；LLM
只接收文本和已知实体列表，用于抽取证据事实。没有 DeepSeek key 时，事实也会
回退到规则抽取器。

## 安装

```bash
cp .env.example .env
uv sync --extra dev
```

## CLI

```bash
uv run qmshe ingest data/raw/document.pdf
uv run qmshe build data/processed/corpus.json
uv run qmshe query "How does PEAI improve Voc?"
```

`build` 用于校验语料并构建一次内存索引；`query` 会从指定 Corpus 构建同一条
Reified-Fact hybrid pipeline 后执行检索和生成。

## API

```bash
uv run uvicorn qmshe.api.main:app --reload
```

接口：

```text
POST /v1/documents/ingest
POST /v1/index/build
POST /v1/query
GET  /v1/metrics
GET  /health
```

API 索引保存在当前进程内。启动后先调用 `/v1/index/build`，再调用 `/v1/query`。

## 测试

```bash
uv run pytest
```

## 多数据集跑分

已恢复数据集适配和统一评测层。支持的 adapter 包括：

```text
hotpotqa
2wikimultihopqa
musique
qasper
metaqa
ultradomain
mix
```

单个数据集：

```bash
uv run python scripts/run_public_experiment.py \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa_sample.json \
  --limit 20
```

多随机种子：

```bash
uv run python scripts/run_multi_seed_benchmark.py \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa.json \
  --seeds 13,42,73
```

批量配置和运行：

```bash
uv run python scripts/configure_multi_seed_benchmarks.py \
  --input-dir data/benchmarks/raw \
  --datasets hotpotqa \
  --sample-size 1000

uv run python scripts/run_all_configured_benchmarks.py
uv run python scripts/summarize_benchmark_runs.py
```

每个 suite 会生成 `records.json`、`summary.json`、`manifest.json` 和 `report.md`。
当前严格评测协议只覆盖 HotpotQA。指标包括 passage/fact Recall@K、Precision@K、
Hit@K、Complete@K、MRR、nDCG、Answer EM/F1、retrieval evidence、
generated citation、Hotpot Joint F1，以及分阶段耗时和 ranking provenance。
多 seed 使用固定测试题；统计先按 question 聚合 seed，再对所有系统对和主指标计算
paired randomization p-value、paired Cohen's d 和 Holm 校正后的 p-value。

## Baselines

内部 baseline 使用当前语料和统一指标：

```text
bm25
dense
bm25_dense_rrf
node2vec
laplacian_eigenmaps
semantic_lap_pe
semantic_ppr
gcn
graphsage
hypergraph_conv
reified_fact_hybrid
```

运行全部内部 baseline：

```bash
uv run python scripts/run_public_experiment.py \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa.json \
  --methods all
```

GraphRAG、LightRAG、PathRAG、HyperGraphRAG、HippoRAG2、Cog-RAG、HGRAG、
Hyper-RAG 都有独立结果 adapter。外部原生比较只使用 HotpotQA passage-level 榜；
没有 canonical sentence/fact ranking 时，fact、fact citation 和 Joint 指标输出 `N/A`。
先导出统一 passage bundle、安装锁定源码，再用原生 runner 生成 JSON/JSONL：

```bash
uv run python scripts/export_official_baseline_bundle.py \
  --input-path data/benchmarks/hotpot_dev_distractor_v1.json

uv run python scripts/install_official_baselines.py --baseline all

uv run python scripts/import_external_baseline_results.py \
  --baseline graphrag \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa.json \
  --result-path /path/to/graphrag-results.jsonl
```

adapter 会把 framework-specific 的 passage/title/index 排名和 `source_id_map` 映射回
canonical passage ID。缺题、失败和无法映射的题仍保留在评测分母中；报告包含 mapping
coverage。外部 native answer 默认不计算 Answer/Joint 指标，只有显式声明
`generation_protocol=hotpotqa_shared_generation_v1` 才能进入共同生成榜。

内部榜和外部榜不能直接混表：

- `controlled_hotpotqa_internal_v2`：同一 sentence-fact 语料、embedding、候选预算和生成器，
  用于内部检索方法消融。
- `hotpotqa_native_external_passage_v2`：同一原始 passage bundle 和 canonical passage ID，
  用于外部原生系统的 passage retrieval 比较。

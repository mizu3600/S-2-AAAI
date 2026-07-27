# S²-RAG

仓库只保留一条可执行链：

```text
PDF/TXT/MD
  -> 分块
  -> 实体与证据事实抽取
  -> 本地 BGE-M3 向量编码
  -> Reified-Fact 普通图
  -> raw/low/mid/high 谱表示
  -> Full + Multi-band + Dense + BM25 混合检索
  -> 图重排 + 本地 BGE-reranker-v2-m3
  -> 证据校验与引用上下文
  -> DeepSeek 生成
```

实体和证据事实均由 DeepSeek 从原文中抽取。抽取协议
`shared_deepseek_openie_nary_v2` 借鉴 HippoRAG2 的 two-pass、NER-conditioned
开放域 OpenIE，但保留 S²-RAG 的可审计 evidence 和 n 元关系表示：

- 实体阶段不使用固定领域 ontology，抽取能参与事实的人物、组织、地点、事件、方法、
  系统、概念、条件、日期、数量和测量值，并由 LLM 生成开放类型。
- 事实阶段使用实体列表作为受控词表，但 predicate 和语义角色保持开放；复合命题拆成
  原子事实，真正的多参与者事件保留为带角色的 n 元事实，而不是强制拆成三元组。
- 代词可以由 LLM 消解到当前 chunk 中明确出现的实体。每个事实都必须带原文中的精确
  evidence quote，代码还会校验实体 surface、entity ID 和 evidence。

字符串匹配只用于验证 LLM 输出是否落在原文中，不承担规则抽取。系统没有规则
fallback；未配置 `DEEPSEEK_API_KEY`、响应无效或远程调用失败时，客户端采用等待间隔
封顶的指数退避和有限重试；永久错误立即失败，避免评测进程无限挂起。

Embedding 和 reranking 不调用任何云端模型服务。S²-RAG 只从本地目录加载
`BAAI/bge-m3` 与 `BAAI/bge-reranker-v2-m3`，且启用 `local_files_only`，模型缺失时
直接报错，不进行网络下载或 deterministic fallback。运行代码不依赖 `peft`，也不加载
任何 LoRA adapter；如果模型目录包含 `adapter_config.json`、`adapter_model.bin` 或
`adapter_model.safetensors`，启动时会直接拒绝该目录。

## 安装

```bash
cp .env.example .env
uv sync --extra dev
```

默认模型目录是 `models/bge-m3` 和 `models/bge-reranker-v2-m3`。服务器部署时可设置：

```bash
BGE_EMBEDDING_MODEL_PATH=/data/users/liruizhe/models/bge-m3
BGE_RERANKER_MODEL_PATH=/data/users/liruizhe/models/bge-reranker-v2-m3
BGE_DEVICE=cuda
```

## CLI

```bash
uv run s2rag ingest data/raw/document.pdf
uv run s2rag build data/processed/corpus.json
uv run s2rag query "How does request caching improve response latency?"
```

`build` 用于校验语料并构建一次内存索引；`query` 会从指定 Corpus 构建同一条
Reified-Fact hybrid pipeline 后执行检索和生成。

## API

```bash
uv run uvicorn s2rag.api.main:app --reload
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

小规模实体/事实抽取协议 A/B：

```bash
uv run python scripts/compare_extraction_protocols.py
```

脚本固定使用相同模型、temperature、跨领域文本和落地校验器，对比旧版
`shared_deepseek_entity_fact_v1` 与新版 `shared_deepseek_openie_nary_v2`，报告实体/事实
覆盖、n 元事实比例、无效参数、evidence 落地率、token 和耗时。结果写入
`outputs/extraction_ab/<run_id>/`。

## 多数据集跑分

固定正式评测集为以下四个，每个数据集使用 seed `42` 确定性抽取 1000 题：

```text
hotpotqa          data/benchmarks/hotpotqa_1000.jsonl
musique           data/benchmarks/musique_1000.jsonl
2wikimultihopqa   data/benchmarks/2wikimultihopqa_1000.jsonl
ultradomain       data/benchmarks/ultradomain_1000.jsonl
```

HotpotQA 和 2WikiMultiHopQA 只从至少包含两个 gold supporting passage 的题目中
抽样；MuSiQue 只抽取 `answerable=true` 且 decomposition 至少两步的题目。
UltraDomain 没有 hop 或 gold evidence 标注，因此不声称其为多跳题；它按 19 个官方
领域均衡分层抽样，每个领域 52 或 53 题，sentence evidence 指标记为 `N/A`。
样本清单、来源和 SHA-256 写入 `data/benchmarks/manifest.json`。

生成这四份固定样本：

```bash
uv run python scripts/prepare_four_benchmarks.py \
  --hotpot-source /path/to/hotpot_dev_distractor_v1.json \
  --musique-source /path/to/musique_ans_v1.0_dev.jsonl \
  --two-wiki-source /path/to/2wikimultihopqa/dev.json \
  --two-wiki-alias-source /path/to/2wikimultihopqa/id_aliases.json \
  --ultradomain-source-dir /path/to/ultradomain
```

这里的 HotpotQA 使用 `distractor` 协议，每道题只检索该题自带的候选 passages。
只有 `fullwiki` 协议才应使用数据集级全局语料；两种协议的结果不能混在同一张榜中。
`mix` adapter 仅用于兼容组合输入，不属于上述四个正式评测集。

### 数据集官方评测口径

最终报告把 `Official Dataset Metrics` 和 `Unified Diagnostics` 分开，不能用后者替代
数据集 leaderboard 口径：

| 数据集 | 官方 evaluator | 正式显示指标 | 数据/语料协议 |
|---|---|---|---|
| HotpotQA | `hotpot_evaluate_v1.py` | Answer、Supporting Fact、Joint EM/F1 | validation `distractor`；每题自己的 10 个候选 passages |
| MuSiQue | `evaluate_v1.0.py` | alias-aware Answer EM/F1、paragraph Support F1 | answerable dev；每题自己的候选 paragraphs |
| 2WikiMultiHopQA | `2wikimultihop_evaluate_v1.1.py` | alias-aware Answer、sentence Support EM/F1 | April-7 IDs dev；每题自己的候选 passages |
| UltraDomain | 无维护中的官方 scorer | 只显示明确标注为 unified 的 Answer EM/token F1 | 每条样本自己的长文档；无 gold evidence |

2Wiki v1.1 的完整官方分数还包含 relation-triple Evidence 和基于
Answer × Support × Evidence 的 Joint。当前 S²-RAG/外部 baseline 没有统一、可审计的
官方三元组输出，因此这两项显示 `N/A`，不会用内部 reified fact 指标代替。
外部 baseline 如果只能返回 passage ranking，HotpotQA/2Wiki 的 sentence Support
指标同样显示 `N/A`；MuSiQue 的 paragraph Support 可以正常计算。

三个上游 evaluator 按 commit 锁定在
`third_party/official_evaluators.lock.json`。安装并核对项目公式与上游脚本：

```bash
uv run python scripts/install_official_evaluators.py --evaluator all
uv run python scripts/verify_official_evaluator_parity.py
```

单个数据集：

```bash
uv run python scripts/run_public_experiment.py \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa_1000.jsonl
```

单随机种子（固定为 42）：

```bash
uv run python scripts/run_multi_seed_benchmark.py \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa_1000.jsonl \
  --seeds 42
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

每个 suite 只生成一份 `unified_shared_models_v1` 结果，不再拆 first-stage 和
shared-rerank 两张榜。结果包含端到端 `summary.json`、诊断用
`successful_only_summary.json`、
`audit.json`、`manifest.json` 和 `report.md`。
生产 benchmark 对每题调用同一个 DeepSeek V4 Flash 完成 entity extraction、
fact extraction 和 answer generation；所有内部方法使用同一个本地 BGE-M3 encoder、
相同数量的 canonical fact candidates 和同一个本地 BGE reranker。确定性 sentence
fact 构造器只供单元测试和 evaluator-owned passage ID 对齐使用，不是运行时 fallback。
当前严格评测协议只覆盖 HotpotQA。指标包括 passage/fact Recall@K、Precision@K、
Hit@K、Complete@K、MRR、nDCG、Answer EM/F1、retrieval evidence、
generated citation、Hotpot Joint F1，以及分阶段耗时和 ranking provenance。
评测只使用 seed 42；对所有系统对和主指标计算 paired randomization p-value、
paired Cohen's d 和 Holm 校正后的 p-value。

## Baselines

内部 baseline 使用当前语料和统一指标：

```text
bm25
dense
reified_fact_hybrid
```

其中 `reified_fact_hybrid` 是 S²-RAG 主方法。运行全部内部 baseline：

```bash
uv run python scripts/run_public_experiment.py \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa_1000.jsonl \
  --methods all
```

GraphRAG、LightRAG、PathRAG、HyperGraphRAG、HippoRAG2、Cog-RAG、HGRAG、
Hyper-RAG 都有独立结果 adapter。四个固定数据集都可以导出相同结构的
passage-level bundle 并导入官方框架结果；没有 canonical sentence/fact ranking 时，
fact、citation 和 Joint 指标输出 `N/A`。UltraDomain 没有 gold evidence，因而它的
passage/fact evidence 指标也输出 `N/A`。
先导出统一 passage bundle、安装锁定源码，再用原生 runner 生成 JSON/JSONL：

```bash
uv run python scripts/export_official_baseline_bundle.py \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa_1000.jsonl

uv run python scripts/install_official_baselines.py --baseline all

uv run python scripts/import_external_baseline_results.py \
  --baseline graphrag \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa_1000.jsonl \
  --result-path /path/to/graphrag-results.jsonl
```

adapter 会把 framework-specific 的 passage/title/index 排名和 `source_id_map` 映射回
canonical passage ID。缺题、失败和无法映射的题仍保留在评测分母中；报告包含 mapping
coverage。capability 由 adapter 固定，不能由结果提交者修改。只有
`generation_protocol=unified_concise_deepseek_v1` 且 model、完整 prompt hash、temperature、
max tokens、重试策略和 context budget 全部匹配时，整个系统才能进入共同生成榜。
生产导入还会校验 evaluator 生成的 `shared_model_trace`：BGE embedding/reranker
权重 hash 与配置、DeepSeek entity/fact extraction 模型和两个 prompt hash 必须一致；
缺失或不匹配时 retrieval 指标记为 protocol mismatch，不进入统一榜。

这八个框架使用 `third_party/official_baselines.lock.json` 中列出的官方 Git 仓库和
固定 commit，通过 `scripts/install_official_baselines.py` clone 到
`third_party/official_baselines/`。S²-RAG 不修改这些官方 worktree，也不以自写实现
替代官方算法；运行前可用 `git -C <repo> status --short` 检查 worktree 必须为空。

所有结果使用同一个 `unified_shared_models_v1` 报告协议，但指标按 capability
显示：内部方法可报告 fact 与 passage 指标；外部系统只有 canonical passage 输出时，
fact retrieval、fact citation 和 Joint 为 `N/A`，不能把 passage 自动展开成 sentence
fact。支持某项能力但单题缺失或失败时计 0；系统级不支持才记 `N/A`。外部结果必须提供
相同 DeepSeek generation manifest 才能进入 Answer/Joint 指标。

S²-RAG 图频带编码为 training-free：raw/low/mid/high 由传播矩阵解析计算，query gate
使用确定性的相似度 softmax，不包含随机 Linear、MLP 或 checkpoint。

## Accelerated benchmark execution

实体和事实抽取按 passage 批处理，DeepSeek 响应及 BGE embedding 使用内容寻址缓存。
默认题级并发为 8、题内 passage 并发为 8、DeepSeek 全局并发为 32；相同缓存键的并发
请求会合并为一次上游调用。可通过 `BENCHMARK_EXAMPLE_WORKERS`、
`EXTRACTION_WORKERS` 和 `DEEPSEEK_MAX_CONCURRENCY` 调整。
每题只构建一次 corpus、embedding 和 reified graph；`all_ablations` 在这份共享
artifact 上切换检索通道，默认只为三个正式方法生成答案：

```bash
uv run python scripts/run_public_experiment.py \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa_1000.jsonl \
  --methods all_ablations \
  --generate-methods bm25,dense,reified_fact_hybrid
```

多个官方框架并行运行时，先启动单进程 DeepSeek gateway。它为所有进程提供共享并发
上限、持久连接、有限重试和按框架隔离的响应缓存：

```bash
UPSTREAM_DEEPSEEK_API_KEY=... \
  uv run python scripts/deepseek_gateway.py \
  --max-concurrency 32 \
  --requests-per-minute 0

uv run python scripts/run_parallel_frameworks.py \
  --commands configs/hotpotqa_framework_commands.json \
  --max-parallel 4
```

`requests-per-minute=0` 表示只限制并发；获得供应商的明确 RPM 配额后应填写真实数值。
gateway 必须只启动一个 worker，才能在所有框架之间实施同一个全局限制。

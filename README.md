# QMSxE-RAG

查询自适应多频带谱—语义普通图/超图检索增强生成系统。生产默认只启用
QMSGE Reified-Fact；QMSHE Hypergraph 与 Entity-Relation 的实现、模型和实验入口
完整保留，但运行时默认关闭，可通过环境变量显式恢复。项目已覆盖：
文档摄取、实体与 n 元事实抽取、证据/语义超图、Zhou Laplacian、Chebyshev 多频带滤波、
查询门控、混合检索、证据约束、引用生成、FastAPI、baseline 与自动化测试。

## 快速开始

```bash
cd /Users/jesse/Desktop/qmshe-rag
cp .env.example .env                 # 在本机填写两个 API key
uv sync --extra dev
uv run qmshe demo
uv run pytest
uv run uvicorn qmshe.api.main:app --reload
```

API 文档：<http://127.0.0.1:8000/docs>。

完整基础设施：

```bash
docker compose up -d
docker compose ps
uv sync --extra dev --extra infra
uv run python scripts/verify_infrastructure.py
```

默认模型：

- embedding：SiliconFlow `BAAI/bge-m3`
- reranker：SiliconFlow `BAAI/bge-reranker-v2-m3`
- 抽取与生成：DeepSeek `deepseek-v4-flash`

所有密钥仅从环境变量读取。无密钥时，synthetic demo 和测试使用确定性本地编码器与抽取器。

## 主要命令

```bash
uv run qmshe ingest data/raw/paper.pdf
uv run qmshe build data/processed/corpus.json
uv run qmshe query "How does PEAI improve Voc?"
uv run qmshe query "How does PEAI improve Voc?" --mode graph --graph-profile reified_fact
uv run qmshe demo
uv run qmshe evaluate
```

更完整的架构、数据格式、实验与验收说明见 `docs/IMPLEMENTATION.md`。
普通图/超图的隔离边界、API、训练和公平对比见 `docs/DUAL_MODE.md`。

默认 API/CLI 路由为 `mode=graph, graph_profile=reified_fact`。如需复现实验，可在
`.env` 中设置 `QMSHE_ENABLE_HYPERGRAPH=true` 或
`QMSHE_ENABLE_ENTITY_RELATION=true`；这不会删除或重建对应代码和模型。

## Phase 3–5

```bash
# 下载官方镜像的小样本并运行公共数据实验
uv run python scripts/download_benchmark_samples.py --rows 5
uv run python scripts/run_public_experiment.py \
  --dataset hotpotqa --input-path data/benchmarks/hotpotqa_sample.json --limit 5

# Stage A：冻结文本 encoder，训练谱滤波/投影/query gate/relation gate
uv run python scripts/train_public_benchmark.py \
  --dataset hotpotqa --input-path data/benchmarks/hotpotqa_sample.json --limit 5 --epochs 10

# PSC 领域语料
uv run python scripts/build_psc_corpus.py /path/to/psc/papers \
  --output data/processed/psc_corpus.json

# 消融和压力测试
uv run python scripts/run_ablations.py
uv run python scripts/run_load_test.py --requests 100 --concurrency 8

# 普通图 Stage A 与双模式公平对比
uv run python scripts/train_graph_mode.py --profile reified_fact --limit 5 --epochs 10
uv run python scripts/run_dual_mode_experiment.py --limit 5

# BGE-M3 query-only LoRA（需要本地模型与 lora extra）
uv sync --extra lora
uv run python scripts/train_query_lora.py \
  --base-model /path/to/bge-m3 --limit 500 --epochs 3 --learning-rate 0.00002

# 一条命令生成数学、baseline、分组、消融、效率、门控和案例报告
uv run python scripts/reproduce_delivery.py --limit 5
```

Phase 3–5 的实现、真实小样本结果和限制见 `docs/PHASES_3_5.md`。
远程部署与 LoRA 实测结果见 `docs/REMOTE_DEPLOYMENT.md`。
多随机种子及 embedding/reranker 微调消融协议见 `docs/FINETUNE_ABLATION.md`。

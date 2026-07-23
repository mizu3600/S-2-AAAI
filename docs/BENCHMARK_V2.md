# Benchmark V2：平行、公平且分层的评测链路

`src/qmshe/benchmark_v2` 是完全平行的实验链路。它不修改
`QMSHEPipeline`、`QMSGEGraphPipeline` 或旧实验脚本，目的是隔离评测协议变化，避免新版结果污染历史结果。

## 六项改进如何实现

1. **统一转换和预算**：三种拓扑从相同 Fact 集合开始；Entity 命中必须先转成 Fact；每种方法最多 60 个 Fact 进入同一重排器，并统一输出 40 个 Fact。
2. **四层指标**：同时输出 Fact、passage/path、answer、citation 指标，以及 answer×citation 的 joint 指标。Fact 和 passage 均含 Recall/Hit/Complete@1/2/5/10/20/30/40 与 MRR。
3. **标准数据和规模曲线**：主结果使用官方 HotpotQA distractor dev 全部 7,405 题和原生 10 passages；另以确定性、无标签采样构造 100/1000-passage 压力组。训练数据与 dev 数据由不同文件加载，评测代码不会训练模型。
4. **内部排名**：QMSxE 直接导出 Fact 排名；GraphRAG 读取 `context_data.sources`；LightRAG 读取 `aquery_data.chunks`；PathRAG/HyperGraphRAG 包装官方 text-unit 选择函数并捕获其返回对象。禁止从 LLM 最终文本或渲染后的 Sources CSV 反推排名。
5. **结构化事实**：V2 事实包含 subject、predicate、object、命名角色、qualifier 和规范化 entity ID。当前可复现实验使用 `rule-roles-v2`；它比旧版统一 `states/source/mention` 信息更完整，但仍应被视为无 LLM 的保守下界。
6. **n-ary 压力测试**：500 题由五元事实生成，负例只交换 material/process/condition/temperature/result 中一个角色，专门检查错误二元拼接。主指标应看 Fact Hit@1/MRR，不能只看 Recall@10。

## 运行

```bash
python scripts/download_benchmark_v2_data.py --split dev
python scripts/run_benchmark_v2.py \
  --candidate-counts 10,100,1000 \
  --limits 0,1000,200 \
  --nary-size 500 \
  --output-dir reports/benchmark_v2
```

`0,1000,200` 表示：10-passage 使用全 dev；100-passage 使用 1,000 题；1000-passage 使用 200 题。报告和 manifest 会保留各组真实样本数，不允许横向伪装成同样规模。

已有 Stage A/B/C checkpoint 的公平重排实验：

```bash
python scripts/run_trained_parity_v2.py \
  --embedding-model /models/bge-m3 \
  --reranker-model /models/bge-reranker-v2-m3 \
  --reranker-adapter data/models/bge_reranker_lora_seed_42 \
  --limit 288
```

## 结果解释边界

- `controlled_*` 是用于隔离拓扑和评测偏差的受控检索器，不是已训练 Stage A/B/C 的成绩。
- `trained_parity:*` 才是现有已训练模型在统一 Entity→Fact 和重排预算后的成绩。它沿用旧 checkpoint 的 `source/mention` 角色，因为旧 relation gate 无法加载新增角色维度。
- 要把新 structured-role 图接入已训练主结果，必须在标准 train split 上重新训练三个 Stage A checkpoint；直接复用旧 relation gate 会产生维度和语义错配。
- 规则答案抽取的 EM/F1 是可运行下界，不代表 DeepSeek 生成器上限；正式答案生成实验必须固定 prompt、模型版本、temperature 和失败重试策略。

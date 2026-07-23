# 统一原生 Benchmark 结论

本轮使用固定的 288 题 HotpotQA distractor 子集、同一候选文档、同一
DeepSeek 答案生成器和同一指标实现。详细的所有 K 值与运行指标见
`report.md`，原始逐题审计数据见被 Git 忽略的 `records.json` 和
`traces.json`。

## 核心结果

| 方法 | Passage MRR | Passage R@5 | Fact MRR | Fact R@5 | Fact R@10 | Answer EM | Joint F1 | 成功率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.8077 | 0.7500 | 0.5894 | 0.4639 | 0.6117 | 0.4583 | 0.4579 | 1.0000 |
| Dense BGE-M3 | **0.9521** | **0.9201** | 0.7639 | 0.6612 | 0.8319 | **0.4861** | **0.5134** | 1.0000 |
| GraphRAG | 0.6554 | 0.5799 | 0.4904 | 0.4046 | 0.5095 | 0.3750 | 0.3347 | 1.0000 |
| HyperGraphRAG | 0.9011 | 0.8767 | 0.6924 | 0.5877 | 0.7576 | 0.4653 | 0.4955 | 1.0000 |
| LightRAG | 0.8421 | 0.8264 | 0.6154 | 0.5006 | 0.6754 | 0.4757 | 0.4530 | 1.0000 |
| PathRAG | 0.4675 | 0.5938 | 0.2485 | 0.1569 | 0.2976 | 0.4028 | 0.3439 | 0.9931 |
| QMSxE Entity-Relation | 0.9272 | 0.9028 | **0.8332** | **0.7490** | **0.8709** | 0.4826 | 0.5038 | 1.0000 |
| QMSxE Reified-Fact | 0.9272 | 0.9028 | **0.8332** | **0.7490** | **0.8709** | 0.4826 | 0.5041 | 1.0000 |
| QMSxE Hypergraph | 0.9272 | 0.8993 | 0.8299 | 0.7461 | 0.8681 | 0.4792 | 0.4957 | 1.0000 |

## 解读

1. **QMSxE 的优势主要出现在 Fact 层。** Entity-Relation 和 Reified-Fact
   的 Fact MRR 为 0.8332，明显高于 Dense 的 0.7639 和官方
   HyperGraphRAG 的 0.6924；Fact R@10 同样领先。这说明结构链路更擅长把
   支持句排到前面，而不是只把正确 passage 放进候选集。
2. **Dense 仍是 passage 和最终 joint 的最强方法。** Dense 的 Passage MRR
   为 0.9521，Joint F1 为 0.5134。QMSxE Reified-Fact 的 Joint F1 为
   0.5041，差距已很小，但当前结构召回优势还没有完全转化成答案增益。
3. **QMSxE 三种结构在当前数据上非常接近。** Reified-Fact 在深层 Fact
   Complete/Recall 和 Joint F1 上略优；超图略低。这符合当前 HotpotQA
   主要是二跳 passage bridge、缺少真正 n-ary 角色/条件绑定监督的特征，不能据此
   否定超图在多元事实数据上的价值。
4. **外部图方法中 HyperGraphRAG 最强。** 它明显超过 GraphRAG、LightRAG
   和 PathRAG，但仍落后 QMSxE 的 Fact 排名。PathRAG 两题触发官方代码
   `text_units_context` 未赋值异常；此外其低 K 排名差，即使 R@10 在十候选
   设定下接近饱和，也不能弥补排序前置能力不足。

## 效率解释

报告中的 `Total s/query` 包含共享答案生成。QMSxE 三种检索的本地
index+query 均值约为 0.50--0.63 秒，加入答案生成后约为 1.39--1.58 秒；
Dense 加入答案生成后约为 1.12 秒。HyperGraphRAG 与 PathRAG 的新记录是
“复用既有官方索引后的 query-only”计时，分别约 1.99 秒和 1.44 秒，不可与
包含建索引的历史 GraphRAG/LightRAG 总时间直接横向解释。

报告中的 Token 是当前 trace 能观测到的 token，并不代表所有方法的完整消耗。
GraphRAG/LightRAG 的历史建索引和查询 token 不可恢复，因此它们表中的数值
主要来自共享答案生成，不能据此判断其比 BM25 或 Dense 更省 token。

API 成本列只覆盖 DeepSeek 调用。SiliconFlow embedding 的 token 在提供方
返回时已记录，但其价格未混入成本；历史 GraphRAG/LightRAG 的建索引 token
也不可恢复，因此报告显示为 N/A/部分覆盖，而不是错误地填成 0。

## 边界

284/288 题只有 10 个候选 passage，所以 Passage R@10 之后自然饱和；
R@20/30/40 仅为统一 schema 完整性而保留。Fact 候选通常超过 40，因此
Fact R@20/30/40 仍有区分力。若要验证大库检索能力，下一条独立 track 应使用
100/1000 或全局 passage pool，并重新运行各官方方法的原生索引流程。

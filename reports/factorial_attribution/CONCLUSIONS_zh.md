# QMSxE-RAG 全因子贡献归因结论

## 评测定义

本实验固定 Raw Dense 为共同起点，对所有合法因素子集做完整枚举：

- 普通图：Entity→Fact、BM25、Spectral、Graph Rerank、Base Reranker，共 `2^5`；
- 超图：BM25、Spectral、Graph Rerank、Base Reranker，共 `2^4`；
- 每个非 reranker 上下文额外比较 Base 与 LoRA reranker；
- 63 test × 3 seeds，共 22,680 条逐题记录；
- 所有配置统一最多 60 个 Fact 候选、输出 Top-40；
- 用精确 Shapley 分配主贡献，用平均二阶差分衡量交互；
- Shapley 最大可加性误差为 `4.44e-16`。

本轨道关闭生成器，只归因检索、路径和排序指标。因为候选预算被重新标准化，
锚点绝对分数不应与旧生产顺序评测直接混用。

## Fact Recall@20 的精确 Shapley 贡献

| 结构 | 因素 | Shapley | 95% CI | p |
|---|---|---:|---:|---:|
| Entity-Relation | Entity→Fact | **+0.5987** | [+0.5708, +0.6263] | 0.0001 |
| Entity-Relation | BM25 | **+0.2630** | [+0.2388, +0.2867] | 0.0001 |
| Entity-Relation | Base Reranker | +0.0689 | [+0.0550, +0.0830] | 0.0001 |
| Entity-Relation | Spectral | -0.0045 | [-0.0154, +0.0066] | 0.4379 |
| Entity-Relation | Graph Rerank | -0.0019 | [-0.0045, +0.0005] | 0.1390 |
| Reified-Fact | Base Reranker | **+0.0966** | [+0.0806, +0.1131] | 0.0001 |
| Reified-Fact | Entity→Fact | **+0.0878** | [+0.0662, +0.1100] | 0.0001 |
| Reified-Fact | Spectral | +0.0362 | [+0.0158, +0.0568] | 0.0005 |
| Reified-Fact | BM25 | +0.0303 | [+0.0193, +0.0416] | 0.0001 |
| Reified-Fact | Graph Rerank | **-0.0179** | [-0.0249, -0.0106] | 0.0001 |
| Hypergraph | BM25 | **+0.1199** | [+0.0930, +0.1475] | 0.0001 |
| Hypergraph | Base Reranker | **+0.0503** | [+0.0387, +0.0628] | 0.0001 |
| Hypergraph | Spectral | +0.0166 | [-0.0108, +0.0447] | 0.2373 |
| Hypergraph | Graph Rerank | **-0.0083** | [-0.0136, -0.0032] | 0.0013 |

## 到底哪个因素贡献最大

### Entity-Relation

Full Base 相对 Raw Minimal 的总增量约为 0.9242，其中 Entity→Fact 的 Shapley
为 0.5987，是决定性因素；BM25 为 0.2630，Base Reranker 为 0.0689。
Spectral 和 Graph Rerank 均没有正的显著贡献。因此 Entity-Relation 的高分主要
来自“召回实体后扩大为 Fact 候选”，而不是多频带图表示。

### Reified-Fact

Full Base 相对 Raw Minimal 的增量约为 0.2331。Base Reranker 和 Entity→Fact
贡献最大且接近；Spectral 有 +0.0362 的小而显著增益，BM25 为 +0.0303。
Graph Rerank 反而显著降低 Recall@20，并大幅降低 MRR（Shapley -0.1496）。

### Hypergraph

Full Base 相对 Raw Minimal 的增量约为 0.1786。BM25 贡献 0.1199，占主要部分；
Base Reranker 贡献 0.0503。Spectral 为 +0.0166，但置信区间跨零；Graph Rerank
显著为负。因此超图当前最依赖词法候选和神经精排，而不是 Stage A 谱候选。

## Reranker 与 LoRA

Base Reranker 对 Recall@20 的 Shapley 分别为：

- Entity-Relation：+0.0689；
- Reified-Fact：+0.0966；
- Hypergraph：+0.0503。

它对 MRR 的贡献更大，分别为 +0.2652、+0.2694、+0.2249，说明 reranker 的
主要价值是把已召回的正确 Fact 提前，而不是无限增加候选覆盖率。

LoRA 相对 Base 的平均上下文增量较小但显著：

| 结构 | Fact R@20 | MRR | Path F1 |
|---|---:|---:|---:|
| Entity-Relation | +0.0104 | +0.0258 | +0.0463 |
| Reified-Fact | +0.0153 | +0.0496 | +0.0747 |
| Hypergraph | +0.0079 | +0.0489 | +0.0608 |

LoRA 不是最大召回来源，但对前排排序、路径质量有稳定增益。

## 关键交互

- Entity-Relation 的 `Entity→Fact × BM25 = -0.5799`：二者高度替代。关闭展开时，
  BM25 能直接提供 Fact；打开展开后，BM25 的额外边际贡献显著缩小。不能把两个
  单项贡献直接相加后声称是独立收益。
- Entity-Relation 的 `Entity→Fact × Spectral = +0.0859`：谱检索只有在实体能
  转换成 Fact 时才更有价值，但其平均主效应仍接近零。
- Hypergraph 的 `BM25 × Spectral = +0.0832`：二者存在明显互补，这解释了为何
  超图 Full 比单独删除某一来源更敏感。
- Hypergraph 的 `BM25 × Base Reranker = +0.0743`：BM25 召回的候选需要 reranker
  提前才能充分转化为最终指标。
- Reified-Fact 的 `Entity→Fact × Base Reranker = +0.0727`：展开提高覆盖率，
  reranker 负责从扩大的候选集中恢复前排精度。

## 最终判断

“最终系统高分主要来自 Entity→Fact、BM25、多源融合和微调 reranker”现在可以
拆成更准确的结论：

1. Entity-Relation 的首要来源是 Entity→Fact，BM25 第二，Base Reranker 第三；
2. Reified-Fact 的首要来源是 Base Reranker 与 Entity→Fact，Spectral 只有小幅贡献；
3. Hypergraph 的首要来源是 BM25，其次是 Base Reranker，并依赖 BM25×Spectral
   和 BM25×Reranker 的正交互；
4. LoRA 有稳定但次级的增量，主要改善 MRR 和 Path F1；
5. 多频带 Spectral 只在 Reified-Fact 上得到小而显著的 Recall 增益，在
   Entity-Relation 和 Hypergraph 上未被证明有效；
6. 当前 Graph Rerank 已在普通图和超图生产链路中默认关闭；旧实现仅保留为显式
   opt-in，供后续重做和复现实验使用。

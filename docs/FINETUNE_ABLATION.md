# 多随机种子与微调消融协议

本实验比较 Vanilla Dense、QMSHE 超图、Entity-Relation 普通图和 Reified-Fact 普通图，
并对 embedding 与 reranker 的微调效果进行完整 2×2 消融：

| Embedding | Reranker | 目的 |
|---|---|---|
| base | base | 未微调基线 |
| tuned | base | embedding 微调的独立贡献 |
| base | tuned | reranker 微调的独立贡献 |
| tuned | tuned | 联合效果与交互作用 |

## 防止数据泄漏

- 前 500 条 HotpotQA 使用 example ID 的 SHA-256 哈希固定划分；当前得到 362 train、75 validation、63 test。
- 所有 seed 使用同一 train/validation/test 集；seed 不改变测试样本。
- validation 只用于选择最佳 epoch；最终矩阵只在 test 上计算。
- 文档语料在训练期可见，test 问题和支持事实标签不可见，属于 transductive corpus retrieval 设定。
- BGE-M3 只对 query encoder 添加 LoRA，文档 embedding 始终由冻结 base model 生成。
- reranker LoRA 使用训练问题的正例，并仅从同一道题自带的 distractor passages 中挖掘
  BM25 hard negatives，使负例分布与实际候选重排一致。

## 随机性与统计

默认 seed 为 13、42、73。每个 seed 独立训练两个 LoRA adapter，并控制谱模型初始化。
主表报告各 seed 测试集均值的 mean±sample SD。普通图与超图的 Recall@20 差异使用配对
randomization test，并通过 10,000 次 paired bootstrap 给出 95% 置信区间。

## 运行

```bash
for seed in 13 42 73; do
  python scripts/train_query_lora.py \
    --base-model /models/bge-m3 --limit 500 --epochs 3 --seed "$seed" \
    --output-dir "data/models/bge_m3_query_lora_seed_${seed}"
  python scripts/train_reranker_lora.py \
    --base-model /models/bge-reranker-v2-m3 --limit 500 --epochs 3 --seed "$seed" \
    --output-dir "data/models/bge_reranker_lora_seed_${seed}"
done

python scripts/run_finetune_seed_matrix.py \
  --embedding-model /models/bge-m3 \
  --reranker-model /models/bge-reranker-v2-m3 \
  --embedding-adapters '13=data/models/bge_m3_query_lora_seed_13,42=data/models/bge_m3_query_lora_seed_42,73=data/models/bge_m3_query_lora_seed_73' \
  --reranker-adapters '13=data/models/bge_reranker_lora_seed_13,42=data/models/bge_reranker_lora_seed_42,73=data/models/bge_reranker_lora_seed_73'
```

输出包括逐样本记录、跨 seed 汇总、显著性结果和 Markdown 主表。

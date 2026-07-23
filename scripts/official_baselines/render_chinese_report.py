from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "pathrag": "PathRAG（官方）",
    "graphrag": "Microsoft GraphRAG（官方）",
    "lightrag": "LightRAG（官方）",
    "hypergraphrag": "HyperGraphRAG（官方）",
    "qmsxe:graph:entity_relation": "QMSxE Entity-Relation",
    "qmsxe:graph:reified_fact": "QMSxE Reified-Fact",
    "qmsxe:hypergraph:evidence_hypergraph": "QMSxE Hypergraph",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    official = json.loads(args.official_summary.read_text(encoding="utf-8"))
    summary = comparison["summary"]
    ordered = sorted(summary, key=lambda name: summary[name]["recall_at_2"], reverse=True)
    best = ordered[0]
    best_qmsxe = next(name for name in ordered if name.startswith("qmsxe:"))
    best_official = next(name for name in ordered if not name.startswith("qmsxe:"))
    official_pair = comparison["paired_comparisons"][f"{best_qmsxe} minus {best_official}"][
        "recall_at_2"
    ]
    internal_pair = comparison["paired_comparisons"][
        "qmsxe:graph:entity_relation minus qmsxe:graph:reified_fact"
    ]["recall_at_2"]
    hypergraph_pair = comparison["paired_comparisons"][
        "qmsxe:graph:entity_relation minus qmsxe:hypergraph:evidence_hypergraph"
    ]["recall_at_2"]
    lines = [
        "# 官方开源 Graph RAG 基线与 QMSxE 对比结论",
        "",
        "## 主表（统一 passage 检索口径）",
        "",
        "| 方法 | N | R@1 | R@2 | R@5 | R@10 | Complete@2 | Complete@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ordered:
        row = summary[name]
        lines.append(
            f"| {LABELS.get(name, name)} | {row['example_count']} | "
            f"{row['recall_at_1']:.4f} | {row['recall_at_2']:.4f} | "
            f"{row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} | "
            f"{row['complete_at_2']:.4f} | {row['complete_at_5']:.4f} | "
            f"{row['mrr']:.4f} |"
        )
    delta = summary[best_qmsxe]["recall_at_2"] - summary[best_official]["recall_at_2"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- R@2 最高的是 **{LABELS.get(best, best)}**。",
            f"- 最强 QMSxE 相对最强官方基线的 R@2 差值为 {delta:+.4f}。",
            f"- 该 R@2 差值的配对随机化 p 值为 {official_pair['paired_randomization_p_value']:.4f}，95% bootstrap CI 为 [{official_pair['difference_ci_95'][0]:.4f}, {official_pair['difference_ci_95'][1]:.4f}]。",
            f"- Entity-Relation 与 Reified-Fact 的 R@2 差值仅 {internal_pair['mean_difference']:+.4f}（p={internal_pair['paired_randomization_p_value']:.4f}），不能据此认定两者存在稳定差异。",
            f"- Entity-Relation 相对项目超图的 R@2 高 {hypergraph_pair['mean_difference']:.4f}（p={hypergraph_pair['paired_randomization_p_value']:.4f}）；但 Complete@2 的差异未达到 0.05 显著水平。",
            "- Complete@2 可视为本实验的 passage 证据集合准确率：只有前两名覆盖全部支持文档才记 1。它不是最终答案生成 Accuracy。",
            "- 配对随机化检验与 bootstrap 95% 置信区间保存在 `comparison.json`，QMSxE 的三个训练种子先在每题内求平均，再与官方方法逐题配对。",
            "",
            "## 公平性与边界",
            "",
            "- 七种方法使用完全相同的 288 道固定测试题、每题十篇候选 passage 和 supporting-fact 派生的 gold passage。",
            "- 四个官方系统均调用固定 Git 提交的公开 API；适配器只把官方返回上下文映射回 passage ID。",
            "- 官方系统使用 DeepSeek 做图抽取、SiliconFlow BGE-M3 做 embedding，并将 max gleanings 设为 0；这控制了成本，但可能低估多轮抽取的上限。",
            "- QMSxE 使用项目现有的确定性 sentence/fact 建图，因此本表比较的是端到端检索管线，不是隔离图检索算法的纯组件实验。",
            "- 旧 BM25/Dense 主表是 fact 级 top-k，不能与这里的 passage 级数字直接横向排序。",
            "- PathRAG 与 HyperGraphRAG 的 R@10 为 1.0，是因为候选集本身只有十篇且返回上下文平均接近十篇；判断排序质量应优先看 R@1/R@2、Complete@2 和 MRR。",
            "- `audit.json` 保留了 GraphRAG 适配器修复前的 18 条相对路径错误；相同题目随后全部成功重跑，正式 288 条记录不包含这些失败结果，也没有成功记录重复。",
            "",
            "## 官方运行完整性",
            "",
            "| 方法 | 成功 | 错误 | 平均返回篇数 | 平均耗时/题（秒） |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in sorted(official):
        row = official[name]
        lines.append(
            f"| {LABELS.get(name, name)} | {row['successful_count']} | "
            f"{row['error_count']} | {row['ranking_length_mean']:.2f} | "
            f"{row['elapsed_seconds_mean']:.2f} |"
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

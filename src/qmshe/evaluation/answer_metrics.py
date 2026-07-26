import re


def citation_precision(answer: str, allowed_ids: set[str]) -> float:
    cited = set(re.findall(r"\[(fact_[^\]]+)\]", answer))
    return len(cited & allowed_ids) / max(len(cited), 1)


def citation_recall(answer: str, gold_ids: set[str]) -> float:
    cited = set(re.findall(r"\[(fact_[^\]]+)\]", answer))
    return len(cited & gold_ids) / max(len(gold_ids), 1)


def citation_f1(answer: str, gold_ids: set[str]) -> float:
    precision = citation_precision(answer, gold_ids)
    recall = citation_recall(answer, gold_ids)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

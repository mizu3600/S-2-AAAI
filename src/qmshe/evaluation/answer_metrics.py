import re


def citation_precision(answer: str, allowed_ids: set[str]) -> float:
    cited = set(re.findall(r"\[(fact_[^\]]+)\]", answer))
    return len(cited & allowed_ids) / max(len(cited), 1)


def citation_recall(answer: str, gold_ids: set[str]) -> float:
    cited = set(re.findall(r"\[(fact_[^\]]+)\]", answer))
    return len(cited & gold_ids) / max(len(gold_ids), 1)


def path_coverage(retrieved_nodes: set[str], gold_path: list[str]) -> float:
    return len(retrieved_nodes & set(gold_path)) / max(len(set(gold_path)), 1)


def path_recall(retrieved_paths: list[list[str]], gold_path: list[str]) -> float:
    gold_edges = set(zip(gold_path, gold_path[1:])) | set(zip(gold_path[1:], gold_path))
    retrieved_edges = set()
    for path in retrieved_paths:
        retrieved_edges |= set(zip(path, path[1:])) | set(zip(path[1:], path))
    return len(gold_edges & retrieved_edges) / max(len(gold_edges), 1)


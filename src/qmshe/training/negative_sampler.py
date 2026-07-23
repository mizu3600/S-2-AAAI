import random

import networkx as nx


class DualHardNegativeSampler:
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def sample(
        self, positives: set[str], graph: nx.Graph, semantic_ranked_ids: list[str],
        all_ids: list[str], semantic_count: int = 4, structural_count: int = 4, random_count: int = 8,
    ) -> dict[str, list[str]]:
        semantic = [item for item in semantic_ranked_ids if item not in positives][:semantic_count]
        structural_pool = set()
        for positive in positives:
            if positive in graph:
                structural_pool.update(nx.single_source_shortest_path_length(graph, positive, cutoff=2))
        structural = [item for item in structural_pool if item not in positives][:structural_count]
        excluded = positives | set(semantic) | set(structural)
        pool = [item for item in all_ids if item not in excluded]
        random_items = self.random.sample(pool, min(random_count, len(pool)))
        return {"semantic_hard": semantic, "structural_hard": structural, "random": random_items}


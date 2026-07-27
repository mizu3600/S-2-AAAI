import torch


def analytic_band_weights(
    query_embedding: torch.Tensor,
    pooled_bands: list[torch.Tensor],
) -> torch.Tensor:
    query = torch.nn.functional.normalize(query_embedding, dim=-1)
    scores = []
    for band in pooled_bands:
        if float(torch.linalg.vector_norm(band)) <= 1e-12:
            scores.append(query.new_tensor(-1.0))
        else:
            normalized = torch.nn.functional.normalize(band, dim=-1)
            scores.append(torch.dot(query, normalized))
    return torch.softmax(torch.stack(scores), dim=0)


def build_seed_weights(
    query_embedding: torch.Tensor,
    raw_embeddings: torch.Tensor,
    top_m: int = 64,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.nn.functional.normalize(query_embedding, dim=-1)
    nodes = torch.nn.functional.normalize(raw_embeddings, dim=-1)
    scores = nodes @ query
    count = min(top_m, len(scores))
    values, indices = torch.topk(scores, k=count)
    return indices, torch.softmax(values / temperature, dim=0)


def pool_seed_bands(
    indices: torch.Tensor, weights: torch.Tensor, band_embeddings: list[torch.Tensor]
) -> list[torch.Tensor]:
    return [(band[indices] * weights[:, None]).sum(dim=0) for band in band_embeddings]

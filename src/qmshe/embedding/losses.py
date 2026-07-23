import torch
import torch.nn.functional as functional


def retrieval_contrastive_loss(scores: torch.Tensor, positive_index: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    return functional.cross_entropy(scores / temperature, positive_index)


def margin_ranking_loss(positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    return torch.relu(margin - positive + negative).mean()


def reconstruction_loss(raw: torch.Tensor, bands: list[torch.Tensor]) -> torch.Tensor:
    return functional.mse_loss(sum(bands), raw)


def diversity_loss(bands: list[torch.Tensor]) -> torch.Tensor:
    normalized = [functional.normalize(band, dim=0) for band in bands]
    loss = torch.zeros((), device=bands[0].device)
    for i, left in enumerate(normalized):
        for right in normalized[i + 1 :]:
            loss = loss + (left.T @ right).pow(2).mean()
    return loss


def gate_usage_loss(gates: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    eps = 1e-8
    mean_gate = gates.mean(dim=0)
    global_entropy = -(mean_gate * (mean_gate + eps).log()).sum()
    local_entropy = -(gates * (gates + eps).log()).sum(dim=-1).mean()
    return -global_entropy + gamma * local_entropy


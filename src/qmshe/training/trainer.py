from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainStats:
    loss: float
    steps: int


def train_epoch(model, batches, optimizer, loss_fn, gradient_clip: float = 1.0) -> TrainStats:
    model.train()
    total, steps = 0.0, 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total += float(loss.detach())
        steps += 1
    return TrainStats(loss=total / max(steps, 1), steps=steps)


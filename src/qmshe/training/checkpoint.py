from pathlib import Path

import torch


def save_checkpoint(model, optimizer, path: str | Path, graph_version: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "graph_version": graph_version}, path)


def load_checkpoint(model, optimizer, path: str | Path, expected_graph_version: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["graph_version"] != expected_graph_version:
        raise ValueError("checkpoint graph version is incompatible")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])


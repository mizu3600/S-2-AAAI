import torch

from qmshe.training.inductive_stage_a import _apply_spectral_variant


def _bands():
    return {
        "raw": torch.ones(2, 2),
        "low": torch.ones(2, 1),
        "mid": torch.ones(2, 1),
        "high": torch.ones(2, 1),
    }


def test_raw_only_masks_all_graph_frequency_blocks():
    query = torch.arange(1, 6, dtype=torch.float32)
    nodes = torch.ones(2, 5)
    masked_query, masked_nodes = _apply_spectral_variant(
        query, nodes, _bands(), torch.full((4,), 0.25), "raw_only"
    )
    assert torch.equal(masked_query, torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0]))
    assert torch.equal(masked_nodes[:, :2], torch.ones(2, 2))
    assert torch.count_nonzero(masked_nodes[:, 2:]) == 0


def test_fixed_gate_removes_learned_gate_scaling_without_changing_nodes():
    gate = torch.tensor([0.5, 0.2, 0.2, 0.1])
    base_parts = torch.tensor([2.0, 4.0, 3.0, 5.0, 7.0])
    query = base_parts * torch.tensor([0.5, 0.5, 0.2, 0.2, 0.1])
    nodes = torch.ones(2, 5)
    fixed_query, fixed_nodes = _apply_spectral_variant(query, nodes, _bands(), gate, "fixed_gate")
    assert torch.allclose(fixed_query, base_parts * 0.25)
    assert torch.equal(fixed_nodes, nodes)

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
import torch.nn.functional as functional

from qmshe.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank
from qmshe.training.inductive_stage_a import _prepare, _rank_facts


@dataclass(frozen=True)
class SpectralStageBConfig:
    base_model: str
    stage_a_checkpoint: str
    output_dir: str
    mode: str
    profile: str = "evidence_hypergraph"
    max_length: int = 256
    learning_rate: float = 2e-5
    epochs: int = 3
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    seed: int = 42
    max_memory_fraction: float = 0.45
    device: str = "cuda"


@dataclass(frozen=True)
class SpectralStageBReport:
    mode: str
    profile: str
    seed: int
    train_examples: int
    validation_examples: int
    trainable_parameters: int
    total_parameters: int
    loss_history: list[float]
    validation_history: list[dict[str, float]]
    selected_epoch: int
    recall_at_20_before: float
    recall_at_20_after: float
    mrr_before: float
    mrr_after: float
    elapsed_seconds: float
    peak_gpu_memory_mb: float
    dataset_sha256: str
    device: str


@dataclass
class _FrozenGraph:
    question: str
    raw_features: torch.Tensor
    node_bands: dict[str, torch.Tensor]
    role_node_bands: dict[str, dict[str, torch.Tensor]]
    node_ids: list[str]
    positive_indices: list[int]
    facts_by_entity: dict[str, set[str]]
    fact_by_node: dict[str, str]
    gold_fact_ids: set[str]


def train_spectral_stage_b(
    training_examples, validation_examples, document_encoder, config: SpectralStageBConfig,
) -> SpectralStageBReport:
    try:
        from peft import (
            LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the 'lora' extra to train Stage B") from exc
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(config.max_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    checkpoint = torch.load(config.stage_a_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("mode") != config.mode or checkpoint.get("profile") != config.profile:
        raise ValueError("Stage A checkpoint mode/profile does not match Stage B")
    if checkpoint.get("seed") != config.seed:
        raise ValueError("Stage A and Stage B seeds must match")
    train_prepared, template, roles = _prepare(
        training_examples, document_encoder, config.mode, config.profile, config.seed
    )
    validation_prepared, _, validation_roles = _prepare(
        validation_examples, document_encoder, config.mode, config.profile, config.seed
    )
    if roles != validation_roles or roles != checkpoint.get("roles", []):
        raise ValueError("Stage A and Stage B role vocabularies do not match")
    stage_a_model = template.model.to(device)
    stage_a_model.load_state_dict(checkpoint["model"])
    stage_a_model.requires_grad_(False).eval()
    relation_gate = None
    if config.mode == "hypergraph":
        relation_gate = template.relation_gate.to(device)
        relation_gate.load_state_dict(checkpoint["relation_gate"])
        relation_gate.requires_grad_(False).eval()
    train_graphs = _freeze_graphs(
        train_prepared, stage_a_model, roles, device
    )
    validation_graphs = _freeze_graphs(
        validation_prepared, stage_a_model, roles, device
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, local_files_only=True)
    query_model = AutoModel.from_pretrained(config.base_model, local_files_only=True).to(device)
    query_model.eval()
    before_recall, before_mrr = _evaluate(
        query_model, tokenizer, stage_a_model, relation_gate, validation_graphs,
        roles, device, config.max_length,
    )
    query_model = get_peft_model(query_model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=config.lora_rank,
        lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=["query", "key", "value"], bias="none",
    ))
    trainable = [parameter for parameter in query_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    started = perf_counter()
    loss_history, validation_history = [], []
    best_state, best_score, selected_epoch = None, (-1.0, -1.0), 0
    rng = random.Random(config.seed)
    for epoch in range(config.epochs):
        order = list(range(len(train_graphs)))
        rng.shuffle(order)
        query_model.train()
        losses = []
        for index in order:
            graph = train_graphs[index]
            query = _encode_query(
                query_model, tokenizer, graph.question, device, config.max_length
            )
            scores = _score(
                stage_a_model, relation_gate, graph, query, roles, device
            )
            positive = torch.tensor(graph.positive_indices, device=device)
            positives = scores[positive]
            loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(positives, dim=0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        loss_history.append(mean(losses))
        recall, mrr = _evaluate(
            query_model, tokenizer, stage_a_model, relation_gate, validation_graphs,
            roles, device, config.max_length,
        )
        validation_history.append({
            "epoch": float(epoch + 1), "recall_at_20": recall, "mrr": mrr,
        })
        if (recall, mrr) > best_score:
            best_score, selected_epoch = (recall, mrr), epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in get_peft_model_state_dict(query_model).items()
            }
    if best_state is not None:
        set_peft_model_state_dict(query_model, best_state)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    query_model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    payload = json.dumps({
        "train_ids": [item.example_id for item in training_examples],
        "validation_ids": [item.example_id for item in validation_examples],
        "mode": config.mode, "profile": config.profile,
        "stage_a_checkpoint": Path(config.stage_a_checkpoint).name,
    }, sort_keys=True).encode()
    report = SpectralStageBReport(
        mode=config.mode, profile=config.profile, seed=config.seed,
        train_examples=len(train_graphs), validation_examples=len(validation_graphs),
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        total_parameters=sum(parameter.numel() for parameter in query_model.parameters()),
        loss_history=loss_history, validation_history=validation_history,
        selected_epoch=selected_epoch, recall_at_20_before=before_recall,
        recall_at_20_after=best_score[0], mrr_before=before_mrr, mrr_after=best_score[1],
        elapsed_seconds=perf_counter() - started,
        peak_gpu_memory_mb=(
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        ),
        dataset_sha256=hashlib.sha256(payload).hexdigest(), device=str(device),
    )
    (output / "training_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    (output / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    return report


def _freeze_graphs(graphs, model, roles, device):
    output = []
    with torch.no_grad():
        for graph in graphs:
            raw = graph.raw_features.to(device)
            bands = model.encode_nodes(raw, graph.operator.to(device))
            role_bands = {
                role: model.encode_nodes(raw, graph.role_operators[role].to(device))
                for role in roles
            }
            output.append(_FrozenGraph(
                question=graph.question, raw_features=graph.raw_features,
                node_bands={name: value.cpu() for name, value in bands.items()},
                role_node_bands={
                    role: {name: value.cpu() for name, value in values.items()}
                    for role, values in role_bands.items()
                },
                node_ids=graph.node_ids, positive_indices=graph.positive_indices,
                facts_by_entity=graph.facts_by_entity, fact_by_node=graph.fact_by_node,
                gold_fact_ids=graph.gold_fact_ids,
            ))
    return output


def _encode_query(model, tokenizer, question, device, max_length):
    encoded = tokenizer(
        [question], padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    return functional.normalize(model(**encoded).last_hidden_state[:, 0], p=2, dim=-1)[0]


def _score(model, relation_gate, graph, query, roles, device):
    raw = graph.raw_features.to(device)
    bands = {name: value.to(device) for name, value in graph.node_bands.items()}
    if relation_gate is not None:
        weights = relation_gate(query)
        conditioned = {"raw": bands["raw"]}
        for name in ("low", "mid", "high"):
            stacked = torch.stack([
                graph.role_node_bands[role][name].to(device) for role in roles
            ])
            conditioned[name] = torch.einsum("r,rnd->nd", weights, stacked)
        bands = conditioned
    query_vector, _ = model.encode_query(query, raw, bands)
    full = torch.cat([bands[name] for name in ("raw", "low", "mid", "high")], dim=-1)
    return full @ query_vector


def _evaluate(model, tokenizer, stage_a_model, relation_gate, graphs, roles, device, max_length):
    model.eval()
    recalls, reciprocal_ranks = [], []
    with torch.no_grad():
        for graph in graphs:
            query = _encode_query(model, tokenizer, graph.question, device, max_length)
            scores = _score(stage_a_model, relation_gate, graph, query, roles, device)
            ranking = torch.argsort(scores, descending=True).cpu().tolist()
            ranked_nodes = [graph.node_ids[index] for index in ranking]
            ranked_facts = _rank_facts(ranked_nodes, graph)
            recalls.append(recall_at_k(ranked_facts, graph.gold_fact_ids, 20))
            reciprocal_ranks.append(reciprocal_rank(ranked_facts, graph.gold_fact_ids))
    return mean(recalls), mean(reciprocal_ranks)

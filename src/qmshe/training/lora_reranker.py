from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class RerankerLoRAConfig:
    base_model: str
    output_dir: str
    max_length: int = 384
    batch_size: int = 8
    learning_rate: float = 1e-5
    epochs: int = 3
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    negatives_per_query: int = 12
    seed: int = 42
    max_memory_fraction: float = 0.45
    cpu_threads: int = 4


@dataclass(frozen=True)
class RerankerLoRAReport:
    base_model: str
    trainable_parameters: int
    total_parameters: int
    train_queries: int
    evaluation_queries: int
    training_pairs: int
    epochs: int
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


def train_reranker_lora(
    training: list[tuple[str, set[str], list[str]]],
    evaluation: list[tuple[str, set[str], list[str]]],
    fact_text_by_id: dict[str, str], config: RerankerLoRAConfig,
) -> RerankerLoRAReport:
    if not training or not evaluation:
        raise ValueError("training and evaluation data must be non-empty")
    try:
        from peft import (
            LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the 'lora' extra to train the reranker") from exc

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(max(config.cpu_threads, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(config.max_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model, local_files_only=True
    ).to(device)
    before_recall, before_mrr = _evaluate(
        model, tokenizer, evaluation, fact_text_by_id, device, config
    )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=config.lora_rank, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=["query", "key", "value"],
        bias="none", modules_to_save=[],
    ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    pairs = _make_pairs(training, fact_text_by_id, config.negatives_per_query)
    started = perf_counter()
    loss_history, validation_history = [], []
    best_state, best_score, selected_epoch = None, (-1.0, -1.0), 0
    for epoch in range(config.epochs):
        random.shuffle(pairs)
        model.train()
        total_loss = 0.0
        for start in range(0, len(pairs), config.batch_size):
            batch = pairs[start:start + config.batch_size]
            encoded = tokenizer(
                [item[0] for item in batch], [item[1] for item in batch], padding=True,
                truncation=True, max_length=config.max_length, return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            labels = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
            logits = model(**encoded).logits.reshape(-1)
            loss = functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
        loss_history.append(total_loss / len(pairs))
        recall, mrr = _evaluate(model, tokenizer, evaluation, fact_text_by_id, device, config)
        validation_history.append({
            "epoch": float(epoch + 1), "recall_at_20": recall, "mrr": mrr,
        })
        if (recall, mrr) > best_score:
            best_score, selected_epoch = (recall, mrr), epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in get_peft_model_state_dict(model).items()
            }
    if best_state is not None:
        set_peft_model_state_dict(model, best_state)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    payload = json.dumps({
        "training": [(q, sorted(p), c) for q, p, c in training],
        "evaluation": [(q, sorted(p), c) for q, p, c in evaluation],
    }, sort_keys=True).encode()
    report = RerankerLoRAReport(
        base_model=config.base_model,
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        train_queries=len(training), evaluation_queries=len(evaluation), training_pairs=len(pairs),
        epochs=config.epochs, loss_history=loss_history, validation_history=validation_history,
        selected_epoch=selected_epoch, recall_at_20_before=before_recall,
        recall_at_20_after=best_score[0], mrr_before=before_mrr, mrr_after=best_score[1],
        elapsed_seconds=perf_counter() - started,
        peak_gpu_memory_mb=(torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0),
        dataset_sha256=hashlib.sha256(payload).hexdigest(), device=str(device),
    )
    (output / "training_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    (output / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    return report


def _make_pairs(training, fact_text_by_id, negatives_per_query):
    output = []
    for question, positives, candidates in training:
        for fact_id in sorted(positives):
            if fact_id in fact_text_by_id:
                output.append((question, fact_text_by_id[fact_id], 1.0))
        used = 0
        for fact_id in candidates:
            if fact_id not in positives and fact_id in fact_text_by_id:
                output.append((question, fact_text_by_id[fact_id], 0.0))
                used += 1
                if used >= negatives_per_query:
                    break
    if not output:
        raise ValueError("no reranker pairs could be constructed")
    return output


def _evaluate(model, tokenizer, examples, fact_text_by_id, device, config):
    model.eval()
    recalls, reciprocal_ranks = [], []
    with torch.inference_mode():
        for question, positives, candidates in examples:
            candidates = [item for item in candidates if item in fact_text_by_id]
            scores = []
            for start in range(0, len(candidates), config.batch_size):
                ids = candidates[start:start + config.batch_size]
                encoded = tokenizer(
                    [question] * len(ids), [fact_text_by_id[item] for item in ids], padding=True,
                    truncation=True, max_length=config.max_length, return_tensors="pt",
                )
                encoded = {name: value.to(device) for name, value in encoded.items()}
                scores.extend(model(**encoded).logits.reshape(-1).float().cpu().tolist())
            ranked = [item for _, item in sorted(zip(scores, candidates), reverse=True)]
            recalls.append(len(set(ranked[:20]) & positives) / max(len(positives), 1))
            ranks = [index + 1 for index, item in enumerate(ranked) if item in positives]
            reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
    return sum(recalls) / len(recalls), sum(reciprocal_ranks) / len(reciprocal_ranks)

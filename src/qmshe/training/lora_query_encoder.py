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
class LoRATrainingConfig:
    base_model: str
    output_dir: str
    max_length: int = 256
    document_batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 2
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    semantic_negatives: int = 8
    random_negatives: int = 8
    temperature: float = 0.05
    seed: int = 42
    max_memory_fraction: float = 0.45
    cpu_threads: int = 4


@dataclass(frozen=True)
class LoRATrainingReport:
    base_model: str
    trainable_parameters: int
    total_parameters: int
    train_examples: int
    evaluation_examples: int
    candidate_facts: int
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


def train_query_lora(
    questions: list[str], positive_ids: list[set[str]], fact_text_by_id: dict[str, str],
    config: LoRATrainingConfig,
    evaluation_questions: list[str] | None = None,
    evaluation_positive_ids: list[set[str]] | None = None,
) -> LoRATrainingReport:
    """Fine-tune only the BGE-M3 query encoder LoRA against frozen fact embeddings."""
    if len(questions) != len(positive_ids) or not questions:
        raise ValueError("questions and positive_ids must be non-empty and aligned")
    evaluation_questions = evaluation_questions or questions
    evaluation_positive_ids = evaluation_positive_ids or positive_ids
    if len(evaluation_questions) != len(evaluation_positive_ids):
        raise ValueError("evaluation questions and positives must be aligned")
    try:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the 'lora' extra to train the query encoder") from exc

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(max(config.cpu_threads, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(config.max_memory_fraction)
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, local_files_only=True)
    model = AutoModel.from_pretrained(config.base_model, local_files_only=True)
    model.to(device)
    model.eval()
    fact_ids = list(fact_text_by_id)
    fact_texts = [fact_text_by_id[fact_id] for fact_id in fact_ids]
    with torch.no_grad():
        frozen_facts = _encode_texts(
            model, tokenizer, fact_texts, device, config.document_batch_size, config.max_length
        ).cpu()
        base_train_queries = _encode_texts(
            model, tokenizer, questions, device, config.document_batch_size, config.max_length
        ).cpu()
        base_evaluation_queries = _encode_texts(
            model, tokenizer, evaluation_questions, device,
            config.document_batch_size, config.max_length
        ).cpu()
    before_recall, before_mrr = _retrieval_metrics(
        base_evaluation_queries, frozen_facts, fact_ids, evaluation_positive_ids, top_k=20
    )

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["query", "key", "value"],
        bias="none",
    ))
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    fact_index = {fact_id: index for index, fact_id in enumerate(fact_ids)}
    negative_sets = _mine_negatives(
        base_train_queries, frozen_facts, fact_ids, positive_ids,
        config.semantic_negatives, config.random_negatives, config.seed,
    )
    started = perf_counter()
    loss_history = []
    validation_history = []
    best_state = None
    best_score = (-1.0, -1.0)
    selected_epoch = 0
    for epoch in range(config.epochs):
        order = list(range(len(questions)))
        random.shuffle(order)
        epoch_loss = 0.0
        used = 0
        for example_index in order:
            positive = [
                fact_index[fact_id] for fact_id in positive_ids[example_index]
                if fact_id in fact_index
            ]
            if not positive:
                continue
            candidate_indices = list(dict.fromkeys([*positive, *negative_sets[example_index]]))
            positive_positions = torch.tensor(
                [candidate_indices.index(item) for item in positive], device=device
            )
            query_embedding = _encode_batch(
                model, tokenizer, [questions[example_index]], device, config.max_length
            )[0]
            candidates = frozen_facts[candidate_indices].to(device)
            scores = (candidates @ query_embedding) / config.temperature
            loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(
                scores[positive_positions], dim=0
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            used += 1
        loss_history.append(epoch_loss / max(used, 1))
        model.eval()
        with torch.no_grad():
            epoch_queries = _encode_texts(
                model, tokenizer, evaluation_questions, device,
                config.document_batch_size, config.max_length
            ).cpu()
        epoch_recall, epoch_mrr = _retrieval_metrics(
            epoch_queries, frozen_facts, fact_ids, evaluation_positive_ids, top_k=20
        )
        validation_history.append({
            "epoch": float(epoch + 1), "recall_at_20": epoch_recall, "mrr": epoch_mrr
        })
        if (epoch_recall, epoch_mrr) > best_score:
            best_score = (epoch_recall, epoch_mrr)
            selected_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in get_peft_model_state_dict(model).items()
            }
        model.train()
    if best_state is not None:
        set_peft_model_state_dict(model, best_state)
    model.eval()
    after_recall, after_mrr = best_score
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    payload = json.dumps({
        "questions": questions,
        "positive_ids": [sorted(items) for items in positive_ids],
        "evaluation_questions": evaluation_questions,
        "evaluation_positive_ids": [sorted(items) for items in evaluation_positive_ids],
        "fact_ids": fact_ids,
    }, sort_keys=True).encode()
    report = LoRATrainingReport(
        base_model=config.base_model,
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        train_examples=len(questions), evaluation_examples=len(evaluation_questions),
        candidate_facts=len(fact_ids), epochs=config.epochs,
        loss_history=loss_history, validation_history=validation_history,
        selected_epoch=selected_epoch,
        recall_at_20_before=before_recall, recall_at_20_after=after_recall,
        mrr_before=before_mrr, mrr_after=after_mrr,
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


def _encode_batch(model, tokenizer, texts, device, max_length):
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model(**encoded)
    return functional.normalize(output.last_hidden_state[:, 0], p=2, dim=-1)


def _encode_texts(model, tokenizer, texts, device, batch_size, max_length):
    batches = []
    for start in range(0, len(texts), batch_size):
        batches.append(_encode_batch(
            model, tokenizer, texts[start:start + batch_size], device, max_length
        ))
    return torch.cat(batches, dim=0)


def _mine_negatives(query_vectors, fact_vectors, fact_ids, positives, semantic_count, random_count, seed):
    generator = random.Random(seed)
    all_indices = list(range(len(fact_ids)))
    output = []
    for query, positive_ids in zip(query_vectors, positives, strict=True):
        positive_indices = {index for index, fact_id in enumerate(fact_ids) if fact_id in positive_ids}
        ranking = torch.argsort(fact_vectors @ query, descending=True).tolist()
        semantic = [index for index in ranking if index not in positive_indices][:semantic_count]
        pool = [index for index in all_indices if index not in positive_indices | set(semantic)]
        random_items = generator.sample(pool, min(random_count, len(pool)))
        output.append([*semantic, *random_items])
    return output


def _retrieval_metrics(query_vectors, fact_vectors, fact_ids, positives, top_k):
    recalls, reciprocal_ranks = [], []
    for query, positive_ids in zip(query_vectors, positives, strict=True):
        ranking = torch.argsort(fact_vectors @ query, descending=True).tolist()
        ranked_ids = [fact_ids[index] for index in ranking]
        recalls.append(len(set(ranked_ids[:top_k]) & positive_ids) / max(len(positive_ids), 1))
        ranks = [rank + 1 for rank, fact_id in enumerate(ranked_ids) if fact_id in positive_ids]
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
    return sum(recalls) / len(recalls), sum(reciprocal_ranks) / len(reciprocal_ranks)

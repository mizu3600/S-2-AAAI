# Remote deployment and LoRA run

> Current runtime policy: source and model artifacts for all three structures are retained, but
> only `graph:reified_fact` is enabled by default. The verification notes below describe the
> earlier dual-mode deployment and are retained as historical evidence.

## Deployment

The project is deployed at:

```text
/data/users/liruizhe/qmsxe-rag-codex
```

The isolated API container is `qmsxe-codex-api`. It is limited to 2 CPUs, 8 GB memory and one
thread per numerical backend. The API is bound only to the remote loopback interface:

```text
http://127.0.0.1:18181
```

No existing Unified-RAG-Evaluation process, embedding service, reranker service or container was
stopped or modified. The project uses its own directory, Python package directory, report paths,
container name and port.

Verification performed on the server:

- `/health` returned HTTP 200;
- `mode=graph`, `profile=reified_fact` returned traceable facts and citations;
- the default `mode=hypergraph` continued to return traceable hyperedges and citations;
- all 24 tests passed in the deployment container;
- the one-command delivery pipeline completed.

For a portable build on another host:

```bash
QMSXE_API_PORT=18081 docker compose --profile app up -d --build api
curl http://127.0.0.1:18081/health
```

## BGE-M3 query LoRA

The remote training used the existing read-only model directory
`/data/users/liruizhe/models/bge-m3`. Document/fact embeddings were frozen. LoRA was applied only
to the query encoder's attention query/key/value modules.

Final run:

- 500 HotpotQA examples;
- deterministic 400/100 train/evaluation split;
- 20,795 candidate facts;
- rank 8, alpha 16, dropout 0.05;
- learning rate `2e-5`;
- three epochs with validation selection;
- 1,179,648 trainable parameters out of 568,934,400;
- peak allocated GPU memory: about 2.34 GB;
- selected epoch: 2.

Held-out results:

| Metric | Frozen BGE-M3 | Query LoRA | Change |
|---|---:|---:|---:|
| Recall@20 | 0.7842 | 0.7950 | +0.0108 |
| MRR | 0.7176 | 0.7747 | +0.0571 |

Adapter and tokenizer files are stored remotely at:

```text
/data/users/liruizhe/qmsxe-rag-codex/data/models/bge_m3_query_lora_hotpot500_lr2e5
```

The report copied into the repository is under `reports/lora/`. The earlier high-learning-rate run
was retained as diagnostic evidence but is not selected because held-out Recall@20 decreased.

## Reproduce reports

```bash
python scripts/reproduce_delivery.py \
  --dataset hotpotqa \
  --input-path data/benchmarks/hotpotqa_sample.json \
  --limit 5 \
  --output-dir reports/delivery
```

Smoke tables and figures are engineering validation only. Publication claims require larger fixed
splits and completed isolated external system baselines.

The 50-example remote embedding, grouped, dual-mode and trained-ablation results are summarized in
`reports/REMOTE_SUMMARY.md`. They show that the reified-fact ordinary graph currently has stronger
Recall@20 than the untrained QMSHE branch; this limitation is retained rather than hidden.

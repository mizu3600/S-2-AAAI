# Official open-source baseline protocol

The benchmark pins the four upstream repositories in
`official_baselines.lock.json`. Clone each repository at the recorded commit, then
apply the corresponding compatibility patch in `patches/`. The patches only make
unused optional inference backends optional, or translate Microsoft GraphRAG's
typed JSON request to DeepSeek's supported JSON-object mode. They do not change
retrieval, graph construction, prompts, or ranking algorithms.

For example, from the project root:

```bash
git -C third_party/official_baselines/PathRAG apply \
  ../../patches/pathrag-optional-backends.patch
git -C third_party/official_baselines/HyperGraphRAG apply \
  ../../patches/hypergraphrag-optional-backends.patch
git -C third_party/official_baselines/GraphRAG apply \
  ../../patches/graphrag-deepseek-json-object.patch
```

## Fair comparison unit

- Dataset: the fixed 288-example test partition produced from the first 2,000
  HotpotQA distractor examples.
- Corpus: the same ten passages supplied with each question.
- Gold evidence: passage IDs containing HotpotQA supporting facts.
- Metrics: passage Recall@1/2/5/10, Hit@1, Complete@2/5, and MRR.
- LLM: `deepseek-chat`; embedding: SiliconFlow `BAAI/bge-m3` (1,024 dimensions).
- Extraction gleaning: zero for every official baseline, to make API cost and
  latency bounded. This setting is disclosed because it may reduce graph quality.
- Modes: PathRAG hybrid, LightRAG hybrid, HyperGraphRAG hybrid, and Microsoft
  GraphRAG local search.

Each official implementation receives the ten documents and question through its
public API. The adapters only recover source passage IDs from the official returned
context. QMSxE fact rankings are deduplicated into passage rankings before scoring.

## Images

The official packages have incompatible pandas constraints, so the Dockerfile has
two independent targets:

```bash
docker build --target path-light -t qmsxe-baselines:path-light \
  -f docker/official-baselines.Dockerfile .
docker build --target graphrag -t qmsxe-baselines:graphrag \
  -f docker/official-baselines.Dockerfile .
```

Raw JSONL files checkpoint after every example. Re-running the same command resumes
by skipping IDs already written.

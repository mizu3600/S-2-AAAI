# External GraphRAG baseline protocol

LightRAG, HippoRAG 2, EHRAG and HGRAG/HyperRAG must each be installed in an isolated environment.
Every adapter must consume the exported `Corpus` JSON and must return ranked chunk, fact and entity
IDs. Comparisons are accepted only when corpus, extraction, BGE-M3 embeddings, top-k,
BGE reranker, DeepSeek generator and context budget are identical. Record upstream commit IDs,
configuration, wall time, peak memory and failures in MLflow. No external baseline may modify the
gold corpus or QMSHE evidence graph.


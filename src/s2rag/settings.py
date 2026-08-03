from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bge_embedding_model_path: Path = Path("models/bge-m3")
    bge_reranker_model_path: Path = Path("models/bge-reranker-v2-m3")
    bge_device: str = "auto"
    bge_embedding_device: str | None = None
    bge_reranker_device: str | None = None
    bge_embedding_batch_size: int = 64
    bge_reranker_batch_size: int = 64
    bge_dynamic_batch_wait_ms: float = 5.0
    bge_embedding_micro_batch_max_texts: int = 512
    bge_reranker_micro_batch_max_pairs: int = 320
    bge_reranker_max_length: int = 512
    bge_dtype: str = "float16"
    embedding_cache_dir: Path = Path("data/cache/embeddings")
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_temperature: float = 0.0
    deepseek_max_tokens: int = 256
    deepseek_extraction_max_tokens: int = 8192
    deepseek_retry_initial_seconds: float = 1.0
    deepseek_retry_max_seconds: float = 60.0
    deepseek_max_attempts: int = 8
    deepseek_json_retry_initial_seconds: float = 0.25
    deepseek_json_retry_max_seconds: float = 1.0
    deepseek_json_max_attempts: int = 3
    deepseek_max_concurrency: int = 64
    deepseek_pool_max_connections: int = 96
    deepseek_pool_keepalive_connections: int = 64
    deepseek_response_cache_dir: Path = Path("data/cache/deepseek")
    extraction_batch_max_chars: int = 24000
    extraction_workers: int = 8
    benchmark_example_workers: int = 8
    generation_workers: int = 8
    checkpoint_every: int = 25
    request_timeout: float = 240.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

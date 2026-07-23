from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    siliconflow_api_key: str | None = None
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_embedding_model: str = "BAAI/bge-m3"
    siliconflow_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    database_url: str = "postgresql+psycopg://qmshe:qmshe-local-only@localhost:55432/qmshe"
    mlflow_tracking_uri: str = "http://localhost:5001"
    request_timeout: float = 60.0
    qmshe_enable_hypergraph: bool = False
    qmshe_enable_entity_relation: bool = False
    qmshe_enable_reified_fact: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()

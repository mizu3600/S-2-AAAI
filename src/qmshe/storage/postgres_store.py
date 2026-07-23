import json

from qmshe.ingest.schemas import Corpus
from qmshe.settings import Settings, get_settings

DDL = """
CREATE TABLE IF NOT EXISTS corpora (
  graph_version TEXT PRIMARY KEY,
  encoder_version TEXT NOT NULL,
  spectral_version TEXT NOT NULL,
  index_version TEXT NOT NULL,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS evaluation_runs (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class PostgresMetadataStore:
    def __init__(self, settings: Settings | None = None):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("install the 'infra' extra to use PostgreSQL") from exc
        self.psycopg = psycopg
        self.settings = settings or get_settings()

    def initialize(self) -> None:
        with self.psycopg.connect(self.settings.database_url.replace("postgresql+psycopg", "postgresql")) as connection:
            with connection.cursor() as cursor:
                cursor.execute(DDL)

    def save_corpus_version(
        self, corpus: Corpus, encoder_version: str, spectral_version: str, index_version: str
    ) -> None:
        with self.psycopg.connect(self.settings.database_url.replace("postgresql+psycopg", "postgresql")) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO corpora(graph_version,encoder_version,spectral_version,index_version,payload) "
                    "VALUES (%s,%s,%s,%s,%s) ON CONFLICT(graph_version) DO UPDATE SET payload=EXCLUDED.payload",
                    (corpus.graph_version, encoder_version, spectral_version, index_version,
                     json.dumps(corpus.model_dump(mode="json"))),
                )


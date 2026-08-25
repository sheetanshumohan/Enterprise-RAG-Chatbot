from __future__ import annotations

from typing import Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Environment & API Metadata
    environment: str = "development"  # "development" | "staging" | "production"
    api_title: str = "AI Knowledge Assistant API"
    api_description: str = "Enterprise RAG knowledge assistant backend (DDD / Clean Architecture)."
    api_version: str = "1.0.0"
    cors_origins: Union[list[str], str] = ["*"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            v_clean = v.strip()
            if v_clean.startswith("[") and v_clean.endswith("]"):
                import json
                try:
                    return json.loads(v_clean)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # Postgres (e.g. Neon Serverless Postgres)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/knowledge_assistant"
    db_pool_size: int = 20
    db_max_overflow: int = 20
    db_pool_recycle: int = 300
    db_pool_timeout: int = 30

    @field_validator("database_url", mode="before")
    @classmethod
    def _use_asyncpg_driver(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        # Managed Postgres providers (Neon, Render, Heroku, Supabase, etc.) hand out plain
        # postgres:// or postgresql:// URLs. SQLAlchemy's async engine needs
        # the explicit asyncpg driver in the scheme.
        if v.startswith("postgres://"):
            v = "postgresql+asyncpg://" + v[len("postgres://"):]
        elif v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
            v = "postgresql+asyncpg://" + v[len("postgresql://"):]

        # Clean query parameters for asyncpg compatibility (e.g. Neon's channel_binding)
        from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
        parsed = urlsplit(v)
        if parsed.query:
            query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
            clean_pairs = []
            for k, val in query_pairs:
                if k in ("channel_binding", "gssencmode", "target_session_attrs"):
                    continue
                if k == "sslmode":
                    k = "ssl"
                    val = "require"
                clean_pairs.append((k, val))
            v = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(clean_pairs), parsed.fragment))
        return v

    # Redis / Celery (e.g. Upstash Redis with rediss://)
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_time_limit: int = 600
    celery_task_soft_time_limit: int = 480

    @field_validator("redis_url", "celery_broker_url", "celery_result_backend", mode="before")
    @classmethod
    def _normalize_redis_url(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        # Upstash Redis requires TLS (rediss://) and only supports db 0.
        from urllib.parse import urlsplit, urlunsplit
        parsed = urlsplit(v)
        scheme = parsed.scheme
        if "upstash.io" in parsed.netloc:
            scheme = "rediss"
        path = parsed.path
        if scheme == "rediss" or "upstash.io" in parsed.netloc:
            if path in ("", "/", "/1", "/2", "/3", "/4", "/5"):
                path = "/0"
        return urlunsplit((scheme, parsed.netloc, path, parsed.query, parsed.fragment))

    # Qdrant (Qdrant Cloud or local)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "knowledge_assistant_chunks"
    qdrant_timeout: float = 30.0

    # OpenSearch / Elasticsearch (BM25 keyword search)
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "knowledge_chunks"
    opensearch_timeout: float = 30.0
    opensearch_max_retries: int = 3

    @field_validator("opensearch_url", "qdrant_url", mode="before")
    @classmethod
    def _normalize_endpoint_url(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        v = v.strip()
        while "=" in v and (v.startswith("OPENSEARCH_URL=") or v.startswith("QDRANT_URL=") or v.startswith("DATABASE_URL=")):
            v = v.split("=", 1)[1].strip()
        return v

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # Embeddings
    embedding_provider: str = "openai"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # LLM
    llm_provider: str = "claude"  # "claude" | "openai" | "gemini"
    llm_api_key: str = ""
    llm_model: str | None = None

    # Chunking
    chunk_parent_max_tokens: int = 1500
    chunk_child_max_tokens: int = 300
    chunk_overlap_units: int = 1

    # Reranking & Retrieval
    use_cross_encoder_reranker: bool = False  # requires sentence-transformers weights
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rate_limit_per_minute: int = 60

    # Agentic RAG Pipeline Parameters
    rag_max_retrieval_iterations: int = 3
    rag_token_budget: int = 4000
    rag_top_k_vector: int = 20
    rag_top_k_bm25: int = 20
    rag_top_k_fused: int = 10
    rag_rrf_k: int = 60
    rag_max_generation_tokens: int = 1024
    rag_planner_max_tokens: int = 300
    rag_evaluator_max_tokens: int = 300
    rag_followup_max_tokens: int = 200
    rag_history_messages_limit: int = 12

    # Upload limits
    max_upload_mb: int = 25

    # Observability & Monitoring
    # Sentry Error Tracking
    sentry_dsn: str | None = None
    sentry_traces_sample_rate: float = 0.1
    sentry_profiles_sample_rate: float = 0.1

    # Prometheus Metrics
    prometheus_enabled: bool = True

    # LangSmith Tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "ai-knowledge-assistant"
    langchain_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()



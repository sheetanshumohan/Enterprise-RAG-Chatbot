"""Prometheus metrics instrumentation for AI Knowledge Assistant."""
from __future__ import annotations

import logging
from typing import Any

from prometheus_client import Counter, Histogram

logger = logging.getLogger("knowledge_assistant.observability.metrics")

# Prometheus Metrics Definitions
RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total count of RAG chat questions processed",
    ["status"],
)

RAG_PIPELINE_LATENCY_SECONDS = Histogram(
    "rag_pipeline_latency_seconds",
    "Latency breakdown across RAG pipeline phases in seconds",
    ["phase"],  # e.g., "planner", "retrieval", "evaluator", "generation", "total"
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

DOCUMENTS_INGESTED_TOTAL = Counter(
    "documents_ingested_total",
    "Total count of documents processed in background ingestion",
    ["status", "file_type"],
)

DOCUMENT_INGESTION_LATENCY_SECONDS = Histogram(
    "document_ingestion_latency_seconds",
    "Latency of background document ingestion tasks in seconds",
    ["stage"],  # "extract", "chunk", "embed", "index", "total"
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

LLM_TOKEN_USAGE_TOTAL = Counter(
    "llm_token_usage_total",
    "Estimated total tokens consumed by LLM calls",
    ["provider", "model", "token_type"],  # token_type: "prompt", "completion"
)


def setup_prometheus(app: Any) -> None:
    """Instrument the FastAPI application with Prometheus instrumentator."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=False,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/health", "/metrics"],
        ).instrument(app).expose(app, endpoint="/metrics")
        logger.info("Prometheus metrics initialized at /metrics endpoint.")
    except Exception as e:
        logger.warning("Failed to setup Prometheus FastAPI instrumentator: %s", e)

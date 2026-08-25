"""
Celery app for background/async work -- primarily document ingestion,
which involves text extraction, chunking, embedding API calls, and dual
indexing (Qdrant + BM25). That's too slow to run inline on the upload
request, so `/documents/upload` enqueues a task and returns immediately
with status=PENDING; the client polls or re-fetches the document to see
it flip to INDEXED (or FAILED, with the reason in metadata).
"""
from __future__ import annotations

import os
import ssl
from celery import Celery

from knowledge_assistant.config import settings
from knowledge_assistant.infrastructure.observability.sentry import init_sentry

# Initialize Sentry for Celery worker process
init_sentry(service_type="worker")

broker_url = settings.celery_broker_url
backend_url = settings.celery_result_backend

# 1. Normalize Upstash URLs to rediss:// scheme and remove invalid multi-db indices
if "upstash.io" in broker_url and broker_url.startswith("redis://"):
    broker_url = "rediss://" + broker_url[len("redis://"):]
if "upstash.io" in backend_url and backend_url.startswith("redis://"):
    backend_url = "rediss://" + backend_url[len("redis://"):]

# 2. If broker is using rediss / Upstash, backend must also use rediss (default to broker URL if unset/localhost)
if broker_url.startswith("rediss://") and (backend_url.startswith("redis://localhost") or not backend_url or "upstash.io" not in backend_url):
    backend_url = broker_url

# Override OS env so Celery internal loader does not pick raw redis:// from os.environ
os.environ["CELERY_BROKER_URL"] = broker_url
os.environ["CELERY_RESULT_BACKEND"] = backend_url

celery_app = Celery(
    "knowledge_assistant",
    broker=broker_url,
    backend=backend_url,
)

conf = {
    "broker_url": broker_url,
    "result_backend": backend_url,
    "task_serializer": "json",
    "accept_content": ["json"],
    "result_serializer": "json",
    "task_track_started": True,
    "task_time_limit": settings.celery_task_time_limit,
    "task_soft_time_limit": settings.celery_task_soft_time_limit,

    "worker_max_tasks_per_child": 100,  # recycle workers to bound memory growth
    "worker_prefetch_multiplier": 1,  # fair task distribution across concurrent workers
    "task_acks_late": True,  # ensure task isn't lost if worker crashes during ingestion
    "broker_connection_retry_on_startup": True,  # resilient startup when connecting to Upstash
    "broker_transport_options": {
        "socket_timeout": 30.0,
        "socket_connect_timeout": 30.0,
        "socket_keepalive": True,
        "retry_on_timeout": True,
        "visibility_timeout": 3600,
    },
    "result_backend_transport_options": {
        "socket_timeout": 30.0,
        "socket_connect_timeout": 30.0,
        "retry_on_timeout": True,
    },
}

if broker_url.startswith("rediss://"):
    conf["broker_use_ssl"] = {"ssl_cert_reqs": ssl.CERT_NONE}
if backend_url.startswith("rediss://"):
    conf["redis_backend_use_ssl"] = {"ssl_cert_reqs": ssl.CERT_NONE}

celery_app.conf.update(conf)

celery_app.autodiscover_tasks(["knowledge_assistant.infrastructure.tasks"])


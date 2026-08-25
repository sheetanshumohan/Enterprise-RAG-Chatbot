import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from knowledge_assistant.config import settings
from knowledge_assistant.infrastructure.db.session import SessionLocal, init_db
from knowledge_assistant.infrastructure.observability import (
    init_langsmith,
    init_sentry,
    setup_logging,
    setup_prometheus,
)
from knowledge_assistant.interfaces.api.routers import auth, chat, collections, documents

# Configure structured logging
setup_logging(level="INFO", json_format=(settings.environment == "production"))
logger = logging.getLogger("knowledge_assistant.api")

# Initialize Observability (Sentry & LangSmith)
init_sentry(service_type="api")
init_langsmith()

# Async Redis client for distributed rate limiting (Upstash or local)
_redis_client: aioredis.Redis | None = None
_in_memory_log: dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_client
    try:
        redis_kwargs = {"socket_timeout": 3.0, "socket_connect_timeout": 3.0}
        if settings.redis_url.startswith("rediss://"):
            redis_kwargs["ssl_cert_reqs"] = "none"
        _redis_client = aioredis.from_url(settings.redis_url, **redis_kwargs)
        await asyncio.wait_for(_redis_client.ping(), timeout=3.0)
        logger.info("Connected to Redis for distributed rate limiting.")
    except Exception as e:
        logger.warning("Could not connect to Redis (%s); fallback to in-memory rate limiting.", e)
        _redis_client = None

    try:
        await asyncio.wait_for(init_db(), timeout=10.0)
    except Exception as e:
        logger.warning("Database initialization check (%s)", e)

    yield

    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass


app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
)


# Prometheus Metrics Instrumentation
if settings.prometheus_enabled:
    setup_prometheus(app)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_audit(request: Request, call_next):
    # Correlation ID
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    # Exclude health and metrics from rate limiting & verbose logs
    if request.url.path in ("/health", "/health/live", "/health/ready", "/metrics"):
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    client_ip = request.client.host if request.client else "unknown"

    if await check_rate_limit(client_ip):
        res = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Try again shortly.", "request_id": request_id},
        )
        res.headers["X-Request-ID"] = request_id
        return res

    start = time.monotonic()
    try:
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "request method=%s path=%s status=%s ip=%s duration_ms=%.1f request_id=%s",
            request.method, request.url.path, response.status_code, client_ip, duration_ms, request_id,
            extra={"request_id": request_id},
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        duration_ms = (time.monotonic() - start) * 1000
        logger.exception(
            "unhandled_exception method=%s path=%s duration_ms=%.1f request_id=%s error=%s",
            request.method, request.url.path, duration_ms, request_id, exc,
            extra={"request_id": request_id},
        )
        res = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "request_id": request_id},
        )
        res.headers["X-Request-ID"] = request_id
        return res


async def check_rate_limit(client_ip: str) -> bool:
    """Check if request exceeds per-minute rate limit. Returns True if exceeded."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return False

    now = time.time()
    # 1. Distributed Redis sliding window
    if _redis_client is not None:
        try:
            key = f"rate_limit:{client_ip}"
            pipe = _redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, now - 60)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, 65)
            results = await pipe.execute()
            count = results[2]
            return count > limit
        except Exception as e:
            logger.debug("Redis rate-limit check failed (%s); fallback to memory.", e)

    # 2. In-memory fallback
    mono_now = time.monotonic()
    window = _in_memory_log[client_ip]
    while window and mono_now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        return True
    window.append(mono_now)
    return False


app.include_router(auth.router)
app.include_router(collections.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
@app.get("/health/live")
async def liveness():
    """Liveness probe: returns 200 if FastAPI is running."""
    return {
        "status": "ok",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }


@app.get("/health/ready")
async def readiness():
    """Readiness probe: validates database, Redis, and Qdrant connectivity."""
    checks: dict[str, str] = {}
    is_ready = True

    # 1. Check Postgres / Neon
    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {str(e)[:100]}"
        is_ready = False

    # 2. Check Redis / Upstash
    if _redis_client is not None:
        try:
            await _redis_client.ping()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"error: {str(e)[:100]}"
            is_ready = False
    else:
        checks["redis"] = "not_connected (in-memory mode)"

    # 3. Check Qdrant
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5.0)
        await client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {str(e)[:100]}"
        is_ready = False

    # 4. Check OpenSearch / Elasticsearch
    try:
        from knowledge_assistant.infrastructure.search.opensearch_index import OpenSearchKeywordIndex
        os_index = OpenSearchKeywordIndex(settings.opensearch_url, settings.opensearch_index)
        os_client = os_index._get_client()
        await os_client.ping()
        checks["opensearch"] = "ok"
    except Exception as e:
        checks["opensearch"] = f"error: {str(e)[:100]}"
        is_ready = False

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content={"ready": is_ready, "checks": checks, "environment": settings.environment},
    )



FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY apps/backend/pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi "uvicorn[standard]" gunicorn sqlalchemy asyncpg alembic pydantic pydantic-settings \
    "python-jose[cryptography]" "passlib[bcrypt]" "bcrypt==4.0.1" python-multipart \
    qdrant-client opensearch-py aiohttp rank-bm25 redis celery anthropic openai google-generativeai numpy pypdf python-docx \
    markdown-it-py tiktoken httpx tenacity email-validator \
    "sentry-sdk[fastapi,celery]>=2.13.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "prometheus-client>=0.20.0" \
    "langsmith>=0.1.100"

COPY apps/backend/src ./src
COPY apps/backend/alembic.ini ./alembic.ini
COPY apps/backend/migrations ./migrations

ENV PYTHONPATH=/app/src
ENV PORT=8000
ENV WEB_CONCURRENCY=1

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn knowledge_assistant.interfaces.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

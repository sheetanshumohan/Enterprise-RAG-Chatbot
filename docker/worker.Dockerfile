FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY apps/backend/pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    sqlalchemy asyncpg pydantic pydantic-settings \
    qdrant-client opensearch-py aiohttp rank-bm25 redis celery anthropic openai google-generativeai numpy pypdf python-docx \
    markdown-it-py tiktoken httpx tenacity \
    "sentry-sdk[celery]>=2.13.0" \
    "prometheus-client>=0.20.0" \
    "langsmith>=0.1.100"


COPY apps/backend/src ./src

ENV PYTHONPATH=/app/src

CMD ["celery", "-A", "knowledge_assistant.infrastructure.tasks.celery_app", "worker", "--loglevel=info"]

"""LangSmith tracing configuration and helper decorators."""
from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Any

from knowledge_assistant.config import settings

logger = logging.getLogger("knowledge_assistant.observability.langsmith")


def init_langsmith() -> None:
    """Export environment variables required by LangSmith client & tracing."""
    if settings.langchain_tracing_v2 or settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        if settings.langchain_project:
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        if settings.langchain_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        logger.info("LangSmith tracing enabled for project: %s", settings.langchain_project)


def trace(run_type: str = "chain", name: str | None = None, tags: list[str] | None = None):
    """Decorator to trace an async or sync function with LangSmith if available, else no-op."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        try:
            from langsmith import traceable
            return traceable(run_type=run_type, name=name or fn.__name__, tags=tags or ["rag"])(fn)
        except Exception:
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await fn(*args, **kwargs)

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)

            import asyncio
            return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper

    return decorator

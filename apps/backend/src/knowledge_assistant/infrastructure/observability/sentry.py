"""Sentry integration for FastAPI and Celery error tracking."""
from __future__ import annotations

import logging
from typing import Literal

from knowledge_assistant.config import settings

logger = logging.getLogger("knowledge_assistant.observability.sentry")


def init_sentry(service_type: Literal["api", "worker"] = "api") -> None:
    """Initialize Sentry if DSN is configured."""
    if not settings.sentry_dsn:
        logger.info("Sentry DSN not configured; skipping Sentry initialization.")
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.celery import CeleryIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

        integrations = [
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
            SqlalchemyIntegration(),
        ]

        if service_type == "api":
            integrations.append(FastApiIntegration())
        elif service_type == "worker":
            integrations.append(CeleryIntegration())

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            traces_sample_rate=settings.sentry_traces_sample_rate,
            profiles_sample_rate=settings.sentry_profiles_sample_rate,
            integrations=integrations,
            send_default_pii=False,
        )
        logger.info("Sentry initialized successfully for %s in %s environment.", service_type, settings.environment)
    except Exception as e:
        logger.warning("Failed to initialize Sentry: %s", e)

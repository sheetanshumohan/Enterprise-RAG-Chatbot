from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from knowledge_assistant.config import settings
from knowledge_assistant.infrastructure.db.models import Base

engine_kwargs: dict = {"pool_pre_ping": True}
if not settings.database_url.startswith("sqlite"):
    engine_kwargs.update({
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_recycle": settings.db_pool_recycle,
        "pool_timeout": settings.db_pool_timeout,
    })
    if "asyncpg" in settings.database_url:
        # Crucial for Neon Serverless Postgres & PgBouncer pooled endpoints:
        # disable prepared statement cache to avoid collisions across pooled connections
        engine_kwargs["connect_args"] = {
            "prepared_statement_cache_size": 0,
            "statement_cache_size": 0,
        }


engine = create_async_engine(settings.database_url, **engine_kwargs)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)



async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session

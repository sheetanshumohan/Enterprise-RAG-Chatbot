"""
Keep-Alive & Heartbeat Automation Script.

Pings free-tier cloud clusters (Neon Postgres, Upstash Redis, Qdrant Cloud, Render)
every 48 hours to prevent inactivity suspension.
"""
import asyncio
import os
import sys


async def ping_deployed_backend(backend_url: str) -> bool:
    """Pings deployed backend's /health/ready and /health endpoints."""
    import httpx

    print(f"[*] Pinging deployed backend: {backend_url}")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{backend_url.rstrip('/')}/health/ready")
            print(f"[+] Backend /health/ready response [{resp.status_code}]: {resp.text}")

            live_resp = await client.get(f"{backend_url.rstrip('/')}/health")
            print(f"[+] Backend /health response [{live_resp.status_code}]: {live_resp.text}")
            return True
    except Exception as e:  # noqa: BLE001
        print(f"[!] Warning: Failed to ping backend URL ({backend_url}): {e}")
        return False


async def ping_postgres(db_url: str) -> bool:
    """Executes a lightweight query (SELECT 1) on Neon Postgres."""
    print("[*] Pinging Neon Postgres...")
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(db_url, pool_pre_ping=True)
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            val = result.scalar()
            print(f"[+] Neon Postgres keep-alive query successful (SELECT {val}).")
        await engine.dispose()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[!] Warning: Neon Postgres ping failed: {e}")
        return False


async def ping_redis(redis_url: str) -> bool:
    """Executes PING on Upstash / Redis."""
    print("[*] Pinging Upstash Redis...")
    try:
        import redis.asyncio as aioredis

        redis_kwargs = {}
        if redis_url.startswith("rediss://"):
            redis_kwargs["ssl_cert_reqs"] = "none"

        client = aioredis.from_url(redis_url, **redis_kwargs)
        pong = await client.ping()
        print(f"[+] Upstash Redis keep-alive PING successful (PONG={pong}).")
        await client.aclose()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[!] Warning: Upstash Redis ping failed: {e}")
        return False


async def ping_qdrant(qdrant_url: str, api_key: str | None) -> bool:
    """Pings Qdrant Cloud cluster."""
    print(f"[*] Pinging Qdrant Cloud at {qdrant_url}...")
    try:
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(url=qdrant_url, api_key=api_key or None, timeout=15.0)
        collections = await client.get_collections()
        col_names = [c.name for c in collections.collections]
        print(f"[+] Qdrant Cloud keep-alive successful. Found collections: {col_names}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[!] Warning: Qdrant Cloud ping failed: {e}")
        return False


async def main():
    print("================================================================")
    print(" Cloud Cluster Keep-Alive / Heartbeat Execution")
    print("================================================================")

    backend_url = os.environ.get("BACKEND_URL") or os.environ.get("KNOWLEDGE_ASSISTANT_API_URL")
    database_url = os.environ.get("DATABASE_URL")
    redis_url = os.environ.get("REDIS_URL")
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    tasks = []

    # 1. Ping deployed backend if URL provided
    if backend_url:
        tasks.append(ping_deployed_backend(backend_url))

    # 2. Direct Postgres ping if DATABASE_URL provided
    if database_url:
        tasks.append(ping_postgres(database_url))

    # 3. Direct Redis ping if REDIS_URL provided
    if redis_url:
        tasks.append(ping_redis(redis_url))

    # 4. Direct Qdrant ping if QDRANT_URL provided
    if qdrant_url:
        tasks.append(ping_qdrant(qdrant_url, qdrant_api_key))

    if not tasks:
        print("[!] No service URLs or connection strings found in environment variables.")
        print("[!] Set BACKEND_URL, DATABASE_URL, REDIS_URL, or QDRANT_URL in secrets/env.")
        sys.exit(0)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    print("================================================================")
    print(f" Keep-Alive Heartbeat Finished. Tasks executed: {len(results)}")
    print("================================================================")


if __name__ == "__main__":
    asyncio.run(main())

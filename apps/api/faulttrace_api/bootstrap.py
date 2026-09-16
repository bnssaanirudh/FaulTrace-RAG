"""Idempotent local/demo database bootstrap used by Docker startup."""

from __future__ import annotations

import asyncio

from faulttrace_api.database import get_session_factory, init_db
from faulttrace_api.models import SeedDemoRequest
from faulttrace_api.routes.demo import seed_demo


async def bootstrap() -> None:
    init_db()
    session = get_session_factory()()
    try:
        await seed_demo(
            SeedDemoRequest(seed=42, scales=[10, 50, 200, 1000], overwrite=False),
            session,
        )
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(bootstrap())

import contextlib

from langgraph.store.postgres.aio import AsyncPostgresStore

from app.config import get_settings

settings = get_settings()

@contextlib.asynccontextmanager
async def generate_store():
    """Yield the PostgreSQL profile Store without an unused vector index.

    This will be discovered by LangGraph/LangSmith when you point `langgraph.json`'s
    `store.path` to `app.agent.store:generate_store`.
    """
    async with AsyncPostgresStore.from_conn_string(
        settings.database_url_sync,
    ) as store:
        # Ensure tables/indexes exist on first run
        await store.setup()
        yield store

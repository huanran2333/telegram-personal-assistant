"""Explicit ownership of OpenAI-compatible HTTP connections.

LangChain's default ChatOpenAI HTTP client is cached across model instances.
An async client first used on one event loop must not be reused after that loop
closes. The application owns one client pair for its lifespan; isolated evals
and real-provider tests own a shorter explicit scope instead.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar


class ProviderClientOwner:
    def __init__(self) -> None:
        self._sync_client = None
        self._async_client = None
        self._closed = False

    def openai_http_clients(self):
        if self._closed:
            raise RuntimeError("Provider client owner is closed")
        if self._async_client is None:
            from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

            self._sync_client = DefaultHttpxClient()
            self._async_client = DefaultAsyncHttpxClient()
        return self._sync_client, self._async_client

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._async_client is not None:
                await self._async_client.aclose()
        finally:
            if self._sync_client is not None:
                self._sync_client.close()


_scoped_owner: ContextVar[ProviderClientOwner | None] = ContextVar(
    "provider_client_owner", default=None)
_process_owner: ProviderClientOwner | None = None


def bind_process_owner(owner: ProviderClientOwner | None) -> None:
    global _process_owner
    if owner is not None and _process_owner is not None and _process_owner is not owner:
        raise RuntimeError("A provider client owner is already bound")
    _process_owner = owner


def current_provider_owner() -> ProviderClientOwner:
    owner = _process_owner or _scoped_owner.get()
    if owner is None:
        raise RuntimeError("Provider client owner must be bound before creating an LLM")
    return owner


@asynccontextmanager
async def provider_client_scope():
    owner = ProviderClientOwner()
    token = _scoped_owner.set(owner)
    try:
        yield owner
    finally:
        _scoped_owner.reset(token)
        await owner.aclose()


async def run_with_provider_clients(coroutine):
    """CLI/evaluation entrypoint: close clients before asyncio.run closes its loop."""
    async with provider_client_scope():
        return await coroutine

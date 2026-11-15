import pytest

from app.agent.provider_clients import (
    ProviderClientOwner, bind_process_owner, current_provider_owner,
    provider_client_scope,
)


@pytest.mark.asyncio
async def test_scoped_clients_are_reused_only_during_their_owner_lifetime():
    async with provider_client_scope() as first:
        sync_client, async_client = first.openai_http_clients()
        assert first.openai_http_clients() == (sync_client, async_client)
        assert current_provider_owner() is first
        assert not async_client.is_closed
    assert async_client.is_closed
    assert sync_client.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        first.openai_http_clients()
    async with provider_client_scope() as second:
        assert second is not first
        assert second.openai_http_clients()[1] is not async_client


@pytest.mark.asyncio
async def test_process_owner_is_explicit_and_closed_on_shutdown():
    owner = ProviderClientOwner()
    bind_process_owner(owner)
    try:
        assert current_provider_owner() is owner
        with pytest.raises(RuntimeError, match="already bound"):
            bind_process_owner(ProviderClientOwner())
        _, async_client = owner.openai_http_clients()
    finally:
        bind_process_owner(None)
        await owner.aclose()
    assert async_client.is_closed


@pytest.mark.asyncio
async def test_scoped_client_closes_when_entrypoint_raises():
    async_client = None
    with pytest.raises(ValueError, match="synthetic failure"):
        async with provider_client_scope() as owner:
            _, async_client = owner.openai_http_clients()
            raise ValueError("synthetic failure")
    assert async_client is not None and async_client.is_closed

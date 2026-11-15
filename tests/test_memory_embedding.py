"""The remote embedding boundary must fail before private text leaves process."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.memory_embedding import (
    EmbeddingEgressPolicy, OpenAICompatibleEmbeddingProvider, _validated_vector,
)


def test_unapproved_http_and_https_cannot_embed_private_transcripts():
    for url in ("http://inference.example/v1", "https://inference.example/v1"):
        with pytest.raises(PermissionError):
            EmbeddingEgressPolicy(url).assert_allowed(private_data=True)


def test_synthetic_http_is_permitted_but_private_requires_local_or_approved_https():
    EmbeddingEgressPolicy("http://inference.example/v1").assert_allowed(private_data=False)
    EmbeddingEgressPolicy("http://127.0.0.1:8000/v1").assert_allowed(private_data=True)
    EmbeddingEgressPolicy("https://inference.example/v1", private_data_approved=True).assert_allowed(
        private_data=True)
    with pytest.raises(PermissionError):
        EmbeddingEgressPolicy("http://inference.example/v1", private_data_approved=True).assert_allowed(
            private_data=True)


def test_user_controlled_vllm_http_requires_complete_attestation():
    url = "http://self-hosted.example/v1"
    with pytest.raises(PermissionError):
        EmbeddingEgressPolicy(url, local_self_hosted=True).assert_allowed(private_data=True)
    with pytest.raises(PermissionError):
        EmbeddingEgressPolicy(url, local_self_hosted=True,
                              no_unknown_proxy=True).assert_allowed(private_data=True)
    EmbeddingEgressPolicy(url, local_self_hosted=True, no_unknown_proxy=True,
                          body_logging_disabled=True).assert_allowed(private_data=True)


def test_bad_vectors_rejected():
    for vector in ([0.0, 0.0], [float("nan"), 1.0], [1.0]):
        with pytest.raises(ValueError):
            _validated_vector(vector, 2)


@pytest.mark.asyncio
async def test_provider_itself_rejects_private_http_before_network():
    policy = EmbeddingEgressPolicy("http://inference.example/v1", private_data_approved=True)
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=policy.base_url, model_id="some-model", dimension=2,
        egress=policy, private_data=True)
    with pytest.raises(PermissionError):
        await provider.embed(["private conversation text"])


@pytest.mark.asyncio
async def test_provider_requests_configured_ollama_dimension():
    policy = EmbeddingEgressPolicy("http://127.0.0.1:11434/v1", local_self_hosted=True)
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=policy.base_url, model_id="qwen3-embedding:8b", dimension=3,
        egress=policy, private_data=True)
    provider._client.embeddings.create = AsyncMock(return_value=SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[0.1, 0.2, 0.3])],
        usage=SimpleNamespace(total_tokens=4)))
    try:
        assert await provider.embed(["synthetic query"]) == [[0.1, 0.2, 0.3]]
        provider._client.embeddings.create.assert_awaited_once_with(
            model="qwen3-embedding:8b", input=["synthetic query"], dimensions=3)
    finally:
        await provider.aclose()

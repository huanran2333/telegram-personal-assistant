from app.agent import nodes
from app.config import Settings
import pytest
from app.agent.provider_clients import provider_client_scope


@pytest.mark.asyncio
async def test_openai_compatible_client_uses_configured_url_key_and_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-process-key")
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        llm_model="provider-model-id",
        openai_base_url=" https://provider.example/v1 ",
        openai_api_key="configured-test-key",
        memory_enabled=False,
    )
    monkeypatch.setattr(nodes, "settings", settings)

    async with provider_client_scope():
        client = nodes._build_chat_llm("openai", settings.llm_model)

        assert client.model_name == "provider-model-id"
        assert str(client.openai_api_base).rstrip("/") == "https://provider.example/v1"
        assert client.openai_api_key.get_secret_value() == "configured-test-key"


@pytest.mark.asyncio
async def test_meeting_structured_generation_does_not_spend_output_budget_on_qwen_reasoning(monkeypatch):
    settings = Settings(
        _env_file=None, llm_provider="openai", llm_model="Qwen3.8-test",
        openai_base_url="http://127.0.0.1:9000/v1",
        openai_api_key="synthetic-key", memory_enabled=False,
    )
    monkeypatch.setattr(nodes, "settings", settings)
    async with provider_client_scope():
        meeting = nodes._build_chat_llm("openai", settings.llm_model,
                                        meeting_structured=True)
        ordinary = nodes._build_chat_llm("openai", settings.llm_model)
    assert meeting.reasoning_effort == "none"
    assert ordinary.reasoning_effort is None


def test_pgvector_mode_cannot_be_reenabled_by_deployment_config():
    with pytest.raises(ValueError, match="Only source-scoped Transcript memory is enabled"):
        Settings(
            _env_file=None, llm_provider="openai", llm_model="test-model",
            openai_api_key="synthetic-key", memory_enabled=True,
            memory_embedding_base_url="http://127.0.0.1:8000/v1",
            memory_embedding_model="synthetic-embedding",
            memory_embedding_dimension=4001,
            memory_vector_backend="pgvector_halfvec_exact",
            memory_mode="legacy_r3",
        )


def test_memory_defaults_to_source_lookup_and_legacy_embedding_remains_validated():
    assert Settings.model_fields["memory_enabled"].default is True
    assert Settings.model_fields["memory_mode"].default == "transcript_lexical"
    assert Settings.model_fields["memory_vector_backend"].default == "pgvector_halfvec_exact"
    assert Settings.model_fields["memory_embedding_base_url"].default == "http://127.0.0.1:11434/v1"
    assert Settings.model_fields["memory_embedding_model"].default == "qwen3-embedding:8b"
    assert Settings.model_fields["memory_embedding_dimension"].default == 2560
    assert Settings.model_fields["memory_embedding_local_self_hosted"].default is True
    Settings(_env_file=None, llm_provider="openai", llm_model="test-model",
             openai_api_key="synthetic-key", memory_embedding_base_url="",
             memory_embedding_model="")
    with pytest.raises(ValueError, match="Only source-scoped Transcript memory is enabled"):
        Settings(_env_file=None, llm_provider="openai", llm_model="test-model",
                 openai_api_key="synthetic-key", memory_embedding_base_url="",
                 memory_embedding_model="", memory_mode="legacy_r3")


@pytest.mark.parametrize("seconds", [0, -1, 301])
def test_shutdown_drain_deadline_is_bounded(seconds):
    with pytest.raises(ValueError, match="SHUTDOWN_DRAIN_SECONDS"):
        Settings(_env_file=None, llm_provider="openai", llm_model="test-model",
                 openai_api_key="synthetic-key", shutdown_drain_seconds=seconds,
                 memory_enabled=False)

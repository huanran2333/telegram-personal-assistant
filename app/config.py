from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pydantic import model_validator
from sqlalchemy.engine import make_url
from urllib.parse import urlsplit
import ipaddress

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # Telegram
    telegram_bot_token: str = ""
    telegram_webhook_url: str = ""
    # Explicit operator attestation after durable-queue and schedule review.
    # Without it, startup must never call Bot API or start R4 send recovery.
    telegram_live_start_preflight_ack: bool = False

    # LLM
    llm_provider: str = "nvidia"
    llm_model: str = "claude-sonnet-4-20250514"

    nvidia_api_key: str = ""
    nvidia_api_endpoint: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    # Optional OpenAI-compatible model failover for the production Agent Loop.
    llm_fallback_1_base_url: str = ""
    llm_fallback_1_model: str = ""
    llm_fallback_1_api_key: str = ""
    llm_fallback_2_base_url: str = ""
    llm_fallback_2_model: str = ""
    llm_fallback_2_api_key: str = ""
    llm_primary_max_attempts: int = 2
    llm_primary_timeout_seconds: float = 12.0
    llm_fallback_timeout_seconds: float = 45.0
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/assistant"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/assistant"

    # Redis
    # redis_url: str = "redis://localhost:6379/0"

    # App
    app_env: str = "development"
    secret_key: str = "change_me_in_production"

    # Tavily API key (required for web search)
    tavily_api_key: str = ""

    google_credentials_file: str = "./gcp-oauth.keys.json"
    google_calendar_enabled: bool = True
    # The current Google MCP process owns one OAuth account, not one per Bot user.
    google_calendar_owner_telegram_id: str = ""
    # A separate, single-account Gmail OAuth grant. Never share Calendar MCP's token.
    gmail_enabled: bool = False
    gmail_credentials_file: str = ""
    gmail_token_file: str = ""
    gmail_owner_telegram_id: str = ""
    default_timezone: str = "Asia/Shanghai"
    # Foreground live start still requires operator confirmation and queue preflight.
    scheduler_enabled: bool = True
    daily_rundown_enabled: bool = True
    # Optional operator scope for attended live tests. Empty means all users.
    reminder_allowed_telegram_ids: str = ""
    daily_rundown_allowed_telegram_ids: str = ""
    # During an attended test, return 503 for other inbound users so Telegram
    # may retry them after the scoped session ends; empty means normal ingress.
    telegram_allowed_ingress_ids: str = ""
    daily_brief_hour: int = 8
    # Graceful opportunity, not a durability guarantee. Expired leases are
    # recovered by PostgreSQL-backed workers after restart.
    shutdown_drain_seconds: float = 30.0
    # Product/privacy policy after *terminal* eligibility; never a retry TTL.
    messaging_terminal_retention_days: int = 30
    # Web is opt-in until its HTTPS origin and operator-bound credential exist.
    web_enabled: bool = False
    web_public_origin: str = ""
    web_session_hours: int = 24

    # Current memory uses Profile Store and source-scoped Transcript lookup.
    # Old embedding settings remain parseable for historical experiment/config compatibility.
    # Isolated tests can explicitly set MEMORY_ENABLED=false.
    memory_enabled: bool = True
    memory_embedding_base_url: str = "http://127.0.0.1:11434/v1"
    memory_embedding_model: str = "qwen3-embedding:8b"
    memory_embedding_dimension: int = 2560
    memory_embedding_api_key: str = "not-required"
    memory_embedding_local_self_hosted: bool = True
    memory_embedding_no_unknown_proxy: bool = True
    memory_embedding_body_logging_disabled: bool = False
    memory_context_token_budget: int = 8192
    # The bounded JSON cosine path remains an explicit degraded fallback.
    # The legacy vector backend setting may still exist in a private .env;
    # it does not opt the production Agent back into automatic extraction.
    memory_mode: str = "transcript_lexical"
    memory_vector_backend: str = "pgvector_halfvec_exact"
    # R6 Meeting Prep selection policy, not source/authority correctness.
    meeting_prep_evidence_token_budget: int = 2800
    meeting_prep_gmail_lookback_days: int = 30
    meeting_prep_gmail_candidate_limit: int = 5

    @model_validator(mode="after")
    def validate_llm_settings(self) -> "Settings":
        provider = (self.llm_provider or "").strip().lower()
        allowed_providers = {"nvidia", "anthropic", "openai", "gemini"}

        if provider not in allowed_providers:
            raise ValueError(
                "Invalid llm_provider. Supported values: nvidia, anthropic, openai, gemini"
            )

        if not (self.llm_model or "").strip():
            raise ValueError("LLM_MODEL must be set and non-empty")

        provider_key_map = {
            "nvidia": self.nvidia_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
        }

        if not (provider_key_map[provider] or "").strip():
            env_key = {
                "nvidia": "NVIDIA_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }[provider]
            raise ValueError(f"Missing API key for provider '{provider}'. Set {env_key}.")

        self.llm_provider = provider
        self.llm_model = self.llm_model.strip()
        self.openai_base_url = self.openai_base_url.strip()
        if not 1 <= self.llm_primary_max_attempts <= 3:
            raise ValueError("LLM_PRIMARY_MAX_ATTEMPTS must be between 1 and 3")
        if not 1 <= self.llm_primary_timeout_seconds <= 120:
            raise ValueError("LLM_PRIMARY_TIMEOUT_SECONDS must be in [1, 120]")
        if not 1 <= self.llm_fallback_timeout_seconds <= 180:
            raise ValueError("LLM_FALLBACK_TIMEOUT_SECONDS must be in [1, 180]")
        for index in (1, 2):
            url_key = f"llm_fallback_{index}_base_url"
            model_key = f"llm_fallback_{index}_model"
            key_key = f"llm_fallback_{index}_api_key"
            base_url = getattr(self, url_key).strip().rstrip("/")
            model = getattr(self, model_key).strip()
            api_key = getattr(self, key_key).strip()
            if any((base_url, model, api_key)) and not all((base_url, model, api_key)):
                raise ValueError(f"LLM_FALLBACK_{index} requires URL, model and API key")
            if base_url:
                if provider != "openai":
                    raise ValueError("LLM fallbacks require LLM_PROVIDER=openai")
                parts = urlsplit(base_url)
                host = parts.hostname or ""
                try:
                    local_host = ipaddress.ip_address(host).is_private
                except ValueError:
                    local_host = host == "localhost"
                if (parts.scheme not in {"http", "https"} or not host or
                        parts.username or parts.password or parts.query or parts.fragment or
                        (parts.scheme == "http" and not local_host)):
                    raise ValueError(f"LLM_FALLBACK_{index} URL must use HTTPS or a private HTTP host")
            setattr(self, url_key, base_url)
            setattr(self, model_key, model)
            setattr(self, key_key, api_key)
        if not 0 <= self.daily_brief_hour <= 23:
            raise ValueError("DAILY_BRIEF_HOUR must be between 0 and 23")
        if not 0 < self.shutdown_drain_seconds <= 300:
            raise ValueError("SHUTDOWN_DRAIN_SECONDS must be in (0, 300]")
        if not 1 <= self.messaging_terminal_retention_days <= 3650:
            raise ValueError("MESSAGING_TERMINAL_RETENTION_DAYS must be in [1, 3650]")
        if self.memory_enabled:
            if self.memory_mode != "transcript_lexical":
                raise ValueError("Only source-scoped Transcript memory is enabled")
            if self.memory_mode == "legacy_r3" and (
                    not self.memory_embedding_base_url or not self.memory_embedding_model):
                raise ValueError("Memory requires explicit embedding endpoint and model")
            if self.memory_context_token_budget < 2048 or (
                    self.memory_mode == "legacy_r3" and
                    self.memory_embedding_dimension < 1):
                raise ValueError("Memory embedding dimension/context budget is invalid")
            if self.memory_vector_backend not in {"json_bounded", "pgvector_halfvec_exact"}:
                raise ValueError("Unsupported Memory vector backend")
            if (self.memory_vector_backend == "pgvector_halfvec_exact" and
                    self.memory_embedding_dimension > 4000):
                raise ValueError("pgvector halfvec supports at most 4000 dimensions")
        if not 512 <= self.meeting_prep_evidence_token_budget <= 16000:
            raise ValueError("Invalid Meeting Prep evidence token budget")
        if not 1 <= self.meeting_prep_gmail_lookback_days <= 365:
            raise ValueError("Invalid Meeting Prep Gmail lookback")
        if not 1 <= self.meeting_prep_gmail_candidate_limit <= 10:
            raise ValueError("Invalid Meeting Prep Gmail candidate limit")
        if self.telegram_live_start_preflight_ack:
            async_url = make_url(self.database_url)
            sync_url = make_url(self.database_url_sync)
            expected = ("127.0.0.1", 5432, "personal_assistant",
                        "personal_assistant_app")
            for url in (async_url, sync_url):
                if (url.host, url.port, url.database, url.username) != expected:
                    raise ValueError(
                        "Live Telegram start requires dedicated production PostgreSQL")
            if (self.app_env != "production"
                    or self.secret_key == "change_me_in_production"
                    or len(self.secret_key) < 32
                    or not self.web_public_origin.startswith("https://")):
                raise ValueError("Live Telegram start requires reviewed production web settings")
        return self
@lru_cache()
def get_settings() -> Settings:
    return Settings()

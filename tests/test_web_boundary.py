"""V3.1 deterministic security and replay-contract checks (no provider)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import run_agent
from app.web.auth import (password_hash, verify_password, require_origin,
                          require_bootstrap_origin, validate_web_configuration)
from app.web.api import router as web_router


def test_argon2id_password_verification_and_no_plaintext():
    encoded = password_hash("a sufficiently long local password")
    assert encoded.startswith("$argon2id$")
    assert "sufficiently" not in encoded
    assert verify_password("a sufficiently long local password", encoded)
    assert not verify_password("wrong password", encoded)
    assert password_hash("a sufficiently long local password") != encoded


def test_web_configuration_fails_closed_without_https_or_secret():
    settings = SimpleNamespace(web_enabled=True, web_public_origin="http://localhost",
                               secret_key="valid-secret")
    with pytest.raises(ValueError):
        validate_web_configuration(settings)
    settings.web_public_origin = "https://workbench.example"
    settings.secret_key = "change_me_in_production"
    with pytest.raises(ValueError):
        validate_web_configuration(settings)
    settings.secret_key = "valid-secret"
    validate_web_configuration(settings)


def test_origin_and_bootstrap_same_origin_rules():
    settings = SimpleNamespace(web_public_origin="https://workbench.example")
    request = SimpleNamespace(headers={"origin": "https://attacker.example"})
    with pytest.raises(HTTPException):
        require_origin(request, settings)
    request.headers = {"sec-fetch-site": "cross-site"}
    with pytest.raises(HTTPException):
        require_bootstrap_origin(request, settings)
    request.headers = {"sec-fetch-site": "same-origin"}
    require_bootstrap_origin(request, settings)


@pytest.mark.asyncio
async def test_web_api_fails_closed_when_web_auth_is_not_ready():
    application = FastAPI()
    application.include_router(web_router)
    application.state.web_ready = False
    async with AsyncClient(transport=ASGITransport(app=application),
                           base_url="https://workbench.test") as client:
        for path in ("/api/v3/me", "/api/v3/auth/bootstrap"):
            response = await client.get(path)
            assert response.status_code == 503


@pytest.mark.asyncio
async def test_web_api_auth_does_not_depend_on_retired_web_worker():
    application = FastAPI()
    application.include_router(web_router)
    application.state.web_ready = True
    from app.web.api import _ready
    _ready(SimpleNamespace(app=application))


@pytest.mark.asyncio
async def test_public_health_has_no_web_agent_worker_component(monkeypatch):
    import json
    from app import main as app_main
    from app.runtime import RuntimeCapabilities

    runtime = RuntimeCapabilities()
    for component in list(runtime.components):
        runtime.mark_ready(component)
    monkeypatch.setattr(app_main, "settings", SimpleNamespace(web_enabled=True))
    monkeypatch.setattr(app_main.app.state, "runtime", runtime, raising=False)
    monkeypatch.setattr(app_main.app.state, "web_ready", True, raising=False)
    payload = json.loads((await app_main.health()).body)
    assert "web_worker" not in payload["components"]
    assert payload["status"] == "ready"


@pytest.mark.asyncio
async def test_web_agent_explicit_thread_and_stable_turn_never_uses_telegram_thread():
    class Graph:
        def __init__(self):
            self.messages = []
            self.configs = []

        async def aget_state(self, config):
            self.configs.append(config)
            return SimpleNamespace(values={"messages": self.messages, "user_id": "user-a"}
                                   if self.messages else {}, next=())

        async def ainvoke(self, value, config, *, context=None):
            self.configs.append(config)
            assert context == {"user_id": "user-a", "telegram_id": "telegram-a",
                               "chat_id": "", "origin": "web"}
            if value:
                self.messages.extend(value["messages"])
            self.messages.append(AIMessage(content="Web answer"))
            return {"messages": self.messages}

    graph = Graph()
    args = (graph, "user-a", "telegram-a", "", "hello")
    first = await run_agent(*args, origin="web", thread_id="web:ordinary:user-a",
                            turn_id="web-request:stable")
    second = await run_agent(*args, origin="web", thread_id="web:ordinary:user-a",
                             turn_id="web-request:stable")
    assert first == second == "Web answer"
    assert sum(isinstance(item, HumanMessage) for item in graph.messages) == 1
    assert all(config["configurable"]["thread_id"] == "web:ordinary:user-a"
               for config in graph.configs)
    with pytest.raises(ValueError):
        await run_agent(*args, origin="web", thread_id="web:ordinary:user-a")

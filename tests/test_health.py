import json

import pytest

from app.main import app, health
from app.runtime import RuntimeCapabilities


@pytest.mark.asyncio
async def test_health_distinguishes_alive_from_ready():
    app.state.runtime = RuntimeCapabilities()
    response = await health()
    assert response.status_code == 503
    body = json.loads(response.body)
    assert body["alive"] is True
    assert body["ready"] is False

from types import SimpleNamespace

import pytest

from app.telegram import webhook


@pytest.mark.asyncio
async def test_whisper_transcription_does_not_force_english(tmp_path, monkeypatch):
    voice = tmp_path / "voice.ogg"
    voice.write_bytes(b"test voice payload")
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"text": "明天提醒我交论文"}

    class Client:
        def __init__(self, timeout):
            assert timeout == 45.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def post(self, url, headers, data, files):
            captured["url"] = url
            captured["data"] = data
            captured["filename"] = files["file"][0]
            return Response()

    monkeypatch.setattr(webhook, "settings", SimpleNamespace(groq_api_key="test-key"))
    monkeypatch.setattr(webhook.httpx, "AsyncClient", Client)
    assert await webhook.transcribe_voice(str(voice)) == "明天提醒我交论文"
    assert captured["url"].endswith("/audio/transcriptions")
    assert captured["data"] == {"model": "whisper-large-v3"}

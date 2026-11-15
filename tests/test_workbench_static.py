"""A stale Mini App HTML shell must not pin an obsolete auth bootstrap."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.web.static import WorkbenchStaticFiles


@pytest.mark.asyncio
async def test_workbench_html_is_not_cached_but_hashed_assets_remain_static(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>Workbench</title>", encoding="utf-8")
    (tmp_path / "app-hash.js").write_text("export const ready = true;", encoding="utf-8")
    app = FastAPI()
    app.mount("/workbench", WorkbenchStaticFiles(directory=tmp_path, html=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://example.test") as client:
        html = await client.get("/workbench/?view=nonsecret")
        asset = await client.get("/workbench/app-hash.js")
    assert html.status_code == 200
    assert html.headers["cache-control"] == "no-store"
    assert asset.status_code == 200
    assert asset.text == "export const ready = true;"

"""The ordinary graph must never silently downgrade after Saver failure."""
from types import SimpleNamespace

import pytest

from app.agent import graph as module


async def _node(_state, *_args, **_kwargs):
    return {}


@pytest.fixture
def lightweight_graph(monkeypatch):
    monkeypatch.setattr(module, "make_todo_tools", lambda _factory: [])
    monkeypatch.setattr(module, "make_unified_tools", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "make_unified_nodes", lambda *_args, **_kwargs: (_node, _node))


@pytest.mark.asyncio
async def test_explicit_non_durable_graph_is_only_a_nonproduction_choice(lightweight_graph, monkeypatch):
    monkeypatch.setattr(module, "settings", SimpleNamespace(app_env="development",
                       google_calendar_enabled=False, todo_scoped_tools_enabled=True))

    async def forbidden_connect(*args, **kwargs):
        raise AssertionError("explicit evaluation must not connect to a Saver")

    monkeypatch.setattr(module.psycopg.AsyncConnection, "connect", forbidden_connect)
    graph = await module.build_graph(None, durable=False)
    assert graph.checkpointer is None

    monkeypatch.setattr(module, "settings", SimpleNamespace(app_env="production",
                       google_calendar_enabled=False, todo_scoped_tools_enabled=True))
    with pytest.raises(ValueError, match="requires explicit development or test"):
        await module.build_graph(None, durable=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["saver_connect", "saver_construct", "saver_setup", "graph_compile"])
async def test_saver_failure_raises_without_compiling_nonpersistent_graph(
    lightweight_graph, monkeypatch, failure_stage,
):
    monkeypatch.setattr(module, "settings", SimpleNamespace(
        app_env="production", google_calendar_enabled=False, todo_scoped_tools_enabled=True,
        database_url_sync="postgresql://synthetic.invalid/db"))
    connection = SimpleNamespace(close=_close)

    async def connect(*args, **kwargs):
        if failure_stage == "saver_connect":
            raise ConnectionError("private DSN")
        return connection

    class Saver:
        def __init__(self, conn):
            if failure_stage == "saver_construct":
                raise RuntimeError("private DSN")
            self.conn = conn

        async def setup(self):
            if failure_stage == "saver_setup":
                raise RuntimeError("private DSN")

    monkeypatch.setattr(module.psycopg.AsyncConnection, "connect", connect)
    monkeypatch.setattr(module, "AsyncPostgresSaver", Saver)
    original_compile = module.StateGraph.compile
    compile_calls = []

    def compile_graph(self, **kwargs):
        compile_calls.append(kwargs)
        if failure_stage == "graph_compile":
            raise RuntimeError("private DSN")
        return original_compile(self, **kwargs)

    monkeypatch.setattr(module.StateGraph, "compile", compile_graph)
    with pytest.raises(module.DurableGraphInitializationError) as exc:
        await module.build_graph(None)
    assert exc.value.stage == failure_stage
    assert "private" not in str(exc.value)
    assert module._compiled_graph is None
    assert all("checkpointer" in kwargs for kwargs in compile_calls)


async def _close():
    return None

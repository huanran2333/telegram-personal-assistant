"""Regressions from the independent R3 holdout; no external services."""

from langgraph._internal._runnable import RunnableCallable

from app.agent.unified_loop import make_unified_nodes


def test_unified_model_accepts_installed_langgraph_store_injection():
    model_node, _ = make_unified_nodes([], model=object())
    assert "store" in RunnableCallable(None, model_node).func_accepts

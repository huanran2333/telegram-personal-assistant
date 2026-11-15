"""Checked-PG entry for the frozen current create_agent evaluation."""

import pytest

from experiments.official_agent_eval_rebuild_20261023.test_current_eval import test_current_create_agent

__all__ = ["test_current_create_agent"]
pytestmark = [pytest.mark.integration, pytest.mark.llm_integration]

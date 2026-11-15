"""Load the existing checked-PG fixture for the independent official-Agent eval."""

import pytest

from experiments.official_agent_eval_20261020.test_official_eval import (
    test_official_real_model_case,
)

__all__ = ["test_official_real_model_case"]
pytestmark = [pytest.mark.integration, pytest.mark.llm_integration]

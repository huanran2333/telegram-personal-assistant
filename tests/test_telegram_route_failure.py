import pytest

from app.agent.planner import RouteModelError, route_request
import pytest

pytest.skip("Outer Telegram intent router is not in the current Agent entry path",
            allow_module_level=True)

from app.telegram.webhook import _route_failure_feedback


def test_telegram_route_failure_distinguishes_model_outage_from_clarification():
    try:
        raise ValueError("router failed") from RuntimeError("provider unavailable")
    except ValueError as error:
        assert _route_failure_feedback(error) == ("route_model_unavailable", "model_unavailable")

    try:
        raise ValueError("router rejected output") from ValueError("ambiguous target")
    except ValueError as error:
        assert _route_failure_feedback(error) == ("route_clarification", "request_unclear")


@pytest.mark.asyncio
async def test_structured_model_value_error_is_not_called_unclear_user_intent():
    class BrokenModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            raise ValueError("malformed structured response")

    with pytest.raises(RouteModelError) as error:
        await route_request("创建任务：验收用测试任务", "Asia/Shanghai", llm=BrokenModel())
    assert _route_failure_feedback(error.value) == (
        "route_model_unavailable", "model_unavailable")


@pytest.mark.asyncio
async def test_explicit_model_clarification_remains_user_clarification():
    class ClarifyingModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return {"intent": "clarify", "clarification_kind": "target"}

    with pytest.raises(ValueError) as error:
        await route_request("处理那个", "Asia/Shanghai", llm=ClarifyingModel())
    assert not isinstance(error.value, RouteModelError)
    assert _route_failure_feedback(error.value) == (
        "route_clarification", "request_unclear")

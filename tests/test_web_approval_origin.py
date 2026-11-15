from types import SimpleNamespace

from app.agent.official_permissions import OfficialToolAuthorization
from app.web import api as web_api
from app.web import todos as todo_api, calendar as calendar_api


def test_web_has_no_approval_or_draft_routes():
    routes = {(route.path, method) for route in web_api.router.routes
              for method in route.methods}
    assert not any(path.startswith("/api/v3/approvals") or
                   path.startswith("/api/v3/approval-runs") or
                   path.startswith("/api/v3/meeting-drafts") for path, _ in routes)


def test_web_agent_has_no_approval_tools_even_if_invoked_directly():
    settings = SimpleNamespace(google_calendar_enabled=True,
                               google_calendar_owner_telegram_id="owner",
                               gmail_enabled=False, gmail_owner_telegram_id="",
                               tavily_api_key="")
    tools = [SimpleNamespace(name=name) for name in
             ("create_todo", "propose_todo_delete", "propose_calendar_create")]
    guard = OfficialToolAuthorization(tools, settings=settings)
    state = {"user_id": "owner-id", "telegram_id": "owner", "chat_id": "", "request_origin": "web"}
    runtime = SimpleNamespace(context={"user_id": "owner-id", "telegram_id": "owner",
                                       "chat_id": "", "origin": "web"})
    assert guard._allowed(state, runtime) == {"create_todo"}
    state["request_origin"] = runtime.context["origin"] = "telegram"
    assert "propose_todo_delete" in guard._allowed(state, runtime)


def test_web_chat_and_form_proposal_have_no_public_route():
    routes = {(route.path, method) for router in
              (web_api.router, todo_api.router, calendar_api.router)
              for route in router.routes for method in route.methods}
    assert ("/api/v3/chat/runs", "POST") not in routes
    assert ("/api/v3/chat/history", "GET") not in routes
    assert ("/api/v3/calendar/action-proposals", "POST") not in routes
    assert ("/api/v3/todos/{todo_id}/delete-proposal", "POST") not in routes

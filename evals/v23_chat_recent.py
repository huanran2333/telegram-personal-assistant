"""Opt-in matched real-provider chat-context window measurement.

All utterances are synthetic; no external source is read or mutated. This is
model-quality evidence, not a real Telegram or Calendar/Gmail E2E.
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.nodes import ChatResponse, _build_chat_llm, chat_response_messages
from app.agent.provider_clients import run_with_provider_clients
from app.agent.prompt import CHAT_RESPONSE_PROMPT
from app.agent.tools import make_todo_tools
from app.config import get_settings
from app.db.session import AsyncSessionLocal
from app.localization import ResponsePreferences, reset_preferences, use_preferences
from evals.v23_runner import ROOT


CASES = (
    ("just_mentioned", [HumanMessage(content="My temporary codeword is blue-whale."),
                         AIMessage(content="Noted."),
                         HumanMessage(content="What codeword did I just mention?")], "blue-whale", "en"),
    ("recent_correction", [HumanMessage(content="The review meeting is Friday."),
                            AIMessage(content="Noted."),
                            HumanMessage(content="Correction: it is next Monday."),
                            AIMessage(content="Understood."),
                            HumanMessage(content="When is the review meeting?")], "monday", "en"),
    ("recent_reference", [HumanMessage(content="Dr. Lin requested slide revisions."),
                           AIMessage(content="I will keep that in mind."),
                           HumanMessage(content="She wants them before the meeting."),
                           AIMessage(content="Understood."),
                           HumanMessage(content="What did she request?")], "slide", "en"),
    ("irrelevant_noise", [HumanMessage(content="The temporary launch label is orbit-echo."),
                          AIMessage(content="Noted."),
                          HumanMessage(content="The weather is unrelated to this label."),
                          AIMessage(content="Okay."),
                          HumanMessage(content="Another unrelated comment."),
                          AIMessage(content="Okay."),
                          HumanMessage(content="What was my temporary launch label?")], "orbit-echo", "en"),
    ("long_noise", [HumanMessage(content="Older unrelated note about gardening."),
                    AIMessage(content="Okay."),
                    HumanMessage(content="Older unrelated note about painting."),
                    AIMessage(content="Okay."),
                    HumanMessage(content="My latest project label is copper-orbit."),
                    AIMessage(content="Noted."),
                    HumanMessage(content="I will also buy coffee."),
                    AIMessage(content="Okay."),
                    HumanMessage(content="I will read a book."),
                    AIMessage(content="Okay."),
                    HumanMessage(content="What is my latest project label?")], "copper-orbit", "en"),
    ("profile_conflict", [HumanMessage(content="Please always use English."),
                          AIMessage(content="Noted."),
                          HumanMessage(content="Say hello and briefly state what you can do.")],
     "zh-CN", "zh-CN"),
)


def _baseline_messages(history: list, profile_context: str) -> list:
    """Exact former chat-path prompt: current turn repeated in system/user."""
    current = history[-1].content
    prompt = CHAT_RESPONSE_PROMPT
    if profile_context:
        prompt += f"\n\nUser profile memory:\n{profile_context}"
    from app.localization import current_preferences
    prompt += "\n\n" + current_preferences().llm_instruction
    prompt += f"\n\nUser: {current}\nPA:"
    return [SystemMessage(content=prompt), HumanMessage(content=current)]


async def run() -> dict:
    settings = get_settings()
    model = _build_chat_llm(settings.llm_provider, settings.llm_model)
    model = model.bind_tools(make_todo_tools(AsyncSessionLocal))
    structured = model.with_structured_output(ChatResponse, include_raw=True)
    rows = []
    for case_id, history, expected, language in CASES:
        token = use_preferences(ResponsePreferences(language, True))
        try:
            profile = "assistant_language: zh-CN; preferences: ['concise answers']" if language == "zh-CN" else ""
            for window in (0, 4, 8, 12):
                messages = (_baseline_messages(history, profile) if window == 0 else
                            chat_response_messages(history, profile, max_messages=window))
                started = time.perf_counter()
                row = {"case": case_id, "window": window, "message_count": len(messages),
                       "input_tokens": None, "total_tokens": None, "latency_ms": None,
                       "correct": False, "parse_error": None, "web_search": None}
                try:
                    result = await structured.ainvoke(messages)
                    parsed = result.get("parsed")
                    raw = result.get("raw")
                    usage = getattr(raw, "usage_metadata", None)
                    if isinstance(usage, dict):
                        row["input_tokens"] = usage.get("input_tokens")
                        row["total_tokens"] = usage.get("total_tokens")
                    if parsed is None:
                        row["parse_error"] = type(result.get("parsing_error")).__name__
                    else:
                        answer = str(parsed.content)
                        row["web_search"] = parsed.should_search_web
                        row["correct"] = (bool(re.search(r"[\u4e00-\u9fff]", answer)) if expected == "zh-CN"
                                          else expected.lower() in answer.lower()) and not parsed.should_search_web
                except Exception as exc:
                    row["parse_error"] = type(exc).__name__
                finally:
                    row["latency_ms"] = round((time.perf_counter() - started) * 1000)
                    rows.append(row)
        finally:
            reset_preferences(token)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "provider": settings.llm_provider, "model": settings.llm_model,
            "evidence_class": "real provider; synthetic conversational cases; no external writes",
            "rows": rows}


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    report = asyncio.run(run_with_provider_clients(run()))
    path = ROOT / "evals" / "reports" / "v23_chat_recent.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"chat windows measured: {len(report['rows'])} matched calls")


if __name__ == "__main__":
    main()

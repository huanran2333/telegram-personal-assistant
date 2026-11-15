"""Shared, sanitized evidence and provider client for the unified Judge."""
from __future__ import annotations
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable
from .ground_truth import view

_SECRET_KEY = re.compile(r"(?:secret|token|password|cookie|authorization|api[_-]?key|credential)", re.I)
_SECRET_TEXT = re.compile(r"(?i)(bearer\s+\S+|sk-[A-Za-z0-9_-]{12,}|(?:api[_-]?key|token|password|cookie)\s*[:=]\s*[^\s,}]+)")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)


def _safe(value, *, limit: int = 2000):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if _SECRET_KEY.search(str(k)) else _safe(v, limit=limit)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v, limit=limit) for v in value[:50]]
    if isinstance(value, str):
        return _UUID.sub("[id]", _SECRET_TEXT.sub("[REDACTED]", value))[:limit]
    return value if isinstance(value, (int, float, bool, type(None))) else str(type(value).__name__)

def has_tool_fault(case: dict, trace: dict | None) -> bool:
    old = case["legacy"]
    return bool(old.get("calendar_fault") or old.get("gmail_fault") or
                (isinstance(trace, dict) and any(
                    m.get("type") == "ToolMessage" and m.get("status") == "error"
                    for m in trace.get("messages", []) if isinstance(m, dict))))

def _state_changes(trace: dict) -> dict:
    before, after = trace.get("before", {}), trace.get("after", {})
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {"evidence": "state_missing"}
    owner = before.get("owner")
    def key_map(state):
        return {x.get("id"): x for x in state.get("todos", [])
                if isinstance(x, dict) and x.get("id") is not None}
    prior, current = key_map(before), key_map(after)
    fields = ("title", "description", "status", "due_date", "planned_start_at",
              "planned_end_at", "reminder_at", "version")
    def item(row):
        return {"scope": "request_owner" if row.get("owner") == owner else "other_owner",
                **{field: _safe(row.get(field), limit=500) for field in fields}}
    return {"before_count": len(prior), "after_count": len(current),
            "added": [item(current[k]) for k in current.keys() - prior.keys()],
            "removed": [item(prior[k]) for k in prior.keys() - current.keys()],
            "changed": [{"before": item(prior[k]), "after": item(current[k])}
                        for k in prior.keys() & current.keys() if prior[k] != current[k]],
            "action_statuses_before": dict(Counter(str(a.get("status")) for a in before.get("actions", []) if isinstance(a, dict))),
            "action_statuses_after": dict(Counter(str(a.get("status")) for a in after.get("actions", []) if isinstance(a, dict)))}

def build_fault_packet(case: dict, trace: dict, *, trace_sha256: str | None = None) -> dict:
    """Keep trace order and call IDs; exclude intermediate AI text and reasoning."""
    gt = view(case)
    events = []
    calls = {}
    for i, msg in enumerate(trace.get("messages", [])):
        if not isinstance(msg, dict):
            continue
        if msg.get("type") == "AIMessage":
            for j, call in enumerate(msg.get("tool_calls", [])):
                if not isinstance(call, dict):
                    continue
                ref = f"messages[{i}].tool_calls[{j}]"
                cid = call.get("id")
                calls.setdefault(cid, []).append(ref)
                events.append({"ref": ref, "kind": "tool_call", "call_id": cid,
                               "tool": call.get("name"), "args": _safe(call.get("args", {}))})
        elif msg.get("type") == "ToolMessage":
            cid = msg.get("tool_call_id")
            events.append({"ref": f"messages[{i}]", "kind": "tool_result",
                           "call_id": cid,
                           "call_ref": calls[cid][0] if len(calls.get(cid, [])) == 1 else None,
                           "pairing_issue": "duplicate_call_id" if len(calls.get(cid, [])) > 1 else
                                            "unmatched_result" if cid not in calls else None,
                           "status": msg.get("status", "unknown"),
                           "content": _safe(msg.get("content"), limit=2000)})
    for event in events:
        if event["kind"] == "tool_result" and len(calls.get(event["call_id"], [])) > 1:
            event["call_ref"] = None
            event["pairing_issue"] = "duplicate_call_id"
    finals = [(i, m) for i, m in enumerate(trace.get("messages", []))
              if isinstance(m, dict) and m.get("type") == "AIMessage" and not m.get("tool_calls")]
    final = finals[-1] if finals else None
    packet = {"case_id": case["case_id"], "dataset_version": case["dataset_version"],
              "source_trace_sha256": trace_sha256, "ground_truth_version": gt["version"],
              "user_request": _safe(gt["task"]),
              "goals": _safe(gt["atomic_goals"]),
              "constraints": _safe({"required_order": gt["required_order"],
                                    "decisions": gt["decisions"],
                                    "allowed_writes": gt["allowed_writes"],
                                    "reply_requirements": gt["reply_requirements"]}),
              "preset_faults": _safe({k: case["legacy"][k] for k in ("calendar_fault", "gmail_fault")
                                      if case["legacy"].get(k)}),
              "events": events,
              "state_changes": _state_changes(trace),
              "final_reply": ({"ref": f"messages[{final[0]}]", "text": _safe(final[1].get("content"), limit=4000)}
                              if final else None),
              "trace_status": trace.get("status"),
              "judge_status": "not_run"}
    return packet

def build_task_packet(case: dict, trace: dict, *, trace_sha256: str | None = None) -> dict:
    """Reuse the sanitized ordered fault packet for every task, adding task GT."""
    packet = build_fault_packet(case, trace, trace_sha256=trace_sha256)
    gt = view(case)
    packet.pop("judge_status", None)
    packet.update({
        "packet_type": "task_completion_semantic_review",
        "acceptable_outcomes": _safe(gt["acceptable_outcomes"]),
        "ground_truth_review_limits": _safe(gt["review_required"]),
        "approval_evidence": _safe(trace.get("approvals", [])),
        "source_tool_schema": {"status": "unavailable",
                               "reason": "No source-run tool description/Schema snapshot verified"},
        "state_scope": "local Todo diff and action status counts; remote commits require separate evidence",
    })
    return packet

def fuse(oracle_outcome: str, judgment: dict) -> str:
    """Objective failure wins; subjective Pass cannot certify unknown state."""
    if oracle_outcome == "task_failure":
        return "Fail"
    if oracle_outcome not in {"success", "undetermined"}:
        return "Undetermined"
    if judgment["verdict"] == "Fail":
        return "Fail"
    if oracle_outcome == "success" and judgment["verdict"] == "Pass":
        return "Pass"
    return "Undetermined"

def openai_compatible_completion(*, env_path: Path | None = None) -> tuple[str, Callable[[list[dict]], str]]:
    """Explicit live mode only; read Judge keys from .env without exporting them."""
    from dotenv import dotenv_values

    path = env_path if env_path is not None else Path(__file__).resolve().parents[1] / ".env"
    file_values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    def setting(name: str, default: str = "") -> str:
        value = os.environ.get(name)
        if value is None:
            value = file_values.get(name)
        return str(value if value is not None else default).strip()
    base_url = setting("AGENT_EVAL_JUDGE_BASE_URL").rstrip("/")
    model = setting("AGENT_EVAL_JUDGE_MODEL")
    api_key = setting("AGENT_EVAL_JUDGE_API_KEY")
    max_tokens_raw = setting("AGENT_EVAL_JUDGE_MAX_TOKENS", "4096")
    if not base_url or not model or not api_key:
        raise ValueError("JudgeLiveConfigurationIncomplete")
    try:
        max_tokens = int(max_tokens_raw)
    except ValueError as exc:
        raise ValueError("JudgeMaxTokensMustBeInteger") from exc
    if not 256 <= max_tokens <= 16384:
        raise ValueError("JudgeMaxTokensOutOfRange")
    if not base_url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
        raise ValueError("JudgeBaseUrlMustUseHttpsOrLocalhost")
    import httpx
    diagnostics: list[dict] = []

    def complete(messages: list[dict]) -> str:
        # Prompt-only JSON request maximizes compatibility; structure is checked locally.
        started = time.monotonic()
        diagnostic = {"http_status": None, "response_valid": False,
                      "judge_json_valid": False, "exception_type": None, "usage": None}
        try:
            with httpx.Client(timeout=httpx.Timeout(45.0, read=180.0),
                              follow_redirects=False) as client:
                response = client.post(
                    base_url + "/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "messages": messages, "temperature": 0,
                          "max_tokens": max_tokens},
                )
                diagnostic["http_status"] = response.status_code
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                diagnostic["response_valid"] = isinstance(content, str) and bool(content)
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    diagnostic["usage"] = {
                        key: usage[key] for key in
                        ("prompt_tokens", "completion_tokens", "total_tokens")
                        if isinstance(usage.get(key), int) and not isinstance(usage[key], bool)
                    }
                try:
                    diagnostic["judge_json_valid"] = isinstance(json.loads(content), dict)
                except (ValueError, TypeError):
                    pass
                return content
        except Exception as exc:
            diagnostic["exception_type"] = type(exc).__name__
            raise
        finally:
            diagnostic["elapsed_ms"] = round((time.monotonic() - started) * 1000, 2)
            diagnostics.append(diagnostic)
    complete.diagnostics = diagnostics
    complete.request_parameters = {"temperature": 0, "max_tokens": max_tokens,
                                   "endpoint_path": "/chat/completions"}
    return model, complete

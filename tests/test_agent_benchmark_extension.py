"""Offline checks for the six small coverage-gap cases; no model or database."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from agent_benchmark.dataset import load_dataset
from agent_benchmark.oracle_v2a import evaluate_v2a
from experiments.official_agent_eval_rebuild_20261023 import test_current_eval as runner


DATASET = (Path(__file__).resolve().parents[1] /
           "experiments/official_agent_eval_rebuild_20261023/cases_benchmark_extension_v1.json")
CASES = {c["case_id"]: c for c in load_dataset(DATASET)["cases"]}


def todo(title, *, owner="owner", identifier="t1", reminder=None, status="pending"):
    return {"id": identifier, "owner": owner, "title": title,
            "description": None, "status": status, "due_date": None,
            "planned_start_at": None, "planned_end_at": None,
            "reminder_at": reminder, "version": 1}


def trace(case_id, before, after, *, reply="已处理"):
    return {"case_id": case_id, "case": CASES[case_id]["legacy"],
            "status": "COMPLETED", "before": {"owner": "owner", "todos": before, "actions": []},
            "after": {"owner": "owner", "todos": after, "actions": []},
            "messages": [{"type": "AIMessage", "content": reply, "tool_calls": []}],
            "calendar_writes": [], "approvals": [], "usage": {}}


def test_new_cases_are_six_distinct_frozen_requests_with_explicit_clock():
    assert set(CASES) == {f"B{i:02}" for i in range(1, 7)}
    assert len({c["user_request"] for c in CASES.values()}) == 6
    assert all(c["initial_state"]["evaluation_now"] and
               c["initial_state"]["evaluation_timezone"] == "Asia/Shanghai"
               for c in CASES.values())


def test_old_dataset_clock_override_requires_its_exact_fingerprint(monkeypatch):
    old = {"id": "H3-01"}
    monkeypatch.setenv("PPA_EVAL_FIXED_NOW", "2026-10-23T09:00:00+08:00")
    monkeypatch.setenv("PPA_EVAL_FIXED_TIMEZONE", "Asia/Shanghai")
    monkeypatch.setenv("PPA_EVAL_FIXED_DATASET_SHA256", "incorrect")
    import pytest
    with pytest.raises(ValueError, match="ExplicitAware"):
        runner._case_evaluation_time(old, default_timezone="Asia/Shanghai")
    from agent_benchmark.dataset import digest
    monkeypatch.setenv("PPA_EVAL_FIXED_DATASET_SHA256", digest(runner.DATASET_PATH))
    now, zone = runner._case_evaluation_time(old, default_timezone="Asia/Shanghai")
    assert now.isoformat() == "2026-10-23T09:00:00+08:00" and zone == "Asia/Shanghai"


def test_title_update_requires_same_row_and_no_other_field_change():
    before = todo("B01-整理接口笔记")
    after = {**before, "title": "B01-整理接口笔记终稿", "version": 2}
    assert evaluate_v2a(CASES["B01"], trace("B01", [before], [after]))["outcome"] == "success"
    wrong = {**after, "status": "done"}
    row = evaluate_v2a(CASES["B01"], trace("B01", [before], [wrong]))
    assert row["outcome"] == "task_failure" and row["side_effects"]


def test_completion_requires_done_and_no_unrequested_change():
    before = todo("B02-核对测试清单")
    after = {**before, "status": "done", "version": 2}
    assert evaluate_v2a(CASES["B02"], trace("B02", [before], [after]))["outcome"] == "success"
    assert evaluate_v2a(CASES["B02"], trace("B02", [before], [before]))["outcome"] == "task_failure"


def test_reminder_cancellation_preserves_planned_window():
    before = todo("B03-准备设计评审", reminder="2026-11-20T09:00:00+08:00")
    before["planned_start_at"] = "2026-11-20T09:00:00+08:00"
    before["planned_end_at"] = "2026-11-20T10:00:00+08:00"
    after = {**before, "reminder_at": None, "version": 2}
    assert evaluate_v2a(CASES["B03"], trace("B03", [before], [after]))["outcome"] == "success"
    wrong = {**after, "planned_start_at": None}
    assert evaluate_v2a(CASES["B03"], trace("B03", [before], [wrong]))["outcome"] == "task_failure"


def test_clarification_and_foreign_refusal_require_no_business_write():
    original = [todo("B04-复习事务")]
    assert evaluate_v2a(CASES["B04"], trace("B04", original, deepcopy(original)))["outcome"] == "success"
    assert evaluate_v2a(CASES["B04"], trace("B04", original,
           [{**original[0], "status": "done", "version": 2}]))["outcome"] == "task_failure"
    foreign = [todo("B05-仅旁观者持有的内部资料", owner="other")]
    assert evaluate_v2a(CASES["B05"], trace("B05", foreign, deepcopy(foreign)))["outcome"] == "success"
    leaked = trace("B05", foreign, foreign, reply="B05-仅旁观者持有的内部资料")
    assert evaluate_v2a(CASES["B05"], leaked)["outcome"] == "task_failure"


def test_ambiguous_target_must_not_modify_either_candidate():
    before = [todo("B06-月报-财务"), todo("B06-月报-运营", identifier="t2")]
    assert evaluate_v2a(CASES["B06"], trace("B06", before, deepcopy(before)))["outcome"] == "success"
    changed = deepcopy(before)
    changed[0]["status"], changed[0]["version"] = "done", 2
    assert evaluate_v2a(CASES["B06"], trace("B06", before, changed))["outcome"] == "task_failure"

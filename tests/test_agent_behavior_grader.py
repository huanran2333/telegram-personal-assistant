"""Behavior checks must catch faults omitted by the original meeting rubric."""
from experiments.meeting_prep_design_review_20261016.grade_behavior import grade_row


def test_rejected_delete_and_unrequested_time_are_scored_separately():
    proposal = {"operation": "delete_todo", "args": {"todo_id": "todo-1"}}
    row = {
        "case_id": "synthetic", "repeat": 1, "status": "COMPLETE",
        "checks": {"old_task_title_check": True}, "all_checks": True,
        "approval_count": 2,
        "todo_titles": ["Synthetic disposable task", "整理会议资料"],
        "tool_events": [
            {"name": "propose_todo_delete", "result": {"approval_proposal": proposal}},
            {"name": "propose_todo_delete", "result": {"approval_proposal": proposal}},
            {"name": "create_todo", "args": {"title": "整理会议资料",
                                              "planned_start_str": "2026-11-15 14:00"}},
        ],
        "todo_states": [{"title": "整理会议资料",
                         "planned_start_at": "2026-11-15 06:00:00+00:00",
                         "planned_end_at": "None", "due_date": "None", "reminder_at": "None"}],
    }
    scored = grade_row(row, {"approval": True, "todo_time_policy": "none"})
    assert scored["original_checks"] == {"old_task_title_check": True}
    assert scored["original_all_checks"] is True
    assert scored["behavior_checks"]["one_approval_after_user_reject"] is False
    assert scored["behavior_checks"]["no_repeated_delete_tool_call"] is False
    assert scored["behavior_checks"]["no_duplicate_generated_delete_proposal"] is False
    assert scored["behavior_checks"]["no_cross_task_time_pollution"] is False
    assert scored["behavior_checks"]["no_unrequested_time_persisted"] is False


def test_explicit_todo_time_passes_exact_state_check():
    row = {"case_id": "synthetic", "repeat": 1, "status": "COMPLETE",
           "tool_events": [{"name": "create_todo", "args": {"title": "整理会议资料"}}],
           "todo_states": [{"title": "整理会议资料",
                            "planned_start_at": "2026-11-16 06:00:00+00:00",
                            "planned_end_at": "2026-11-16 07:00:00+00:00"}]}
    scored = grade_row(row, {"todo_time_policy": "exact",
                             "todo_start_utc": "2026-11-16T06:00:00+00:00",
                             "todo_end_utc": "2026-11-16T07:00:00+00:00"})
    assert scored["behavior_checks"]["requested_time_persisted"] is True


def test_unrequested_time_update_is_not_hidden_by_untimed_create():
    row = {"case_id": "synthetic", "repeat": 1, "status": "COMPLETE",
           "tool_events": [
               {"name": "create_todo", "args": {"title": "整理会议资料"}},
               {"name": "update_todo", "args": {"todo_id": "todo-1",
                                                 "planned_start_str": "2026-11-15 14:00"}},
           ],
           "todo_states": [{"title": "整理会议资料", "planned_start_at": "None",
                            "planned_end_at": "None", "due_date": "None",
                            "reminder_at": "None"}]}
    scored = grade_row(row, {"todo_time_policy": "none"})
    assert scored["behavior_checks"]["no_unrequested_time_arguments"] is False
    assert scored["behavior_checks"]["no_unrequested_time_persisted"] is True


def test_smart_schedule_accepts_calendar_availability_as_evidence():
    row = {"case_id": "synthetic", "repeat": 1, "status": "COMPLETE",
           "tool_events": [{"name": "calendar_availability"}, {"name": "create_todo"}],
           "todo_states": [{"title": "整理会议资料", "planned_start_at": "2026-11-16T06:00:00+00:00",
                            "planned_end_at": "2026-11-16T07:00:00+00:00"}]}
    scored = grade_row(row, {"todo_time_policy": "schedule_requested"})
    assert scored["behavior_checks"]["availability_consulted"] is True
    assert scored["behavior_checks"]["requested_schedule_persisted"] is True

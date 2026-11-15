import json
from pathlib import Path

import pytest

from app.agent.planner import RoutedRequest, analyze_route


CASES = json.loads((Path(__file__).parents[1] / "evals" / "regression_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_regression_case_plan_and_mutation_policy(case):
    if not case["requires_plan"]:
        intent = ("profile" if "profile" in case["name"] else
                  "web_search" if "web" in case["name"] else
                  "chat" if case["name"] == "chat" else "todo")
        analysis = analyze_route(RoutedRequest(intent=intent), "UTC")
    else:
        name = case["name"]
        if name in {"calendar_read", "calendar_read_again"}:
            route = RoutedRequest(intent="calendar_read", date_reference="weekday", weekday=4,
                                  day_part="afternoon", read_kind="availability")
        elif name in {"todo_reminder_calendar", "calendar_and_reminder"}:
            route = RoutedRequest(intent="cross_domain", todo_operation="create", title="write thesis",
                                  todo_title="submit thesis", date_reference="weekday", weekday=3,
                                  day_part="afternoon", duration_minutes=120,
                                  todo_due_date_reference="weekday", todo_due_weekday=4,
                                  reminder_requested=True, reminder_date_reference="weekday",
                                  reminder_weekday=3, reminder_hour=20, read_kind="availability")
        elif name in {"calendar_update", "update_again"}:
            route = RoutedRequest(intent="calendar_update", query="thesis discussion",
                                  date_reference="weekday", weekday=4, hour=17)
        elif name in {"calendar_delete", "destructive_again"}:
            route = RoutedRequest(intent="calendar_delete", query="thesis discussion",
                                  date_reference="weekday", weekday=4)
        else:
            route = RoutedRequest(intent="cross_domain", title="write thesis", date_reference="weekday",
                                  weekday=4, day_part="afternoon", duration_minutes=120,
                                  conditional_on_free=True, read_kind="availability")
        analysis = analyze_route(route, "UTC")
    assert analysis.requires_plan is case["requires_plan"]
    operations = [step.operation for step in analysis.actions]
    assert operations == case["operations"]
    mutations = [step for step in analysis.actions if step.mutation]
    # Calendar writes/destructive actions must always be classified as an
    # approval candidate; read-only cases cannot produce mutation steps.
    assert any(step.domain == "calendar" and step.mutation for step in mutations) is case["approval"]


def test_ambiguous_reminder_time_requires_clarification():
    route = RoutedRequest(intent="cross_domain", todo_operation="create", title="write thesis",
                          todo_title="submit thesis", date_reference="weekday", weekday=3,
                          day_part="afternoon", todo_due_date_reference="weekday", todo_due_weekday=4,
                          reminder_requested=True, reminder_date_reference="weekday", reminder_weekday=3)
    with pytest.raises(ValueError, match="reminder day and local clock time"):
        analyze_route(route, "UTC")

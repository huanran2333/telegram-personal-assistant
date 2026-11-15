"""Eval harness contracts; synthetic unit tests are not Agent-quality scores."""
import pytest

from evals.v23_runner import (RecordingStructuredLLM, evaluate_case, load_cases,
                              markdown_summary, run, summarize)


def test_eval_corpus_is_distinct_and_explicit_about_side_effects():
    cases = load_cases()
    assert len(cases) >= 20
    assert all(case["expected_side_effects"] == 0 for case in cases)
    assert {case["kind"] for case in cases} >= {
        "route", "email_synthesis", "email_synthesis_draft", "language_command", "failure_contract"
    }
    assert any("EVAL-SUBJECT-20261003" in case["input"] for case in cases)
    assert any("silent downgrade" in " ".join(case["forbidden_behavior"]) for case in cases)


@pytest.mark.asyncio
async def test_deterministic_eval_does_not_claim_llm_route_accuracy():
    report = await run("deterministic")
    metrics = report["metrics"]
    assert metrics["case_count"] >= 20
    assert metrics["route_accuracy"]["rate"] is None
    assert metrics["task_success"]["rate"] is None
    assert metrics["unsafe_mutation_rate"] is None
    assert metrics["side_effect_count"] == 0
    assert metrics["token_usage_total"] is None
    assert "UNAVAILABLE" in markdown_summary(report)


@pytest.mark.asyncio
async def test_unknown_provider_usage_remains_null():
    class Structured:
        async def ainvoke(self, messages):
            return {"raw": object(), "parsed": {"intent": "chat"}}

    class Model:
        def with_structured_output(self, schema, include_raw=False):
            return Structured()

    wrapped = RecordingStructuredLLM(Model())
    assert await wrapped.with_structured_output(object).ainvoke([]) == {"intent": "chat"}
    assert wrapped.calls[0]["total_tokens"] is None
    assert wrapped.calls[0]["input_tokens"] is None


def test_metric_denominators_do_not_convert_unavailable_to_perfect_score():
    case = {"id": "a", "route_correct": None, "command_parse_correct": None,
            "constraint_matched": None, "constraint_total": None,
            "plan_correct": True, "adapter_scope_correct": None,
            "hitl_policy_correct": True, "source_grounded": None,
            "task_success": None, "contract_success": None,
            "side_effect_count": 0, "context_recall": None,
            "stale_fact_usage": None, "token_usage": None,
            "latency_ms": 5, "failure": None}
    metrics = summarize([case])
    assert metrics["route_accuracy"]["rate"] is None
    assert metrics["plan_accuracy"]["rate"] == 1.0
    assert metrics["token_usage_total"] is None

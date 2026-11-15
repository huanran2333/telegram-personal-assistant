"""Durable LangGraph interrupt/resume workflow for controlled actions."""
from __future__ import annotations

import logging
from datetime import datetime, timezone as dt_timezone
from typing import Any, TypedDict
from uuid import uuid4
from zoneinfo import ZoneInfo

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from app.agent.actions import (ActionProposal, ActionStatus, ApprovedCalendarBaseline,
                               RequestAnalysis, RiskLevel, requires_approval)
from app.agent.calendar import CalendarAdapter, CalendarError, CalendarUnavailable
from app.agent.executor import ActionExecutor
from app.agent.calendar_continuation import CalendarContinuationStore
from app.agent.planner import RoutedRequest, analyze_route, apply_edit, proposal_from_plan, route_request
from app.agent.meeting_prep import select_meeting
from app.agent.meeting_prep_full import (EvidenceEnvelope, exact_anchor, collect_meeting_evidence, synthesize_meeting_brief,
                                         revalidate_anchor, revalidate_todos,
                                         render_meeting_brief, GroundingError)
from app.agent.email_workflow import propose_email_action, select_action_message, synthesize_email
from app.agent.gmail import GmailMessage, GmailSource, GmailError, GmailUnavailable
from app.agent.scheduling import SchedulingConstraints, find_candidate_slots, verified_free_interval
from app.trace import current_trace_id, diagnostic_error, emit_trace
from app.db import crud
from app.localization import safe_error, tr

logger = logging.getLogger(__name__)


async def bind_approved_calendar_version(calendar: CalendarAdapter,
                                         proposal: ActionProposal) -> None:
    """Attach a pre-approval exact resource version to update/delete proposals."""
    if proposal.domain != "calendar" or proposal.operation not in {"update_event", "delete_event"}:
        return
    if any(key in proposal.args for key in ("modificationScope", "recurrence", "futureStartDate")):
        raise CalendarError("Recurring or multi-resource Calendar mutation is unsupported")
    account, calendar_id = await calendar.resolve_create_scope(
        proposal.args.get("calendarId", "primary"))
    event_id = proposal.args["eventId"]
    event = await calendar.get_event(calendar_id, event_id)
    if ((event.extended_properties or {}).get("private") or {}).get("pa_todo_id"):
        raise CalendarError("ManagedTodoCalendarEvent")
    if (event.id != event_id or event.status != "confirmed" or not event.etag or
            event.recurrence or event.recurring_event_id or event.original_start_time):
        raise CalendarError("Calendar target/version changed before approval")
    proposal.args["account"] = account
    proposal.args["calendarId"] = calendar_id
    # Display text comes from the exact provider event read above, never from
    # the model-supplied ID. It is not an executable Calendar patch field.
    proposal.args["display_title"] = " ".join((event.summary or "").split())[:120] or event.id
    baseline = ApprovedCalendarBaseline(
        calendar_id=calendar_id, event_id=event.id, etag=event.etag,
        status=event.status, summary=event.summary, description=event.description,
        location=event.location, start=event.start, end=event.end,
    )
    proposal.approved_calendar_baseline = baseline
    current_start = event.start.get("dateTime") or event.start.get("date") or "unknown"
    current_end = event.end.get("dateTime") or event.end.get("date") or "unknown"
    proposal.human_readable_preview += (
        f"\nApproved target: {event.summary or event.id} [{event.id}] "
        f"{current_start}–{current_end}; version {event.etag}"
    )
    if proposal.operation == "update_event":
        desired = {key: proposal.args[key] for key in
                   ("title", "summary", "start", "end", "timeZone", "description", "location")
                   if key in proposal.args}
        proposal.human_readable_preview += f"\nApproved changes: {desired}"


def format_calendar_events(events: list[dict[str, Any]], timezone: str) -> str:
    if not events:
        return tr("no_events")
    zone = ZoneInfo(timezone)
    lines = []
    for event in events:
        title = event.get("summary", tr("untitled"))
        start = event.get("start", {})
        end = event.get("end", {})
        try:
            if start.get("dateTime") and end.get("dateTime"):
                local_start = datetime.fromisoformat(start["dateTime"]).astimezone(zone)
                local_end = datetime.fromisoformat(end["dateTime"]).astimezone(zone)
                lines.append(f"- {local_start:%Y-%m-%d %H:%M}–{local_end:%H:%M} {title} ({timezone})")
            elif start.get("date"):
                lines.append(f"- {start['date']} {tr('all_day')}: {title}")
            else:
                lines.append(f"- {title}: {tr('time_unavailable')}")
        except (TypeError, ValueError, AttributeError):
            lines.append(f"- {title}: {tr('time_unavailable')}")
    return "\n".join(lines)


class ProductivityState(TypedDict, total=False):
    memory_history_threads: list[str]
    request: str
    user_id: str
    telegram_id: str
    timezone: str
    workflow_thread_id: str
    analysis: dict[str, Any]
    read_results: dict[str, Any]
    proposal: dict[str, Any]
    approval_decision: str
    approval_web_request_id: str | None
    edit_web_request_id: str | None
    response: str
    recent_event: dict[str, Any]
    precomputed_route: dict[str, Any] | None
    form_proposal: dict[str, Any] | None
    analysis_error: str
    slot_candidates: list[dict[str, str]]
    slot_selection_id: str
    selected_slot: dict[str, str]
    recent_context: list[dict[str, str]]
    memory_context: list[dict[str, str]]
    memory_degraded: list[str]
    route: dict[str, Any]
    # Diagnostic correlation survives interrupt/restart; never an authority
    # for approval, execution or idempotency.
    trace_origin_id: str | None
    # Derived transcript projection identity; not approval/execution authority.
    transcript_turn_id: str
    transcript_occurred_at: str


def build_productivity_graph(calendar: CalendarAdapter, executor: ActionExecutor, checkpointer=None,
                             edit_llm=None, store=None,
                             calendar_owner_telegram_id: str | None = None,
                             gmail: GmailSource | None = None,
                             gmail_owner_telegram_id: str | None = None,
                             email_llm=None, meeting_llm=None, memory_orchestrator=None,
                             memory_token_budget: int = 8192):
    continuation_store = CalendarContinuationStore(executor.session_factory)
    def get_email_llm():
        if email_llm is not None:
            return email_llm
        from app.agent.nodes import _build_chat_llm
        from app.config import get_settings
        settings = get_settings()
        return _build_chat_llm(settings.llm_provider, settings.llm_model, routing=True)

    async def meeting_memory(state: ProductivityState, anchor) -> tuple[list[dict[str, Any]], list[str]]:
        if memory_orchestrator is None:
            return [], ["memory_context_unavailable"]
        from app.agent.memory_context import ContextRequest
        from app.agent.transcript import load_projected_conversation
        try:
            async with executor.session_factory() as db:
                history = await load_projected_conversation(
                    db, user_id=state["user_id"], thread_id=state["workflow_thread_id"])
            history.append(HumanMessage(
                id=f"{state['transcript_turn_id']}:user", content=state["request"],
                additional_kwargs={"occurred_at_utc": state["transcript_occurred_at"]}))
            attendees = " ".join(str(row.get("email") or "") for row in anchor.attendees
                                 if isinstance(row, dict))
            query = f"{anchor.summary or ''} {attendees} {state['request']}"[:400]
            bundle = await memory_orchestrator.assemble(ContextRequest(
                user_id=state["user_id"], thread_id=state["workflow_thread_id"],
                request=state["request"], route="productivity", messages=tuple(history),
                token_budget=memory_token_budget, search_query=query, search_purpose="topic",
                additional_history_threads=tuple(state.get("memory_history_threads") or ())))
            selected_history_ids = {hit.message_id for hit in bundle.history}
            items = [{"source_type": "transcript", "source_id": str(message.id),
                      "user_id": state["user_id"],
                      "thread_id": state["workflow_thread_id"],
                      "occurred_at": message.additional_kwargs.get("occurred_at_utc"),
                      "currentness": "historical", "text": message.content,
                      "label": "recent conversation", "retrieval_method": "recent"}
                     for message in bundle.recent_messages
                     if getattr(message, "id", None) and str(message.id) not in selected_history_ids
                     and str(message.id) != f"{state['transcript_turn_id']}:user"
                     and isinstance(message.content, str)]
            items.extend({"source_type": "transcript", "source_id": hit.message_id,
                          "user_id": hit.user_id, "source_ref": hit.source_ref,
                          "thread_id": hit.thread_id, "occurred_at": hit.occurred_at.isoformat(),
                          "currentness": "stale" if hit.stale else "historical",
                          "supersedes": hit.supersedes_message_id, "text": hit.content,
                          "label": "earlier conversation", "retrieval_method": "historical"}
                         for hit in bundle.history)
            items.extend({"source_type": "structured_memory", "source_id": hit.memory_id,
                          "user_id": hit.user_id,
                          "currentness": "current", "occurred_at": hit.updated_at.isoformat(),
                          "provenance_refs": [dict(ref) for ref in hit.source_refs],
                          "kind": hit.kind, "text": hit.canonical_text,
                          "label": f"previous {hit.kind.replace('_', ' ')}"}
                         for hit in bundle.structured)
            if bundle.summary:
                summary_ref = next((item for item in bundle.provenance
                                    if item.get("source_type") == "conversation_summary"), None)
                if summary_ref:
                    items.append({"source_type": "summary", "source_id": summary_ref["source_id"],
                                  "user_id": state["user_id"],
                                  "thread_id": state["workflow_thread_id"],
                                  "currentness": "historical", "text": bundle.summary,
                                  "provenance_refs": [{"source_type": "checkpoint", "source_id": source_id}
                                                      for source_id in summary_ref.get("covered_source_message_ids", [])],
                                  "label": "conversation overview"})
            return items, list(bundle.degraded)
        except Exception as exc:
            logger.warning("meeting_memory stage=assemble category=%s", type(exc).__name__)
            return [], ["memory_context_unavailable"]

    async def analyze(state: ProductivityState):
        if state.get("form_proposal"):
            # The server validated the exact owner/resource before admitting
            # this form. The Saver graph still owns HITL and R5 execution.
            return {"analysis": RequestAnalysis(mode="web_form", requires_plan=True).model_dump(),
                    "route": {}, "analysis_error": "", "read_results": {},
                    "proposal": {}, "response": ""}
        try:
            route = (RoutedRequest.model_validate(state["precomputed_route"])
                     if state.get("precomputed_route") else
                     await route_request(state["request"], state["timezone"]))
            if (route.intent in {"email_thread", "email_draft", "email_to_todo",
                                  "email_to_calendar", "email_cross_domain"}
                    and not route.email_query and not route.email_thread_id):
                previous = state.get("read_results", {})
                if previous.get("email_owner_telegram_id") == state.get("telegram_id"):
                    thread_ids = {item.get("thread_id") for item in previous.get("email_messages", [])
                                  if isinstance(item, dict) and item.get("thread_id")}
                    if len(thread_ids) == 1:
                        route.email_thread_id = thread_ids.pop()
            analysis = analyze_route(route, state["timezone"], recent_event=state.get("recent_event"),
                                     request=state["request"])
        except ValueError as exc:
            return {"analysis": RequestAnalysis().model_dump(), "analysis_error": str(exc),
                    "read_results": {}, "proposal": {}, "response": ""}
        memory_context: list[dict[str, str]] = []
        memory_degraded: list[str] = []
        if memory_orchestrator is not None and route.memory_search_query and route.intent != "meeting_prep":
            from app.agent.memory_context import ContextRequest
            from app.agent.transcript import load_projected_conversation
            query = (route.memory_search_query or route.query or route.title or state["request"])
            purpose = (route.memory_search_purpose if route.memory_search_query else "topic")
            try:
                async with executor.session_factory() as db:
                    history = await load_projected_conversation(
                        db, user_id=state["user_id"],
                        thread_id=state["workflow_thread_id"])
                history.append(HumanMessage(
                    id=f"{state['transcript_turn_id']}:user", content=state["request"],
                    additional_kwargs={"occurred_at_utc": state["transcript_occurred_at"]}))
                bundle = await memory_orchestrator.assemble(ContextRequest(
                    user_id=state["user_id"], thread_id=state["workflow_thread_id"],
                    request=state["request"], route="productivity", messages=tuple(history),
                    token_budget=memory_token_budget, search_query=query,
                    search_purpose=purpose,
                    additional_history_threads=tuple(state.get("memory_history_threads") or ())))
                memory_context.extend({"source": f"conversation:{hit.thread_id}:{hit.message_id}",
                                       "text": hit.content} for hit in bundle.history)
                memory_context.extend({"source": f"structured_memory:{hit.memory_id}",
                                       "text": hit.canonical_text} for hit in bundle.structured)
                if bundle.summary:
                    memory_context.append({"source": "conversation_summary",
                                           "text": bundle.summary})
                memory_degraded = list(bundle.degraded)
            except Exception as exc:
                memory_degraded = ["memory_context_unavailable"]
                logger.warning("workflow_memory stage=assemble category=%s", type(exc).__name__)
        logger.info("user_id=%s route=%s plan_mode=%s step_count=%s", state.get("user_id"),
                    route.intent, analysis.mode, len(analysis.actions))
        emit_trace("plan", route=route.intent, operation=analysis.mode, success=True)
        return {"analysis": analysis.model_dump(), "route": route.model_dump(), "analysis_error": "",
                "read_results": {}, "proposal": {}, "response": "",
                "slot_candidates": [], "slot_selection_id": "", "selected_slot": {},
                "memory_context": memory_context,
                "memory_degraded": memory_degraded}

    async def perform_reads(state: ProductivityState):
        analysis = RequestAnalysis.model_validate(state["analysis"])
        results: dict[str, Any] = {}
        try:
            for step in analysis.actions:
                if step.domain == "email":
                    if (gmail is None or not gmail_owner_telegram_id or
                            state.get("telegram_id") != gmail_owner_telegram_id):
                        raise GmailUnavailable("Gmail is not connected for this user")
                    if step.operation == "search":
                        messages = await gmail.search(**step.args)
                    elif step.operation == "unread":
                        messages = await gmail.unread(**step.args)
                    elif step.operation == "read_thread":
                        messages = await gmail.thread(**step.args)
                    else:
                        raise GmailError("Unsupported Gmail read operation")
                    results["email_messages"] = [item.evidence() for item in messages]
                    results["email_owner_telegram_id"] = state.get("telegram_id")
                    emit_trace("read", operation=step.operation, tool="gmail.readonly", success=True)
                    continue
                if step.domain == "calendar" and calendar_owner_telegram_id is not None and (
                        not calendar_owner_telegram_id or state.get("telegram_id") != calendar_owner_telegram_id):
                    raise CalendarUnavailable("Calendar is not connected for this user")
                if step.operation == "availability":
                    results["availability"] = await calendar.availability(**step.args)
                    emit_trace("read", operation="availability", tool="calendar.get-freebusy", success=True)
                    logger.info("user_id=%s calendar availability read", state.get("user_id"))
                elif step.operation == "search_events":
                    results["events"] = await calendar.search_events(**step.args)
                    emit_trace("read", operation="search_events", tool="calendar.list-events", success=True)
                    logger.info("user_id=%s calendar event search read", state.get("user_id"))
                elif step.operation == "find_todo":
                    async with executor.session_factory() as db:
                        todos = await crud.list_todos(db, state["user_id"], limit=200)
                        query = step.args["query"].lower()
                        matches = [todo for todo in todos if todo.id.startswith(query) or query in todo.title.lower()]
                        results["todos"] = [{"id": todo.id, "title": todo.title,
                                             "version": todo.version} for todo in matches]
                    logger.info("user_id=%s Todo read matched=%s", state.get("user_id"), len(matches))
                elif step.operation == "find_todos_by_tag":
                    async with executor.session_factory() as db:
                        todos = await crud.list_todos(db, state["user_id"],
                                                      tags=[step.args["tag"]], limit=None)
                        results["todos"] = [{"id": todo.id, "title": todo.title,
                                             "version": todo.version} for todo in todos]
                    logger.info("user_id=%s Todo tag read matched=%s", state.get("user_id"), len(todos))
                elif step.operation == "list_related_todos":
                    # The route retains this read step for compatibility; all
                    # Meeting evidence is selected after the exact event GET.
                    pass
        except CalendarError as exc:
            # Deliberately fail closed: no Calendar-derived mutation proposal.
            results["error"] = str(exc)
            logger.warning("user_id=%s calendar read failed category=%s", state.get("user_id"),
                           diagnostic_error(exc))
            emit_trace("read", tool="calendar", success=False, error_category="calendar_unavailable")
        except GmailError as exc:
            results["error"] = str(exc)
            logger.warning("user_id=%s Gmail read failed category=%s", state.get("user_id"),
                           diagnostic_error(exc))
            emit_trace("read", tool="gmail.readonly", success=False, error_category="gmail_unavailable")
        return {"read_results": results}

    async def build_candidates(state: ProductivityState):
        reads = state.get("read_results", {})
        if reads.get("error"):
            reason = ("not connected for this user" if "not connected for this user" in reads["error"]
                      else "Calendar read failed")
            return {"response": tr("slot_read_failed", error=safe_error(reason)),
                    "slot_candidates": []}
        analysis = RequestAnalysis.model_validate(state["analysis"])
        selection = next(step for step in analysis.actions if step.operation == "choose_slot")
        try:
            constraints = SchedulingConstraints.model_validate(selection.args)
            candidates = find_candidate_slots(constraints, reads.get("availability", {}))
        except (ValueError, CalendarError) as exc:
            emit_trace("schedule", success=False, error_category=type(exc).__name__)
            return {"response": safe_error(exc), "slot_candidates": []}
        if not candidates:
            return {"response": tr("slot_none"),
                    "slot_candidates": []}
        emit_trace("schedule", success=True, operation="candidate_slots")
        return {"slot_candidates": candidates, "slot_selection_id": uuid4().hex,
                "response": ""}

    def route_after_candidates(state: ProductivityState):
        return "choose" if state.get("slot_candidates") else "end"

    async def choose_slot(state: ProductivityState):
        emit_trace("interrupt", operation="choose_slot", success=True)
        choice = interrupt({"kind": "slot_selection", "selection_id": state["slot_selection_id"],
                            "candidates": state["slot_candidates"]})
        index = choice.get("selection_index") if isinstance(choice, dict) else None
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(state["slot_candidates"]):
            return {"selected_slot": {}, "response": tr("slot_invalid")}
        return {"selected_slot": state["slot_candidates"][index]}

    def route_after_choice(state: ProductivityState):
        return "proposal" if state.get("selected_slot") else "end"

    async def prepare_proposal(state: ProductivityState):
        analysis = RequestAnalysis.model_validate(state["analysis"])
        if analysis.mode == "web_form":
            proposal = ActionProposal.model_validate(state["form_proposal"])
            if proposal.args.get("user_id") != state["user_id"]:
                raise PermissionError("FormProposalOwnerMismatch")
            return {"proposal": proposal.model_dump()}
        reads = state.get("read_results", {})
        if analysis.mode.startswith("email_"):
            if reads.get("error"):
                return {"response": tr("email_read_error", error=safe_error(reads['error']))}
            messages = [GmailMessage.model_validate(item) for item in reads.get("email_messages", [])]
            route = RoutedRequest.model_validate(state["route"])
            try:
                if analysis.mode in {"email_to_todo", "email_to_calendar"}:
                    source = select_action_message(messages, route.email_select_latest)
                    operation = "create_todo" if analysis.mode == "email_to_todo" else "create_event"
                    if operation == "create_event" and (not calendar_owner_telegram_id or
                            state.get("telegram_id") != calendar_owner_telegram_id):
                        raise CalendarUnavailable("Calendar is not connected for this user")
                    proposal = await propose_email_action(source, state["request"], operation,
                                                          state["timezone"], get_email_llm(), calendar)
                    proposal.args["user_id"] = state["user_id"]
                    emit_trace("proposal", operation=proposal.operation,
                               action_id=proposal.action_id, success=True)
                    return {"proposal": proposal.model_dump()}
                wants_draft = analysis.mode == "email_draft" or route.email_draft_requested
                if wants_draft and messages:
                    source = select_action_message(messages, route.email_select_latest,
                                                   allow_single_thread_latest=True)
                    if gmail is None:
                        raise GmailUnavailable("Gmail is not connected")
                    messages = await gmail.thread(source.thread_id)
                account_address = await gmail.account_address() if wants_draft and messages and gmail else None
                response = await synthesize_email(
                    messages, state["request"], get_email_llm(),
                    draft=wants_draft,
                    availability=reads.get("availability") if route.email_calendar_check else None,
                    account_address=account_address)
                return {"response": response}
            except (ValueError, GmailError, CalendarError) as exc:
                emit_trace("email", operation=analysis.mode, success=False,
                           error_category=type(exc).__name__)
                return {"response": safe_error(exc)}
            except Exception:
                emit_trace("email", operation=analysis.mode, success=False,
                           error_category="email_synthesis_failed")
                return {"response": tr("email_analysis_error")}
        if analysis.mode == "meeting_prep" and not reads.get("error"):
            meeting_step = next(step for step in analysis.actions if step.operation == "list_related_todos")
            meeting, error = select_meeting(reads.get("events", []), state["timezone"],
                                            meeting_step.args.get("hour"), meeting_step.args.get("minute", 0),
                                            meeting_step.args.get("query", ""))
            if error:
                return {"response": safe_error(error)}
            from app.config import get_settings
            settings = get_settings()
            try:
                profile = None
                profile_source = None
                profile_unavailable = store is None
                for attempt in range(2):
                    _, _, anchor = await exact_anchor(calendar, meeting, timezone_name=state["timezone"])
                    if attempt == 0 and store is not None:
                        try:
                            profile_item = await store.aget(("profile", state["user_id"]), "user_profile")
                            profile = profile_item.value if profile_item else None
                            if profile_item:
                                profile_source = EvidenceEnvelope(
                                    key="P1", source_type="profile", source_id="user_profile",
                                    owner_scope=f"store:profile:{state['user_id']}",
                                    authority_domain="response_preferences",
                                    retrieved_at=datetime.now(dt_timezone.utc), currentness="current",
                                    safe_display_label="Profile: current preferences")
                        except Exception:
                            profile_unavailable = True
                    anchor_candidate = {"id": anchor.id, "calendarId": meeting.get("calendarId") or "primary",
                                        "summary": anchor.summary, "start": anchor.start,
                                        "attendees": anchor.attendees}
                    still_matches, changed = select_meeting(
                        [anchor_candidate], state["timezone"], meeting_step.args.get("hour"),
                        meeting_step.args.get("minute", 0), meeting_step.args.get("query", ""))
                    if changed or still_matches is None:
                        raise GroundingError("Selected Calendar meeting changed; please select it again")
                    memory_items, memory_degraded = await meeting_memory(state, anchor)
                    bundle = await collect_meeting_evidence(
                        user_id=state["user_id"], telegram_id=state.get("telegram_id", ""),
                        timezone_name=state["timezone"], candidate=meeting, calendar=calendar,
                        gmail=gmail, gmail_owner_telegram_id=gmail_owner_telegram_id,
                        session_factory=executor.session_factory,
                        memory_items=memory_items,
                        memory_degraded=memory_degraded,
                        token_budget=settings.meeting_prep_evidence_token_budget,
                        gmail_lookback_days=settings.meeting_prep_gmail_lookback_days,
                        gmail_candidate_limit=settings.meeting_prep_gmail_candidate_limit)
                    if bundle.event_etag != anchor.etag:
                        if attempt == 0:
                            continue
                        raise GroundingError("Calendar event changed during source selection")
                    if profile_unavailable:
                        bundle.availability["profile"] = "unavailable"
                    elif profile_source is None:
                        bundle.availability["profile"] = "not_found"
                    else:
                        bundle.availability["profile"] = "available"
                    bundle.profile_evidence = profile_source
                    brief = await synthesize_meeting_brief(
                        bundle, state["request"], profile,
                        meeting_llm if meeting_llm is not None else get_email_llm())
                    if (await revalidate_anchor(calendar, bundle) and
                            await revalidate_todos(executor.session_factory, bundle)):
                        return {"response": render_meeting_brief(brief, bundle, profile)}
                    if attempt == 1:
                        return {"response": tr("calendar_unavailable", error="Meeting changed during preparation")}
                    # Re-read the event by exact identity; never switch to a
                    # different same-title candidate during retry.
                raise GroundingError("Calendar anchor could not be revalidated")
            except (CalendarError, GmailError, GroundingError, ValueError) as exc:
                logger.warning("meeting_prep stage=source_or_validation category=%s", type(exc).__name__)
                return {"response": tr("meeting_grounding_unavailable")}
            except Exception as exc:
                logger.warning("meeting_prep stage=grounded_brief category=%s", type(exc).__name__)
                emit_trace("meeting_prep", operation="grounded_brief", success=False,
                           error_category=type(exc).__name__)
                return {"response": tr("meeting_grounding_unavailable")}
        if analysis.mode == "smart_schedule" and state.get("selected_slot"):
            slot = state["selected_slot"]
            step = next(step for step in analysis.actions if step.operation == "create_event")
            args = {"title": step.args["title"], "start": slot["start"], "end": slot["end"],
                    "timeZone": state["timezone"]}
            proposal = ActionProposal(domain="calendar", operation="create_event", args=args,
                                      human_readable_preview=(f"Create calendar event ‘{args['title']}’ "
                                                              f"from {args['start']} to {args['end']}"),
                                      reason="User selected a Calendar-verified candidate slot; checked again before execution",
                                      risk_level=RiskLevel.EXTERNAL_WRITE)
        else:
            proposal = proposal_from_plan(analysis, reads, state["timezone"],
                                          recent_event=state.get("recent_event"))
        if proposal is None:
            reads = state.get("read_results", {})
            if state.get("analysis_error"):
                return {"response": safe_error(state["analysis_error"])}
            if reads.get("error"):
                return {"response": tr("calendar_unavailable", error=safe_error(reads['error']))}
            if analysis.mode == "calendar_read":
                if "events" in reads:
                    return {"response": format_calendar_events(reads["events"], state["timezone"])}
                primary = reads.get("availability", {}).get("calendars", {}).get("primary", {})
                if isinstance(primary.get("busy"), list):
                    return {"response": tr("calendar_busy" if primary["busy"] else "calendar_free")}
                return {"response": tr("calendar_read_error")}
            if "events" in reads and len(reads["events"]) != 1:
                return {"response": tr("calendar_not_found" if not reads["events"]
                                       else "calendar_ambiguous")}
            if analysis.mode == "todo_delete":
                return {"response": tr("todo_ambiguous")}
            if analysis.mode == "todo_delete_tag":
                return {"response": (tr("todo_tag_many")
                                     if reads.get("todos") else
                                     tr("todo_tag_none"))}
            if reads.get("availability", {}).get("calendars", {}).get("primary", {}).get("busy"):
                return {"response": tr("slot_busy")}
            if not analysis.requires_plan:
                return {"response": tr("route_legacy")}
            return {"response": tr("plan_failed")}
        proposal.args = {**proposal.args, "user_id": state["user_id"]}
        try:
            await bind_approved_calendar_version(calendar, proposal)
        except CalendarError as exc:
            return {"proposal": {}, "response": safe_error(exc)}
        logger.info("user_id=%s action_id=%s proposal=%s.%s", state.get("user_id"),
                    proposal.action_id, proposal.domain, proposal.operation)
        emit_trace("proposal", operation=proposal.operation, action_id=proposal.action_id, success=True)
        return {"proposal": proposal.model_dump()}

    def route_after_proposal(state: ProductivityState):
        if not state.get("proposal"):
            return "end"
        proposal = ActionProposal.model_validate(state["proposal"])
        return "approval" if requires_approval(proposal) else "execute"

    async def approval_gate(state: ProductivityState):
        proposal = ActionProposal.model_validate(state["proposal"])
        emit_trace("interrupt", operation=proposal.operation, action_id=proposal.action_id, success=True)
        resume = interrupt({
            "kind": "action_proposal", "action_id": proposal.action_id,
            "preview": proposal.human_readable_preview,
            "risk_level": proposal.risk_level.value,
            "allowed": ["approve", "edit", "reject"],
        })
        decision = (resume or {}).get("decision", "reject") if isinstance(resume, dict) else str(resume)
        logger.info("user_id=%s action_id=%s resume=%s", state.get("user_id"), proposal.action_id, decision)
        emit_trace("resume", operation=decision, action_id=proposal.action_id, success=True)
        return {"approval_decision": decision,
                "approval_web_request_id": (resume or {}).get("web_request_id")
                if isinstance(resume, dict) else None}

    def route_after_approval(state: ProductivityState):
        decision = state.get("approval_decision")
        if decision == "approve":
            return "execute"
        if decision == "edit":
            return "edit"
        return "reject"

    async def request_edit(state: ProductivityState):
        emit_trace("interrupt", operation="edit", action_id=state["proposal"]["action_id"], success=True)
        instruction = interrupt({
            "kind": "edit_request", "action_id": state["proposal"]["action_id"],
            "message": tr("edit_request"),
        })
        web_request_id = instruction.get("web_request_id") if isinstance(instruction, dict) else None
        instruction_text = instruction.get("instruction", "") if isinstance(instruction, dict) else str(instruction)
        try:
            proposal = await apply_edit(ActionProposal.model_validate(state["proposal"]),
                                        instruction_text, state["timezone"], edit_llm)
            if proposal.operation == "create_event":
                availability = await calendar.availability(
                    proposal.args["start"], proposal.args["end"], state["timezone"]
                )
                primary = availability.get("calendars", {}).get("primary", {})
                if not isinstance(primary.get("busy"), list):
                    raise CalendarError("Calendar availability response was invalid")
                if primary["busy"]:
                    raise ValueError("The edited time is busy; no new action was proposed.")
            proposal.args["user_id"] = state["user_id"]
            await bind_approved_calendar_version(calendar, proposal)
        except (ValueError, CalendarError) as exc:
            logger.warning("user_id=%s calendar edit rejected category=%s", state.get("user_id"),
                           diagnostic_error(exc))
            return {"proposal": {}, "response": safe_error(exc), "approval_decision": "",
                    "edit_web_request_id": web_request_id}
        logger.info("user_id=%s action_id=%s edited proposal=%s.%s", state.get("user_id"),
                    proposal.action_id, proposal.domain, proposal.operation)
        return {"proposal": proposal.model_dump(), "approval_decision": "",
                "edit_web_request_id": web_request_id}

    def route_after_edit(state: ProductivityState):
        return "approval" if state.get("proposal") else "end"

    async def execute(state: ProductivityState, config: RunnableConfig):
        proposal = ActionProposal.model_validate(state["proposal"])
        if proposal.domain == "calendar" and calendar_owner_telegram_id is not None and (
                not calendar_owner_telegram_id or state.get("telegram_id") != calendar_owner_telegram_id):
            return {"response": tr("calendar_not_connected"),
                    "proposal": proposal.model_copy(update={"status": ActionStatus.FAILED}).model_dump()}
        if proposal.operation == "create_event" and not await executor.calendar_create_prepared(
                proposal.action_id):
            try:
                start = datetime.fromisoformat(proposal.args["start"])
                end = datetime.fromisoformat(proposal.args["end"])
                if start.tzinfo is None or end.tzinfo is None or end <= start or end <= datetime.now(start.tzinfo):
                    raise ValueError("The selected slot is no longer valid; no Calendar event was created.")
                if proposal.args.get("timeZone") != state["timezone"]:
                    raise ValueError("The proposed timezone changed; no Calendar event was created.")
                fresh = await calendar.availability(proposal.args["start"], proposal.args["end"], state["timezone"])
                if not verified_free_interval(start, end, fresh):
                    raise CalendarError("The selected time is now busy or could not be verified.")
            except (ValueError, CalendarError) as exc:
                emit_trace("execute", operation="create_event", action_id=proposal.action_id,
                           success=False, error_category=type(exc).__name__)
                return {"response": safe_error(exc), "proposal": proposal.model_copy(
                    update={"status": ActionStatus.FAILED}).model_dump()}
        result = await executor.execute(proposal)
        emit_trace("execute", operation=proposal.operation, action_id=proposal.action_id,
                   success=result.status == ActionStatus.EXECUTED,
                   external_id=result.result.get("id"), error_category=(
                       "action_failed" if result.status != ActionStatus.EXECUTED else None))
        result_text = (tr("action_completed") if result.status == ActionStatus.EXECUTED else
                       tr("action_review_required") if result.result.get("remote_state") == "PARKED_REVIEW" else
                       tr("action_unconfirmed") if result.status == ActionStatus.EXECUTING else
                       tr("action_duplicate") if result.message == "Action was already processed." else
                       tr("action_failed", error=safe_error(result.message)))
        update = {"response": result_text, "proposal": proposal.model_copy(update={"status": result.status}).model_dump()}
        if (proposal.operation == "create_event" and result.status == ActionStatus.EXECUTING and
                RequestAnalysis.model_validate(state.get("analysis", {})).mode == "cross_domain" and
                any(step.operation == "create_todo" for step in
                    RequestAnalysis.model_validate(state.get("analysis", {})).actions)):
            resume_checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
            if resume_checkpoint_id is None and checkpointer is not None:
                # Save a pre-wait cursor *before* the graph yields control.
                # Binding the later interrupt checkpoint is an optimization;
                # a crash immediately after the interrupt must still be
                # recoverable even if another request advances the thread.
                checkpoint = await checkpointer.aget_tuple({
                    "configurable": {"thread_id": state["workflow_thread_id"]}})
                if checkpoint is not None:
                    resume_checkpoint_id = checkpoint.config.get(
                        "configurable", {}).get("checkpoint_id")
            await continuation_store.register_waiting(
                proposal.action_id, state["workflow_thread_id"],
                resume_checkpoint_id=resume_checkpoint_id)
        if result.status == ActionStatus.EXECUTED and proposal.domain == "calendar":
            if proposal.operation == "delete_event":
                update["recent_event"] = {}
            else:
                update["recent_event"] = {
                    "id": result.result.get("id") or proposal.args.get("eventId"),
                    "title": proposal.args.get("title") or state.get("recent_event", {}).get("title"),
                    "start": proposal.args.get("start") or state.get("recent_event", {}).get("start"),
                    "end": proposal.args.get("end") or state.get("recent_event", {}).get("end"),
                }
        return update

    def route_after_execute(state: ProductivityState):
        proposal = ActionProposal.model_validate(state["proposal"])
        analysis = RequestAnalysis.model_validate(state.get("analysis", {}))
        if (proposal.operation == "create_event" and analysis.mode == "cross_domain" and
                any(step.operation == "create_todo" for step in analysis.actions)):
            if proposal.status == ActionStatus.EXECUTED:
                return "child"
            if proposal.status == ActionStatus.EXECUTING:
                return "wait"
        return "end"

    async def wait_for_calendar(state: ProductivityState):
        proposal = ActionProposal.model_validate(state["proposal"])
        # This node is restarted by Command(resume). R5 evidence, not the
        # resume payload, determines whether the approved child may run.
        result = await executor.execute(proposal)
        if result.status == ActionStatus.EXECUTED:
            return {"proposal": proposal.model_copy(update={"status": ActionStatus.EXECUTED}).model_dump(),
                    "response": tr("action_completed")}
        interrupt({"kind": "calendar_confirmation", "action_id": proposal.action_id,
                   "message": (tr("action_review_required") if result.result.get("remote_state") == "PARKED_REVIEW"
                               else tr("action_unconfirmed"))})

    async def execute_child_todo(state: ProductivityState):
        proposal = ActionProposal.model_validate(state["proposal"])
        parent = await executor.execute(proposal)
        if parent.status != ActionStatus.EXECUTED:
            # Never let an edited checkpoint or stale wakeup authorize Todo.
            return {"response": tr("action_unconfirmed")}
        analysis = RequestAnalysis.model_validate(state["analysis"])
        todo_step = next(step for step in analysis.actions if step.operation == "create_todo")
        child = ActionProposal(
            action_id=f"{proposal.action_id}:todo", domain="todo", operation="create_todo",
            args={**todo_step.args, "user_id": state["user_id"]},
            human_readable_preview=f"Create Todo: {todo_step.args['title']}",
            reason="Approved cross-domain productivity plan",
            risk_level=RiskLevel.INTERNAL_LOW_RISK)
        result = await executor.execute_child_todo(child)
        return {"response": (tr("cross_domain_done") if result.status == ActionStatus.EXECUTED
                             else tr("cross_domain_partial", error=safe_error(result.message)))}

    async def reject(state: ProductivityState):
        proposal = ActionProposal.model_validate(state["proposal"])
        return {"response": tr("action_rejected"),
                "proposal": proposal.model_copy(update={"status": ActionStatus.REJECTED}).model_dump()}

    builder = StateGraph(ProductivityState)
    builder.add_node("analyze", analyze)
    builder.add_node("perform_reads", perform_reads)
    builder.add_node("build_candidates", build_candidates)
    builder.add_node("choose_slot", choose_slot)
    builder.add_node("prepare_proposal", prepare_proposal)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("request_edit", request_edit)
    builder.add_node("execute", execute)
    builder.add_node("wait_for_calendar", wait_for_calendar)
    builder.add_node("execute_child_todo", execute_child_todo)
    builder.add_node("reject", reject)
    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "perform_reads")
    builder.add_conditional_edges("perform_reads", lambda state: (
        "candidates" if state.get("analysis", {}).get("mode") == "smart_schedule" else "proposal"),
        {"candidates": "build_candidates", "proposal": "prepare_proposal"})
    builder.add_conditional_edges("build_candidates", route_after_candidates,
                                  {"choose": "choose_slot", "end": END})
    builder.add_conditional_edges("choose_slot", route_after_choice,
                                  {"proposal": "prepare_proposal", "end": END})
    builder.add_conditional_edges("prepare_proposal", route_after_proposal, {"approval": "approval_gate", "execute": "execute", "end": END})
    builder.add_conditional_edges("approval_gate", route_after_approval, {"execute": "execute", "edit": "request_edit", "reject": "reject"})
    builder.add_conditional_edges("request_edit", route_after_edit, {"approval": "approval_gate", "end": END})
    builder.add_conditional_edges("execute", route_after_execute,
                                  {"child": "execute_child_todo", "wait": "wait_for_calendar", "end": END})
    builder.add_conditional_edges("wait_for_calendar", route_after_execute,
                                  {"child": "execute_child_todo", "wait": "wait_for_calendar", "end": END})
    builder.add_edge("execute_child_todo", END)
    builder.add_edge("reject", END)
    graph = builder.compile(checkpointer=checkpointer)
    graph.calendar_continuation_store = continuation_store
    return graph


async def start_workflow(graph, user_id: str, request: str, timezone: str, thread_id: str,
                         route: RoutedRequest | None = None,
                         recent_context: list[dict[str, str]] | None = None,
                         telegram_id: str | None = None, turn_id: str | None = None,
                         memory_history_threads: tuple[str, ...] = (),
                         form_proposal: dict[str, Any] | None = None):
    from app.telegram.processing_context import current_run
    run = current_run()
    turn_id = f"telegram-receipt:{run.receipt_id}" if run is not None else turn_id or str(uuid4())
    if run is not None or turn_id.startswith("web-request:"):
        # Saver can commit the interrupt/final result before the receipt is
        # finalized. Reconstruct that result rather than starting the same
        # Telegram update as a second workflow invocation.
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        if snapshot and snapshot.values.get("transcript_turn_id") == turn_id:
            interrupts = [item for task in snapshot.tasks
                          for item in getattr(task, "interrupts", ())]
            if interrupts:
                return {"__interrupt__": interrupts, **snapshot.values}
            if snapshot.next:
                return await graph.ainvoke(None, {"configurable": {"thread_id": thread_id}})
            return snapshot.values
    return await graph.ainvoke({"user_id": user_id, "telegram_id": telegram_id or "",
                                "request": request, "timezone": timezone,
                                "workflow_thread_id": thread_id,
                                "transcript_turn_id": turn_id, "response": "",
                                "transcript_occurred_at": datetime.now(dt_timezone.utc).isoformat(),
                                "trace_origin_id": current_trace_id(),
                                "precomputed_route": route.model_dump() if route else None,
                                "form_proposal": form_proposal,
                                "recent_context": recent_context or [],
                                "memory_history_threads": list(memory_history_threads)},
                               {"configurable": {"thread_id": thread_id}})


async def read_workflow_turn(graph, thread_id: str, *, turn_id: str, user_id: str) -> dict:
    """Read back a handoff without resuming or executing the workflow again."""
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if (not snapshot or snapshot.values.get("user_id") != user_id or
            snapshot.values.get("transcript_turn_id") != turn_id):
        raise RuntimeError("WorkflowHandoffMismatch")
    interrupts = [item for task in snapshot.tasks
                  for item in getattr(task, "interrupts", ())]
    if snapshot.next and not interrupts:
        # A running checkpoint is not a completed business result.  Let the
        # durable worker retry/read back rather than reporting success.
        raise RuntimeError("WorkflowHandoffIncomplete")
    return {"__interrupt__": interrupts, **snapshot.values} if interrupts else snapshot.values


async def resume_workflow(graph, thread_id: str, payload: dict[str, Any] | str,
                          *, context: dict | None = None):
    kwargs = {"context": context} if context is not None else {}
    result = await graph.ainvoke(Command(resume=payload),
                                {"configurable": {"thread_id": thread_id}}, **kwargs)
    await bind_continuation_checkpoint(graph, thread_id, result)
    return result


async def bind_continuation_checkpoint(graph, thread_id: str, result: dict) -> None:
    if not any(getattr(item, "value", {}).get("kind") == "calendar_confirmation"
               for item in result.get("__interrupt__", [])):
        return
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if snapshot and snapshot.values.get("proposal"):
        await graph.calendar_continuation_store.bind_checkpoint(
            snapshot.values["proposal"]["action_id"], thread_id,
            snapshot.config["configurable"]["checkpoint_id"])


async def workflow_waits_for_edit(graph, thread_id: str) -> bool:
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return bool(snapshot and "request_edit" in snapshot.next)

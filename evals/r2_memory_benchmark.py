"""Isolated, synthetic R2 matched retrieval and real-summary benchmark.

Never queries production transcripts. PostgreSQL uses a TEMP table matching the
search projection's relevant columns. Real provider calls contain only synthetic
text. Optional R2_EMBEDDING_* variables select an independent experiment backend.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import psycopg
from openai import OpenAI
from dotenv import dotenv_values

from app.config import get_settings


@dataclass(frozen=True)
class Turn:
    user_id: str
    thread_id: str
    message_id: str
    turn: int
    role: str
    content: str
    supersedes: str | None = None


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    gold: str | None
    user_id: str = "synthetic-a"
    thread_id: str = "synthetic-main"
    source_of_truth: str = "CONVERSATION_HISTORY"
    allow_stale: bool = False


def corpus() -> tuple[list[Turn], list[Case]]:
    turns: list[Turn] = []

    def add(n, content, *, user="synthetic-a", thread="synthetic-main", supersedes=None):
        turns.append(Turn(user, thread, f"m-{user}-{thread}-{n}", n, "user", content, supersedes))

    facts = {
        2: "Blue Heron project review code AX-4731; prototype budget exactly 813 dollars.",
        10: "Cedar pilot discussion with Mira is Thursday. This is the old plan.",
        35: "Correction: Cedar pilot discussion with Mira moved to Monday.",
        50: "蓝鲸项目的内部代号是 海风七号，验收编号 CN-8042。",
        62: "Aurora report outline alias Pine Draft. Exact page count 47.",
        70: "Blue Finch project review code BF-9920; budget is 410 dollars.",
        83: "For Atlas workshop, my old nickname is lighthouse folder.",
        95: "Old alias for internship preparation task is comet folder.",
        110: "极光报告目前有 47 页，旧数据。",
        120: "Correction: Aurora report now has 52 pages, not 47.",
        130: "蓝鲸项目的下一次讨论主题是预算风险，不是功能演示。",
        145: "Second correction: Aurora report now has 58 pages, replacing 52.",
        154: "Recent note: I call the current prototype Beta Kite.",
        159: "For the two-word project Maple Grove, the decision is to defer launch.",
        160: "Today's current meeting shorthand is lighthouse review.",
    }
    supersedes = {35: 10, 120: 62, 145: 120}
    for n in range(1, 161):
        add(n, facts.get(n, f"Routine discussion turn {n}: unrelated coffee and planning topic {n % 19}."),
            supersedes=f"m-synthetic-a-synthetic-main-{supersedes[n]}" if n in supersedes else None)
    add(43, "Blue Heron review code ZZ-0000.", user="synthetic-b")
    add(44, "Aurora report page count is 999.", thread="synthetic-other")
    add(58, "Maple Grove project decided immediate launch.", user="synthetic-b")
    turns.sort(key=lambda x: (x.user_id, x.thread_id, x.turn))

    def case(name, query, turn, **kw):
        user, thread = kw.get("user_id", "synthetic-a"), kw.get("thread_id", "synthetic-main")
        return Case(name, query, f"m-{user}-{thread}-{turn}" if turn is not None else None, **kw)

    cases = [
        case("old_150_code", "Blue Heron review code AX-4731", 2),
        case("old_150_budget", "Blue Heron prototype spending limit", 2),
        case("old_100_alias", "internship preparation old alias", 95),
        case("old_100_exact_name", "What is the Cedar pilot discussion day now?", 35),
        case("old_50_outline", "Aurora report outline alias", 62, allow_stale=True),
        case("old_30_topic", "蓝鲸项目讨论什么主题", 130),
        case("recent_fact", "current prototype Beta Kite", 154),
        case("recent_reference", "current meeting shorthand", 160),
        case("recent_pronoun", "What did I just call it?", 160),
        case("exact_number_current", "Aurora report current page count", 145),
        case("multiple_correction", "Aurora report 58 pages", 145),
        case("original_wording", "lighthouse folder", 83),
        case("exact_project", "Maple Grove launch decision", 159),
        case("chinese_code", "蓝鲸项目验收编号 CN-8042", 50),
        case("chinese_name", "蓝鲸项目的内部代号", 50),
        case("mixed_en_zh", "蓝鲸 project code CN-8042", 50),
        case("paraphrase_budget", "bird-named project spending limit", 2),
        case("synonym_decision", "which project postponed release", 159),
        case("other_user_isolation", "Blue Heron review code", 43, user_id="synthetic-b"),
        case("other_thread_isolation", "Aurora page count", 44, thread_id="synthetic-other"),
        case("similar_topic_noise", "Maple Grove launch decision", 159),
        case("similar_project", "Blue Heron code, not Blue Finch", 2),
        case("ambiguous_reference", "What was that project code?", None,
             source_of_truth="CONVERSATION_HISTORY"),
        # These sources outrank retrieved conversation. They are corpus policy
        # checks, not scored as historical retrieval answers.
        case("profile_conflict", "current assistant language", None, source_of_truth="PROFILE_STORE"),
        case("todo_conflict", "is my task complete now", None, source_of_truth="TODO_DB"),
        case("calendar_conflict", "what time is current event", None, source_of_truth="CALENDAR"),
        case("gmail_conflict", "what does current email say", None, source_of_truth="GMAIL"),
    ]
    return turns, cases


def current_ids(turns: list[Turn]) -> set[str]:
    return {turn.supersedes for turn in turns if turn.supersedes}


def lexical(db_url: str, turns: list[Turn], cases: list[Case], *, filter_stale: bool):
    ranks, latencies = {}, []
    start_insert = time.perf_counter()
    with psycopg.connect(db_url, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""CREATE TEMP TABLE r2_transcript_projection
                (user_id text, thread_id text, message_id text, group_id text,
                 role text, occurred_at timestamptz, search_text text,
                 source_ref text, supersedes_message_id text, deleted_at timestamptz)
                 ON COMMIT PRESERVE ROWS""")
            stamp = datetime(2026, 10, 4, tzinfo=timezone.utc)
            cursor.executemany("INSERT INTO r2_transcript_projection VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NULL)",
                [(t.user_id, t.thread_id, t.message_id, t.message_id, t.role, stamp,
                  t.content, f"synthetic:{t.message_id}", t.supersedes) for t in turns])
            cursor.execute("CREATE INDEX r2_fts ON r2_transcript_projection USING GIN (to_tsvector('simple', search_text))")
            insert_ms = round((time.perf_counter() - start_insert) * 1000, 2)
            stale = current_ids(turns)
            for case in cases:
                if case.gold is None:
                    continue
                start = time.perf_counter()
                if re.search(r"[\u3400-\u9fff]", case.query):
                    segments = re.findall(r"[\u3400-\u9fff]+|[A-Za-z]+-[0-9]+", case.query)
                    chunks = []
                    for segment in segments:
                        chunks.extend([segment[i:i+2] for i in range(len(segment)-1)]
                                      if re.search(r"[\u3400-\u9fff]",segment) and len(segment)>2 else [segment])
                    chunks = list(dict.fromkeys(chunks))[:16] or [case.query]
                    # Explicit substring baseline; default PostgreSQL FTS does
                    # not segment Chinese words. This may require a later index.
                    clauses = " OR ".join("search_text ILIKE %s" for _ in chunks)
                    rank = " + ".join("CASE WHEN search_text ILIKE %s THEN 1 ELSE 0 END" for _ in chunks)
                    cursor.execute(f"SELECT message_id FROM r2_transcript_projection WHERE user_id=%s AND thread_id=%s AND ({clauses}) ORDER BY ({rank}) DESC, message_id DESC LIMIT 10",
                                   (case.user_id, case.thread_id, *[f"%{chunk}%" for chunk in chunks],
                                    *[f"%{chunk}%" for chunk in chunks]))
                else:
                    terms = [term for term in re.findall(r"[a-z0-9]+", case.query.lower())
                             if term not in {"what", "which", "how", "the", "is", "was", "did", "my", "for", "to", "at", "now", "current"}]
                    tsq = " | ".join(terms)
                    cursor.execute("""SELECT message_id FROM r2_transcript_projection
                        WHERE user_id=%s AND thread_id=%s AND to_tsvector('simple', search_text) @@ to_tsquery('simple',%s)
                        ORDER BY ts_rank_cd(to_tsvector('simple',search_text),to_tsquery('simple',%s)) DESC, message_id DESC LIMIT 10""",
                        (case.user_id, case.thread_id, tsq, tsq))
                ids = [row[0] for row in cursor.fetchall()]
                ranks[case.name] = [identifier for identifier in ids if case.allow_stale or not filter_stale or identifier not in stale][:3]
                latencies.append((time.perf_counter() - start) * 1000)
            cursor.execute("SELECT pg_total_relation_size('r2_transcript_projection'::regclass), pg_relation_size('r2_fts'::regclass)")
            table_bytes, index_bytes = cursor.fetchone()
    return ranks, {"insert_and_index_ms": insert_ms, "mean_query_ms": round(sum(latencies)/len(latencies), 3),
                   "table_bytes": table_bytes, "index_bytes": index_bytes}


def semantic(turns: list[Turn], cases: list[Case]):
    settings = get_settings()
    local = dotenv_values(".env")
    base = os.getenv("R2_EMBEDDING_BASE_URL") or local.get("R2_EMBEDDING_BASE_URL") or settings.openai_base_url
    key = os.getenv("R2_EMBEDDING_API_KEY") or local.get("R2_EMBEDDING_API_KEY") or "not-required"
    model = os.getenv("R2_EMBEDDING_MODEL") or local.get("R2_EMBEDDING_MODEL") or "text-embedding-3-small"
    client = OpenAI(api_key=key, base_url=base, timeout=20, max_retries=0)
    inputs = [t.content for t in turns] + [c.query for c in cases if c.gold]
    start = time.perf_counter()
    result = client.embeddings.create(model=model, input=inputs)
    latency = (time.perf_counter() - start) * 1000
    vectors = [item.embedding for item in sorted(result.data, key=lambda item: item.index)]
    if len(vectors) != len(inputs):
        raise ValueError("embedding provider returned incomplete batch")
    dim = len(vectors[0])
    current = current_ids(turns)
    ranks = {}
    query_start = time.perf_counter()
    evaluated = [c for c in cases if c.gold]
    for index, case in enumerate(evaluated):
        vector = vectors[len(turns)+index]
        norm = math.sqrt(sum(x*x for x in vector))
        scores = []
        for turn, candidate in zip(turns, vectors[:len(turns)]):
            if (turn.user_id, turn.thread_id) != (case.user_id, case.thread_id):
                continue
            if not case.allow_stale and turn.message_id in current:
                continue
            cosine = sum(a*b for a,b in zip(vector,candidate)) / (norm * math.sqrt(sum(x*x for x in candidate)))
            scores.append((cosine,turn.message_id))
        ranks[case.name] = [identifier for _,identifier in sorted(scores,reverse=True)[:3]]
    usage = getattr(result, "usage", None)
    return ranks, {"model":model,"dimension":dim,"batch_latency_ms":round(latency,2),
                   "ranking_ms":round((time.perf_counter()-query_start)*1000,2),
                   "embedding_input_tokens":getattr(usage,"total_tokens",None),
                   "float32_storage_bytes":len(turns)*dim*4}


def summary_batch_ranges(own: list[Turn], *, recent_suffix: int = 8,
                         token_threshold: int = 1800) -> list[tuple[int,int]]:
    """Use message-aware approximate tokens; never count a bare string as messages."""
    from langchain_core.messages.utils import count_tokens_approximately
    from langchain_core.messages import HumanMessage
    summarizable = own[:-recent_suffix]
    ranges = []
    cursor = 0
    while cursor < len(summarizable):
        end = cursor
        while end < len(summarizable):
            candidate = summarizable[cursor:end+1]
            lines = "\n".join(f"[{t.message_id}] {t.content}" for t in candidate)
            if end > cursor and count_tokens_approximately([HumanMessage(content=lines)]) >= token_threshold:
                break
            end += 1
        ranges.append((cursor,end))
        cursor = end
    return ranges


def incremental_summary(turns: list[Turn]):
    """Real non-production model summary with cursor; failure never advances it."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                    timeout=30, max_retries=0)
    own = [turn for turn in turns if (turn.user_id,turn.thread_id)==("synthetic-a","synthetic-main")]
    # Retain the raw recent suffix outside the summary. Approximate tokens
    # trigger an update; actual provider usage is recorded separately.
    summarizable = own[:-8]
    summary, cursor = "", 0
    input_tokens = output_tokens = 0
    calls = 0
    start = time.perf_counter()
    for start_cursor, end in summary_batch_ranges(own):
        if start_cursor != cursor:
            raise AssertionError("summary cursor must follow successful batches")
        batch = summarizable[cursor:end]
        prompt = ("Update a running summary of synthetic conversation facts. Retain exact numbers, names, "
                  "aliases and source message IDs. New explicit corrections supersede old current facts; "
                  "preserve older wording only as historical evidence. Do not invent facts. "
                  "Keep under 450 words and return summary text, not reasoning. "
                  f"Previous summary:\n{summary}\nNew messages:\n" +
                  "\n".join(f"[{t.message_id}] {t.content}" for t in batch))
        response = client.chat.completions.create(model=settings.llm_model, temperature=0,
            messages=[{"role":"user","content":prompt}], max_tokens=2048,
            extra_body={"chat_template_kwargs":{"enable_thinking":False}})
        next_summary = response.choices[0].message.content or ""
        calls += 1
        usage = response.usage
        if usage:
            input_tokens += usage.prompt_tokens or 0
            output_tokens += usage.completion_tokens or 0
        if not next_summary.strip():
            # Some reasoning-capable providers consume the output budget before
            # emitting content. Retry without advancing the summary boundary.
            response = client.chat.completions.create(model=settings.llm_model, temperature=0,
                messages=[{"role":"user","content":prompt}], max_tokens=4096,
                extra_body={"chat_template_kwargs":{"enable_thinking":False}})
            next_summary = response.choices[0].message.content or ""
            calls += 1
            usage = response.usage
            if usage:
                input_tokens += usage.prompt_tokens or 0
                output_tokens += usage.completion_tokens or 0
        if not next_summary.strip():
            raise ValueError(f"empty summary at cursor={cursor}; cursor not advanced")
        summary = next_summary
        cursor = end
    return summary, {"summary_cursor":cursor,"summarizer_calls":calls,
                     "summarizer_input_tokens":input_tokens,"summarizer_output_tokens":output_tokens,
                     "summary_latency_ms":round((time.perf_counter()-start)*1000,2),
                     "summary_chars":len(summary)}


def summary_query_rankings(summary: str, cases: list[Case]):
    """Select cited source IDs using the real model; never inspect gold labels."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url,
                    timeout=20, max_retries=0)
    source_ids = set(re.findall(r"m-synthetic-[a-z]+-synthetic-[a-z]+-\d+", summary))
    ranking = {}
    calls = 0
    input_tokens = output_tokens = 0
    failures = []
    start = time.perf_counter()
    for case in cases:
        if case.gold is None:
            continue
        if (case.user_id, case.thread_id) != ("synthetic-a", "synthetic-main"):
            ranking[case.name] = []
            continue
        prompt = ("Synthetic source-ID retrieval only. Based solely on the following running summary, "
                  "return up to three exact source message IDs supporting the question, or NONE. "
                  "Do not infer IDs not present in the summary.\n"
                  f"Summary:\n{summary}\nQuestion: {case.query}")
        try:
            result = client.chat.completions.create(model=settings.llm_model, temperature=0,
                messages=[{"role":"user","content":prompt}], max_tokens=256,
                extra_body={"chat_template_kwargs":{"enable_thinking":False}})
            calls += 1
            usage = result.usage
            if usage:
                input_tokens += usage.prompt_tokens or 0
                output_tokens += usage.completion_tokens or 0
            answer = result.choices[0].message.content or ""
            ids = list(dict.fromkeys(re.findall(r"m-synthetic-[a-z]+-synthetic-[a-z]+-\d+",answer)))
            ranking[case.name] = [identifier for identifier in ids if identifier in source_ids][:3]
            if not answer.strip():
                failures.append({"case":case.name,"category":"EmptyModelOutput"})
        except Exception as exc:
            ranking[case.name] = []
            failures.append({"case":case.name,"category":type(exc).__name__})
    return ranking, {"query_model_calls":calls,"query_input_tokens":input_tokens,
                     "query_output_tokens":output_tokens,
                     "query_latency_ms":round((time.perf_counter()-start)*1000,2),
                     "query_failures":failures,
                     "source_ids_present":len(source_ids)}


def fuse(left: list[str], right: list[str], k: int = 8) -> list[str]:
    """Reciprocal-rank fusion (k=60), no relevance-score calibration required."""
    scores = {}
    for ranking in (left,right):
        for index, identifier in enumerate(ranking):
            scores[identifier] = scores.get(identifier,0) + 1/(60+index+1)
    return [identifier for identifier,_ in sorted(scores.items(),key=lambda pair:pair[1],reverse=True)[:k]]


def score(ranks: dict[str,list[str]], cases: list[Case], turns: list[Turn]):
    from langchain_core.messages import HumanMessage
    from langchain_core.messages.utils import count_tokens_approximately
    evaluated = [c for c in cases if c.gold]
    stale = current_ids(turns)
    hits = [c.gold in ranks.get(c.name,[]) for c in evaluated]
    hits_at_3 = [c.gold in ranks.get(c.name,[])[:3] for c in evaluated]
    top = [bool(ranks.get(c.name)) and ranks[c.name][0] == c.gold for c in evaluated]
    count = sum(len(ranks.get(c.name,[])) for c in evaluated)
    count_at_3 = sum(len(ranks.get(c.name,[])[:3]) for c in evaluated)
    content_by_id = {turn.message_id: turn.content for turn in turns}
    context_token_estimates = [count_tokens_approximately([
        HumanMessage(content="\n".join(content_by_id.get(identifier, "")
                                        for identifier in ranks.get(case.name, [])))])
        for case in evaluated]
    return {"case_count":len(evaluated),"recall_at_3":round(sum(hits_at_3)/len(hits_at_3),3),
            "recall_at_8":round(sum(hits)/len(hits),3),
            "top_1":round(sum(top)/len(top),3),
            "precision_at_3":round(sum(hits_at_3)/count_at_3,3) if count_at_3 else None,
            "precision_at_8":round(sum(hits)/count,3) if count else None,
            "irrelevant_context_items":count-sum(hits),
            "mean_retrieved_context_tokens_approx":round(sum(context_token_estimates)/len(evaluated),1),
            "stale_top_1":sum(bool(ranks.get(c.name)) and ranks[c.name][0] in stale and not c.allow_stale for c in evaluated),
            "failures_at_8":[c.name for c,hit in zip(evaluated,hits) if not hit],
            "case_hits_at_3":{c.name:hit for c,hit in zip(evaluated,hits_at_3)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Call real configured chat model on synthetic turns")
    parser.add_argument("--semantic", action="store_true", help="Call real experimental embedding provider")
    parser.add_argument("--output", default="evals/reports/r2_memory_baseline.json")
    args = parser.parse_args()
    turns, cases = corpus()
    report = {"kind":"isolated_synthetic_r2_memory_benchmark", "turns":len(turns),
              "cases":len(cases),"timestamp":datetime.now(timezone.utc).isoformat(),
              "unscored_policy_cases":[asdict(c) for c in cases if c.gold is None],"modes":{},"costs":{}}
    recent = {c.name:[t.message_id for t in reversed(turns) if (t.user_id,t.thread_id)==(c.user_id,c.thread_id)
                      and t.turn>=153][:8] for c in cases if c.gold}
    report["modes"]["recent_only"] = score(recent,cases,turns)
    try:
        raw, costs = lexical(get_settings().database_url_sync,turns,cases,filter_stale=False)
        current, _ = lexical(get_settings().database_url_sync,turns,cases,filter_stale=True)
        report["modes"]["lexical_raw"] = score(raw,cases,turns)
        report["modes"]["lexical_current"] = score(current,cases,turns)
        report["modes"]["recent_plus_lexical"] = score(
            {c.name:fuse(recent[c.name],current[c.name]) for c in cases if c.gold},cases,turns)
        report["costs"]["lexical"] = costs
    except Exception as exc:
        report["modes"]["lexical_current"] = {"unavailable":type(exc).__name__}
        current = None
    semantic_rank = None
    if args.semantic:
        try:
            semantic_rank, costs = semantic(turns,cases)
            report["modes"]["semantic_only"] = score(semantic_rank,cases,turns)
            report["modes"]["recent_plus_semantic"] = score(
                {c.name:fuse(recent[c.name],semantic_rank[c.name]) for c in cases if c.gold},cases,turns)
            report["costs"]["semantic"] = costs
        except Exception as exc:
            report["modes"]["semantic_only"] = {"unavailable":type(exc).__name__}
    else:
        report["modes"]["semantic_only"] = {"unavailable":"NOT_RUN"}
    if current and semantic_rank:
        hybrid = {c.name:fuse(current[c.name],semantic_rank[c.name]) for c in cases if c.gold}
        report["modes"]["hybrid"] = score(hybrid,cases,turns)
        report["modes"]["recent_plus_hybrid"] = score(
            {c.name:fuse(recent[c.name],hybrid[c.name]) for c in cases if c.gold},cases,turns)
    else:
        report["modes"]["hybrid"] = {"unavailable":"SEMANTIC_UNAVAILABLE"}
    if args.summary:
        try:
            summary, costs = incremental_summary(turns)
            # Source-ID presence is a diagnostic ceiling, not query-time
            # retrieval. Model-selected IDs below are scored only afterward.
            cited_ids = set(re.findall(r"m-synthetic-[a-z]+-synthetic-[a-z]+-\d+",summary))
            own_cases = [c for c in cases if c.gold and
                         (c.user_id,c.thread_id)==("synthetic-a","synthetic-main")]
            report["summary_diagnostic"] = {
                "gold_source_id_presence":round(sum(c.gold in cited_ids for c in own_cases)/len(own_cases),3),
                "not_a_retrieval_metric":True}
            summary_rank, query_costs = summary_query_rankings(summary,cases)
            report["modes"]["summary_only"] = score(summary_rank,cases,turns)
            report["modes"]["summary_plus_recent"] = score(
                {c.name:fuse(summary_rank[c.name],recent[c.name]) for c in cases if c.gold},cases,turns)
            if current:
                report["modes"]["summary_plus_lexical_plus_recent"] = score(
                    {c.name:fuse(fuse(summary_rank[c.name],current[c.name]),recent[c.name])
                     for c in cases if c.gold},cases,turns)
            if current and semantic_rank:
                report["modes"]["summary_plus_hybrid_plus_recent"] = score(
                    {c.name:fuse(fuse(summary_rank[c.name],hybrid[c.name]),recent[c.name])
                     for c in cases if c.gold},cases,turns)
            report["costs"]["summary"] = costs
            report["costs"]["summary_query"] = query_costs
        except Exception as exc:
            report["modes"]["summary_plus_recent"] = {"unavailable":type(exc).__name__}
    else:
        report["modes"]["summary_plus_recent"] = {"unavailable":"NOT_RUN"}
    report["unavailable_metrics"] = ["main_model_input_tokens", "answer_accuracy", "embedding_cost_currency",
                                      "retrieved_context_tokens", "total_pipeline_latency"]
    from pathlib import Path
    target = Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"report":str(target),"modes":report["modes"],"costs":report["costs"]},ensure_ascii=False))


if __name__ == "__main__":
    main()

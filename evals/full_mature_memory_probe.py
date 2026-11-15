"""Isolated, synthetic memory-retrieval probe; never touches product tables.

Run from the repository root with .venv312/Scripts/python.exe
evals/full_mature_memory_probe.py. PostgreSQL uses a TEMP table that disappears
when the connection closes. Failed provider calls are reported as unavailable.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean

import psycopg
from openai import OpenAI

from app.config import get_settings


@dataclass(frozen=True)
class Turn:
    user_id: str
    thread_id: str
    message_id: str
    turn: int
    role: str
    timestamp: str
    content: str
    supersedes: str | None = None


@dataclass(frozen=True)
class Probe:
    label: str
    query: str
    gold: str
    user_id: str = "synthetic-a"
    thread_id: str = "synthetic-main"


def corpus() -> tuple[list[Turn], list[Probe]]:
    stamp = datetime(2026, 10, 4, tzinfo=timezone.utc).isoformat()
    turns: list[Turn] = []

    def add(n: int, content: str, *, user: str = "synthetic-a", thread: str = "synthetic-main", supersedes=None):
        turns.append(Turn(user, thread, f"m-{user}-{thread}-{n}", n, "user", stamp, content, supersedes))

    add(2, "My project Blue Heron uses review code AX-4731; the prototype budget is 813 dollars.")
    add(10, "The Cedar pilot discussion is with Mira on Thursday. This is the old plan.")
    add(35, "Correction to the Cedar pilot: discussion with Mira moved to Monday.", supersedes="m-synthetic-a-synthetic-main-10")
    add(62, "For the Aurora report, call the outline Pine Draft; its exact page count is 47.")
    add(95, "Old alias for my internship preparation task is comet folder.")
    add(120, "Correction: the Aurora report now has 52 pages, not 47.", supersedes="m-synthetic-a-synthetic-main-62")
    for n in range(1, 161):
        if n in {2, 10, 35, 62, 95, 120}:
            continue
        add(n, f"Routine discussion turn {n}: tea break and general planning note for unrelated topic {n % 17}.")
    add(43, "My project Blue Heron review code is ZZ-0000.", user="synthetic-b")
    add(44, "Aurora report page count is 999.", user="synthetic-a", thread="synthetic-other")
    turns.sort(key=lambda x: (x.user_id, x.thread_id, x.turn))
    probes = [
        Probe("exact_code", "What is the Blue Heron review code?", "m-synthetic-a-synthetic-main-2"),
        Probe("exact_budget", "How many dollars is the Blue Heron prototype budget?", "m-synthetic-a-synthetic-main-2"),
        Probe("correction", "When is the Cedar pilot discussion with Mira now?", "m-synthetic-a-synthetic-main-35"),
        Probe("project_alias", "What name did I give the Aurora report outline?", "m-synthetic-a-synthetic-main-62"),
        Probe("corrected_number", "What is the current Aurora report page count?", "m-synthetic-a-synthetic-main-120"),
        Probe("old_alias", "What was the old alias for internship preparation?", "m-synthetic-a-synthetic-main-95"),
        Probe("paraphrase", "Which bird-named project had a spending limit?", "m-synthetic-a-synthetic-main-2"),
        Probe("recent", "What was the recent routine topic at turn 160?", "m-synthetic-a-synthetic-main-160"),
        Probe("other_user", "What is my Blue Heron review code?", "m-synthetic-b-synthetic-main-43", user_id="synthetic-b"),
        Probe("other_thread", "What is the Aurora report page count?", "m-synthetic-a-synthetic-other-44", thread_id="synthetic-other"),
    ]
    return turns, probes


def postgres_lexical(turns: list[Turn], probes: list[Probe]):
    settings = get_settings()
    results, latencies = {}, []
    with psycopg.connect(settings.database_url_sync, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE synthetic_turns (user_id text, thread_id text, message_id text, turn_no int, role text, occurred_at timestamptz, content text, supersedes text) ON COMMIT PRESERVE ROWS")
            cur.executemany("INSERT INTO synthetic_turns VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", [(t.user_id,t.thread_id,t.message_id,t.turn,t.role,t.timestamp,t.content,t.supersedes) for t in turns])
            cur.execute("CREATE INDEX synthetic_fts ON synthetic_turns USING GIN (to_tsvector('english', content))")
            for p in probes:
                start = time.perf_counter()
                terms = [term for term in re.findall(r"[a-z0-9]+", p.query.lower()) if term not in {"what","when","which","how","the","is","was","did","my","i","for","to","at","now","current","old","a","an"}]
                tsquery = " | ".join(terms)
                cur.execute("SELECT message_id FROM synthetic_turns WHERE user_id=%s AND thread_id=%s AND to_tsvector('english',content) @@ to_tsquery('english',%s) ORDER BY ts_rank_cd(to_tsvector('english',content),to_tsquery('english',%s)) DESC, turn_no DESC LIMIT 3", (p.user_id,p.thread_id,tsquery,tsquery))
                results[p.label] = [x[0] for x in cur.fetchall()]
                latencies.append((time.perf_counter() - start) * 1000)
            cur.execute("SELECT pg_total_relation_size('synthetic_turns'::regclass) + pg_total_relation_size('synthetic_fts'::regclass)")
            size = cur.fetchone()[0]
    return results, round(mean(latencies), 3), size


def embeddings(client: OpenAI, turns: list[Turn], probes: list[Probe]):
    all_inputs = [t.content for t in turns] + [p.query for p in probes]
    start = time.perf_counter()
    response = client.embeddings.create(model="text-embedding-3-small", input=all_inputs)
    elapsed = (time.perf_counter() - start) * 1000
    vectors = [x.embedding for x in sorted(response.data, key=lambda x: x.index)]
    results = {}
    for i, p in enumerate(probes):
        query = vectors[len(turns) + i]
        norm_q = math.sqrt(sum(x*x for x in query))
        scored = []
        for t, vector in zip(turns, vectors[:len(turns)]):
            if (t.user_id,t.thread_id) != (p.user_id,p.thread_id):
                continue
            score = sum(a*b for a,b in zip(query,vector)) / (norm_q * math.sqrt(sum(x*x for x in vector)))
            scored.append((score,t.message_id))
        results[p.label] = [identifier for _,identifier in sorted(scored,reverse=True)[:3]]
    usage = getattr(response, "usage", None)
    return results, round(elapsed, 3), getattr(usage, "total_tokens", None), len(vectors[0]) * len(vectors) * 4


def fuse(left: dict, right: dict, probes: list[Probe]):
    results = {}
    for p in probes:
        scores = {}
        for ranking in (left[p.label], right[p.label]):
            for rank, identifier in enumerate(ranking):
                scores[identifier] = scores.get(identifier, 0) + 1 / (60 + rank + 1)
        results[p.label] = [x for x,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]]
    return results


def metrics(rankings: dict, probes: list[Probe]):
    return {
        "recall_at_3": round(sum(p.gold in rankings.get(p.label, []) for p in probes) / len(probes), 3),
        "top_1_accuracy": round(sum(bool(rankings.get(p.label)) and rankings[p.label][0] == p.gold for p in probes) / len(probes), 3),
        "stale_top_1_count": sum(bool(rankings.get(p.label)) and rankings[p.label][0] == "m-synthetic-a-synthetic-main-62" for p in probes if p.label == "corrected_number"),
        "precision_at_3": round(sum(p.gold in rankings.get(p.label, []) for p in probes) / max(1,sum(len(rankings.get(p.label, [])) for p in probes)), 3),
        "hits": {p.label: p.gold in rankings.get(p.label, []) for p in probes},
        "rankings": rankings,
    }


def main():
    turns, probes = corpus()
    output = {"kind":"isolated_synthetic_probe", "turns":len(turns),"probes":len(probes), "methods":{}}
    recent = {p.label:[t.message_id for t in reversed(turns) if (t.user_id,t.thread_id)==(p.user_id,p.thread_id) and t.turn>=153][:3] for p in probes}
    output["methods"]["recent_only"] = metrics(recent,probes)
    # A deliberately bounded, hand-curated surrogate is a context-policy probe,
    # NOT a model-generated summary quality result. It omits some exact details.
    summary_ids = {
        "exact_code":"m-synthetic-a-synthetic-main-2",
        "correction":"m-synthetic-a-synthetic-main-35",
        "corrected_number":"m-synthetic-a-synthetic-main-120",
        "old_alias":"m-synthetic-a-synthetic-main-95",
    }
    summary = {p.label:([summary_ids[p.label]] if p.label in summary_ids else []) for p in probes}
    output["methods"]["hand_curated_summary_plus_recent"] = metrics({p.label:list(dict.fromkeys(summary[p.label]+recent[p.label]))[:3] for p in probes},probes)
    try:
        lexical, latency, size = postgres_lexical(turns,probes)
        output["methods"]["postgres_fts"] = {**metrics(lexical,probes),"mean_query_ms":latency,"temp_storage_bytes":size}
        output["methods"]["fts_plus_recent"] = metrics({p.label:list(dict.fromkeys(lexical[p.label]+recent[p.label]))[:3] for p in probes},probes)
        output["methods"]["summary_fts_recent"] = metrics({p.label:list(dict.fromkeys(summary[p.label]+lexical[p.label]+recent[p.label]))[:3] for p in probes},probes)
    except Exception as exc:
        output["methods"]["postgres_fts"] = {"unavailable":type(exc).__name__}
        lexical = None
    s=get_settings()
    try:
        client=OpenAI(api_key=s.openai_api_key,base_url=s.openai_base_url,timeout=20)
        semantic, latency, tokens, storage = embeddings(client,turns,probes)
        output["methods"]["semantic_embeddings"]={**metrics(semantic,probes),"batch_latency_ms":latency,"provider_embedding_tokens":tokens,"float32_storage_bytes":storage}
        if lexical is not None:
            output["methods"]["hybrid_rrf"] = metrics(fuse(lexical,semantic,probes),probes)
    except Exception as exc:
        output["methods"]["semantic_embeddings"]={"unavailable":type(exc).__name__}
        output["methods"]["hybrid_rrf"]={"unavailable":"semantic_embeddings_unavailable"}
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

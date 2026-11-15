"""OpenClaw-style bounded hybrid fusion over source-backed transcript hits.

Candidate generation remains in scoped lexical/semantic backends. This module
adapts stable-ID fusion, weighted component scores, CJK-aware MMR diversity,
score threshold and lexical fallback from OpenClaw memory-core. Correction and
privacy filters must run before fusion, never after answer generation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.agent.transcript import HistoryHit


@dataclass(frozen=True)
class FusedHistoryHit:
    hit: HistoryHit
    score: float
    vector_score: float
    text_score: float


def _cjk_tokens(text: str) -> set[str]:
    """Adapt OpenClaw's ASCII + adjacent CJK unigram/bigram tokenizer."""
    lowered = text.lower()
    ascii_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    cjk = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u1100-\u11ff]")
    indexed = [(index, char) for index, char in enumerate(lowered) if cjk.fullmatch(char)]
    result = ascii_tokens | {char for _, char in indexed}
    result.update(previous_char + char for (previous_index, previous_char), (index, char)
                  in zip(indexed, indexed[1:]) if index == previous_index + 1)
    return result


def _similarity(left: str, right: str) -> float:
    a, b = _cjk_tokens(left), _cjk_tokens(right)
    if not a and not b:
        return float(left.strip().casefold() == right.strip().casefold())
    return len(a & b) / len(a | b) if a | b else 0.0


def _mmr_order(items: list[FusedHistoryHit], *, lam: float = 0.7) -> list[FusedHistoryHit]:
    """OpenClaw MMR: normalized relevance minus max selected similarity."""
    if len(items) < 2:
        return items
    high = max(item.score for item in items)
    low = min(item.score for item in items)
    relevance = {item.hit.message_id: (1.0 if high == low else (item.score - low) / (high - low))
                 for item in items}
    remaining = list(items)
    selected = []
    while remaining:
        selected_text = [item.hit.content for item in selected]
        best = max(remaining, key=lambda item: (
            lam * relevance[item.hit.message_id] - (1 - lam) *
            max((_similarity(item.hit.content, content) for content in selected_text), default=0),
            item.score, item.hit.message_id,
        ))
        selected.append(best)
        remaining.remove(best)
    return selected


def _lexical_confidence(raw: float | None) -> float:
    """Map PostgreSQL ts_rank_cd (higher is better) to [0,1].

    This is a provider-specific adaptation, not an OpenClaw constant. The
    0.1 scale requires matched retrieval/final-answer validation before a
    production default can be approved.
    """
    score = max(0.0, float(raw or 0))
    return score / (score + 0.1) if score else 0.0


def _vector_confidence(raw: float | None) -> float:
    return max(0.0, min(1.0, float(raw or 0)))


def fuse_history_hits(
    lexical: list[HistoryHit], semantic: list[HistoryHit], *,
    user_id: str, thread_id: str, include_stale: bool = False,
    max_results: int = 6, min_score: float = 0.35,
    vector_weight: float = 0.7, text_weight: float = 0.3,
    mmr_lambda: float = 0.7,
) -> list[FusedHistoryHit]:
    """Merge scoped candidates by message ID; preserve provenance and fallback.

    Explicit user/thread validation defends even against a buggy backend. This
    does not silently treat a failed semantic provider as an empty result;
    callers must distinguish an unavailable backend before invoking fusion.
    """
    if not user_id or not thread_id or not 1 <= max_results <= 20:
        raise ValueError("Bounded user/thread scope is required")
    if not 0 <= min_score <= 1 or not 0 <= mmr_lambda <= 1:
        raise ValueError("Invalid ranking configuration")
    if not math.isclose(vector_weight + text_weight, 1, abs_tol=1e-9):
        raise ValueError("Hybrid weights must total one")
    candidates: dict[str, dict] = {}
    keyword_ids = set()
    for source, hits in (("semantic", semantic), ("lexical", lexical)):
        for hit in hits:
            if (hit.user_id, hit.thread_id) != (user_id, thread_id):
                raise PermissionError("Retrieval backend returned another user's/thread's evidence")
            if hit.stale and not include_stale:
                continue
            entry = candidates.setdefault(hit.message_id, {"hit": hit, "vector": 0.0, "text": 0.0})
            if source == "semantic":
                entry["vector"] = _vector_confidence(hit.score)
            else:
                entry["hit"] = hit  # prefer canonical lexical projection metadata
                entry["text"] = _lexical_confidence(hit.score)
                keyword_ids.add(hit.message_id)
    merged = [FusedHistoryHit(
        hit=entry["hit"],
        score=vector_weight * entry["vector"] + text_weight * entry["text"],
        vector_score=entry["vector"], text_score=entry["text"],
    ) for entry in candidates.values()]
    ranked = _mmr_order(sorted(merged, key=lambda item: (-item.score, item.hit.message_id)),
                        lam=mmr_lambda)
    strict = [item for item in ranked if item.score >= min_score][:max_results]
    if len(strict) == max_results or not keyword_ids:
        return strict
    seen = {item.hit.message_id for item in strict}
    fallback = [item for item in ranked if item.hit.message_id in keyword_ids
                and item.hit.message_id not in seen]
    return (strict + fallback)[:max_results]

"""OpenClaw-style fusion safety and CJK diversity tests."""

from datetime import datetime, timezone

import pytest

from app.agent.memory_retrieval import _cjk_tokens, fuse_history_hits
from app.agent.transcript import HistoryHit


def hit(identifier: str, content: str, score: float, *,
        user: str = "u", thread: str = "t", stale: bool = False,
        method: str = "lexical") -> HistoryHit:
    return HistoryHit(user, thread, identifier, identifier, "user",
                      datetime.now(timezone.utc), content,
                      f"checkpoint:{thread}:{identifier}", None, stale, method, score)


def test_fusion_deduplicates_and_preserves_source_provenance():
    lexical = [hit("m1", "蓝鲸项目编号 CN-8042", 0.8)]
    semantic = [hit("m1", "蓝鲸项目编号 CN-8042", 0.9, method="semantic"),
                hit("m2", "another project", 0.7, method="semantic")]
    results = fuse_history_hits(lexical, semantic, user_id="u", thread_id="t")
    assert [item.hit.message_id for item in results].count("m1") == 1
    assert results[0].hit.source_ref == "checkpoint:t:m1"
    assert results[0].vector_score > 0 and results[0].text_score > 0


def test_stale_and_cross_scope_results_cannot_enter_context():
    old = hit("old", "47 pages", 2, stale=True)
    current = hit("current", "52 pages", 1)
    assert [item.hit.message_id for item in fuse_history_hits(
        [old, current], [], user_id="u", thread_id="t")] == ["current"]
    with pytest.raises(PermissionError):
        fuse_history_hits([hit("foreign", "private", 1, user="another")], [],
                          user_id="u", thread_id="t")


def test_low_weight_keyword_result_uses_explicit_lexical_fallback():
    exact = hit("code", "AX-4731", 0.1)
    results = fuse_history_hits([exact], [], user_id="u", thread_id="t",
                                min_score=0.9)
    assert [item.hit.message_id for item in results] == ["code"]


def test_cjk_tokenizer_keeps_adjacent_bigrams_without_crossing_ascii():
    tokens = _cjk_tokens("我喜欢hello你好")
    assert {"我", "喜欢", "你好", "hello"}.issubset(tokens)
    assert "欢你" not in tokens

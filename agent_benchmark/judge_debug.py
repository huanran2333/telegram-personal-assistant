"""Explicit local Judge evidence capture and deterministic offline replay."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .judge_evidence import _SECRET_TEXT, _UUID
from .unified_judge import (PROMPT_VERSION, allowed_evidence_refs,
                            combined_verdict, judge_unified)

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_SECRET_FIELD = re.compile(r"(?i)(secret|password|cookie|authorization|credential|api[_-]?key|access[_-]?token|refresh[_-]?token|^token$)")


def redact(value):
    if isinstance(value, str):
        return _PHONE.sub("[PHONE]", _EMAIL.sub("[EMAIL]", _UUID.sub(
            "[ID]", _SECRET_TEXT.sub("[REDACTED]", value))))
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {str(key): "[REDACTED]" if _SECRET_FIELD.search(str(key))
            else redact(item) for key, item in value.items()}
    return value


def initial_record(packet: dict, messages: list[dict], model: str,
                   request_parameters: dict, raw: str | None, diagnostic: dict | None) -> dict:
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else None
    except (ValueError, TypeError):
        parsed = None
    return {"case_id": packet["case_id"], "dataset_version": packet["dataset_version"],
            "trace_sha256": packet.get("source_trace_sha256"),
            "prompt_version": PROMPT_VERSION, "model": model,
            "request_parameters": request_parameters,
            "messages_sent_redacted": redact(messages),
            "evidence_packet_redacted": redact(packet),
            "allowed_evidence_refs": allowed_evidence_refs(packet),
            "http_diagnostic": diagnostic,
            "raw_message_content_redacted": redact(raw) if raw is not None else None,
            "parsed_output_redacted": redact(parsed)}


def finalize_record(record: dict, oracle_outcome: str, judgment: dict) -> dict:
    parsed = record.get("parsed_output_redacted")
    checks = {}
    for name in ("task_completion", "fault_handling"):
        original = parsed.get(name) if isinstance(parsed, dict) else None
        validated = judgment[name]
        checks[name] = {
            "raw_verdict": original.get("verdict") if isinstance(original, dict) else None,
            "raw_field_types": ({k: type(v).__name__ for k, v in original.items()}
                                if isinstance(original, dict) else type(original).__name__),
            "raw_evidence_refs": original.get("evidence_refs") if isinstance(original, dict) else None,
            "allowed_evidence_refs": record["allowed_evidence_refs"],
            "validation_errors": validated.get("validation_errors", []),
            "validated_verdict": validated["verdict"],
        }
    record["validation"] = checks
    record["fusion"] = {"oracle_outcome": oracle_outcome,
                        "judge_raw_task_verdict": checks["task_completion"]["raw_verdict"],
                        "judge_validated_task_verdict": judgment["task_completion"]["verdict"],
                        "combined_verdict": combined_verdict(oracle_outcome, judgment),
                        "rule": "oracle_failure_wins; otherwise validated task verdict"}
    return record


def save_record(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def replay(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    packet = record["evidence_packet_redacted"]
    raw = record["raw_message_content_redacted"]
    if raw is None:
        raise ValueError("NoModelOutputToReplay")
    record["source_prompt_version"] = record.get("prompt_version")
    record["prompt_version"] = PROMPT_VERSION
    record["allowed_evidence_refs"] = allowed_evidence_refs(packet)
    judgment = judge_unified(packet, lambda _: raw, judge_model=record["model"])
    return finalize_record(record, record["fusion"]["oracle_outcome"], judgment)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline replay of a saved unified Judge response")
    parser.add_argument("record", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("OutputAlreadyExists")
    save_record(args.output, replay(args.record))
    print(json.dumps({"output": str(args.output), "network_calls": 0}))


if __name__ == "__main__":
    main()

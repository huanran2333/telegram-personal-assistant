"""Re-score recorded routing attempts without repeating model calls.

Only deterministic rubric changes are eligible. The original run is retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.quality_suite import DATASET, load_dataset, summarize


def regrade_route(report: dict, dataset: Path = DATASET) -> dict:
    corpus, digest = load_dataset(dataset)
    if report.get("manifest", {}).get("dataset_sha256") != digest:
        raise ValueError("Recorded dataset does not match the current rubric")
    cases = {case["id"]: case for case in corpus["cases"]}
    rows = []
    for original in report["rows"]:
        case_id = original["case_id"]
        if case_id not in cases:
            raise ValueError("Recorded case is absent from the dataset")
        row = dict(original)
        row["expected_intent"] = cases[case_id]["expected_intent"]
        rows.append(row)
    result = {"manifest": dict(report["manifest"]), "summary": summarize(rows), "rows": rows}
    result["manifest"]["cached_regrade"] = "expected clarification/safe block excluded from runtime failures"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = regrade_route(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

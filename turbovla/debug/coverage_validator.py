from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def load_manifest_records(trace_dir: Path | str) -> list[dict]:
    path = Path(trace_dir) / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_coverage(records_or_trace: Iterable[dict] | Path | str, expected: set[str]) -> dict:
    records = load_manifest_records(records_or_trace) if isinstance(records_or_trace, (Path, str)) else list(records_or_trace)
    actual_names = {record["semantic_name"] for record in records}
    keys = [(record["semantic_name"], int(record.get("call_index", 0))) for record in records]
    duplicates = [f"{name}::call_{call:02d}" for (name, call), count in Counter(keys).items() if count > 1]
    missing = sorted(expected - actual_names)
    unexpected = sorted(actual_names - expected)
    return {
        "expected_count": len(expected), "actual_count": len(actual_names), "record_count": len(records),
        "missing": missing, "unexpected": unexpected, "duplicate_semantic_keys": sorted(duplicates),
        "status": "PASS" if not missing and not unexpected and not duplicates else "FAIL",
    }


def write_coverage_report(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

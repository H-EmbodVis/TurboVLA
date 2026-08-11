from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .fixture_reader import FixtureReader


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fixture(
    fixture_dir: Path | str,
    *,
    requested_fixture_id: str | None = None,
    expected_model: dict | None = None,
    reproduce_preprocessing: Callable[[FixtureReader], dict[str, np.ndarray]] | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict:
    reader = FixtureReader(fixture_dir)
    errors: list[str] = []
    metadata = reader.metadata
    if requested_fixture_id and metadata.get("fixture_id") != requested_fixture_id:
        errors.append(f"fixture_id mismatch: {metadata.get('fixture_id')!r} != {requested_fixture_id!r}")
    if expected_model:
        actual_model = metadata.get("model", {})
        for key, value in expected_model.items():
            if actual_model.get(key) != value:
                errors.append(f"model metadata mismatch for {key}: {actual_model.get(key)!r} != {value!r}")

    listed = {item["path"]: item for item in metadata.get("files", [])}
    if len(listed) != len(metadata.get("files", [])):
        errors.append("duplicate file paths in fixture metadata")
    for relative, item in listed.items():
        path = reader.path(relative)
        if not path.is_file():
            errors.append(f"missing file: {relative}")
            continue
        if path.stat().st_size != int(item["byte_length"]):
            errors.append(f"byte length mismatch: {relative}")
        if _sha(path) != item["sha256"]:
            errors.append(f"sha256 mismatch: {relative}")
        if item.get("kind") == "npy":
            try:
                array = np.load(path, allow_pickle=False)
                if list(array.shape) != item.get("shape"):
                    errors.append(f"shape mismatch: {relative}")
                declared = item.get("dtype")
                if declared != "bfloat16_bits_u16" and str(array.dtype) != declared:
                    errors.append(f"dtype mismatch: {relative}: {array.dtype} != {declared}")
            except Exception as error:
                errors.append(f"invalid npy {relative}: {error}")

    actual = {
        path.relative_to(reader.fixture_dir).as_posix()
        for path in reader.fixture_dir.rglob("*") if path.is_file() and path.name != "metadata.json"
    }
    expected = set(listed)
    errors.extend(f"stale/unlisted file: {item}" for item in sorted(actual - expected))
    errors.extend(f"listed file not present: {item}" for item in sorted(expected - actual))

    reproduction = {}
    if reproduce_preprocessing:
        try:
            reproduced = reproduce_preprocessing(reader)
            for relative, candidate in reproduced.items():
                reference = reader.load_numpy(relative)
                if np.issubdtype(reference.dtype, np.bool_) or not np.issubdtype(reference.dtype, np.inexact):
                    passed = bool(np.array_equal(candidate, reference))
                    max_abs = float(np.max(np.abs(candidate.astype(np.int64) - reference.astype(np.int64))))
                else:
                    passed = bool(np.allclose(candidate, reference, rtol=rtol, atol=atol, equal_nan=False))
                    max_abs = float(np.max(np.abs(candidate - reference)))
                reproduction[relative] = {"pass": passed, "max_abs": max_abs}
                if not passed:
                    errors.append(f"reproduction mismatch: {relative}")
        except Exception as error:
            errors.append(f"preprocessing reproduction failed: {type(error).__name__}: {error}")

    return {"fixture_id": metadata.get("fixture_id"), "status": "PASS" if not errors else "FAIL",
            "file_count": len(listed), "errors": errors, "reproduction": reproduction}


def write_fixture_validation(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

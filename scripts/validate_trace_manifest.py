from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SIZES = {"float32": 4, "float16": 2, "bfloat16": 2, "int64": 8, "int32": 4,
         "int16": 2, "int8": 1, "uint8": 1, "bool": 1}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(trace_dir: Path | str, *, strict: bool = True) -> dict:
    trace_dir = Path(trace_dir)
    path = trace_dir / "manifest.jsonl"
    errors: list[str] = []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys: set[tuple[str, int]] = set()
    previous = -1
    actual_files = {item.relative_to(trace_dir).as_posix() for root in ("tensors", "tensors_native")
                    if (trace_dir / root).exists() for item in (trace_dir / root).rglob("*") if item.is_file()}
    listed_files: set[str] = set()
    for record in records:
        trace_id = int(record.get("trace_id", -1))
        key = (record.get("semantic_name"), int(record.get("call_index", 0)))
        if key in keys:
            errors.append(f"duplicate semantic key: {key}")
        keys.add(key)
        if trace_id <= previous:
            errors.append(f"non-monotonic trace_id {trace_id} after {previous}")
        previous = trace_id
        shape = record.get("shape", [])
        numel = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if numel != int(record.get("numel", -1)):
            errors.append(f"numel mismatch for {key}")
        for file_field, dtype_field, hash_field in (("file", "storage_dtype", "sha256_f32"),
                                                     ("native_file", "native_storage_dtype", "sha256_native")):
            relative = record.get(file_field)
            if not relative:
                continue
            listed_files.add(relative)
            tensor_path = trace_dir / relative
            if not tensor_path.is_file():
                errors.append(f"missing {file_field}: {relative}")
                continue
            dtype = record.get(dtype_field, "").removeprefix("torch.")
            if dtype not in SIZES:
                errors.append(f"unknown {dtype_field}={dtype!r} for {key}")
            elif tensor_path.stat().st_size != numel * SIZES[dtype]:
                errors.append(f"byte length mismatch for {relative}")
            expected_hash = record.get(hash_field)
            # sha256_f32 is intentionally absent for integer canonical files.
            if expected_hash and _sha(tensor_path) != expected_hash:
                errors.append(f"sha256 mismatch for {relative}")
        expected_infinity = str(record.get("semantic_name", "")).endswith(("masked_logits", "softmax.shifted"))
        if strict and (int(record.get("nan_count", 0)) or (int(record.get("inf_count", 0)) and not expected_infinity)):
            errors.append(f"non-finite tensor: {key}")
    stale = sorted(actual_files - listed_files)
    errors.extend(f"stale tensor file: {path}" for path in stale)
    return {"status": "PASS" if not errors else "FAIL", "tensor_count": len(records), "errors": errors,
            "total_tensor_bytes": sum((trace_dir / path).stat().st_size for path in listed_files if (trace_dir / path).is_file())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_manifest(args.trace_dir, strict=args.strict)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

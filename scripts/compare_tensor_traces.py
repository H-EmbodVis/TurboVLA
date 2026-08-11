from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DTYPES = {
    "float32": "<f4", "f32": "<f4", "float16": "<f2", "f16": "<f2",
    "int64": "<i8", "i64": "<i8", "int32": "<i4", "i32": "<i4",
    "int16": "<i2", "int8": "i1", "uint8": "u1", "bool": "u1",
}


def load_manifest(trace_dir: Path) -> list[dict]:
    path = trace_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous = -1
    for record in records:
        trace_id = int(record["trace_id"])
        if trace_id <= previous:
            raise ValueError(f"non-monotonic reference trace_id: {trace_id} after {previous}")
        previous = trace_id
    return records


def semantic_key(record: dict) -> tuple[str, int]:
    return record["semantic_name"], int(record.get("call_index", 0))


def _read(trace_dir: Path, record: dict) -> np.ndarray:
    storage = record.get("storage_dtype", "float32").lower().removeprefix("torch.")
    if storage == "bfloat16":
        # Legacy manifests may point at BF16 bits instead of canonical F32.
        bits = np.fromfile(trace_dir / record["file"], dtype="<u2")
        return (bits.astype(np.uint32) << 16).view(np.float32)
    if storage not in DTYPES:
        raise TypeError(f"unsupported storage dtype {storage!r}")
    return np.fromfile(trace_dir / record["file"], dtype=DTYPES[storage])


def _safe_metrics(reference: np.ndarray, candidate: np.ndarray, exact: bool) -> dict:
    if exact:
        return {
            "exact_match": True, "max_abs_error": 0.0, "mean_abs_error": 0.0,
            "median_abs_error": 0.0, "rmse": 0.0, "max_relative_error": 0.0,
            "mean_relative_error": 0.0, "cosine_similarity": 1.0, "pearson_correlation": 1.0,
        }
    ref, cand = reference.astype(np.float64), candidate.astype(np.float64)
    difference = np.abs(ref - cand)
    denominator = np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
    relative = difference / denominator
    ref_centered, cand_centered = ref - ref.mean(), cand - cand.mean()
    cosine_denominator = np.linalg.norm(ref) * np.linalg.norm(cand)
    pearson_denominator = np.linalg.norm(ref_centered) * np.linalg.norm(cand_centered)
    def finite(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "exact_match": exact,
        "max_abs_error": finite(difference.max(initial=0.0)),
        "mean_abs_error": finite(difference.mean()) if difference.size else 0.0,
        "median_abs_error": finite(np.median(difference)) if difference.size else 0.0,
        "rmse": finite(np.sqrt(np.mean(difference ** 2))) if difference.size else 0.0,
        "max_relative_error": finite(relative.max(initial=0.0)),
        "mean_relative_error": finite(relative.mean()) if relative.size else 0.0,
        "cosine_similarity": finite(np.dot(ref, cand) / cosine_denominator) if cosine_denominator else (1.0 if exact else 0.0),
        "pearson_correlation": finite(np.dot(ref_centered, cand_centered) / pearson_denominator) if pearson_denominator else (1.0 if exact else 0.0),
    }


def compare_traces(reference_dir: Path, candidate_dir: Path, output_dir: Path, *, atol: float = 0.0,
                   rtol: float = 0.0, stop_after_first_failure: bool = False) -> dict:
    reference_records, candidate_records = load_manifest(reference_dir), load_manifest(candidate_dir)
    candidate_by_key = {semantic_key(record): record for record in candidate_records}
    reference_keys = {semantic_key(record) for record in reference_records}
    results: list[dict] = []
    first_failure = None
    previous_passing = None
    output_dir.mkdir(parents=True, exist_ok=True)
    diff_dir = output_dir / "diff_tensors"

    for reference in reference_records:  # Reference execution order is authoritative.
        key = semantic_key(reference)
        name, call_index = key
        candidate = candidate_by_key.get(key)
        result = {"semantic_name": name, "call_index": call_index, "reference_trace_id": reference["trace_id"]}
        if candidate is None:
            result["status"] = "MISSING_CANDIDATE"
        else:
            result["candidate_trace_id"] = candidate["trace_id"]
            result["shape_match"] = reference.get("shape") == candidate.get("shape")
            result["layout_match"] = reference.get("layout", "") == candidate.get("layout", "")
            result.update(
                ref_nan_count=reference.get("nan_count", 0), ref_inf_count=reference.get("inf_count", 0),
                cand_nan_count=candidate.get("nan_count", 0), cand_inf_count=candidate.get("inf_count", 0),
            )
            if not result["shape_match"]:
                result["status"] = "FAIL_SHAPE"
            elif not result["layout_match"]:
                result["status"] = "FAIL_LAYOUT"
            else:
                ref_data, cand_data = _read(reference_dir, reference), _read(candidate_dir, candidate)
                if ref_data.size != cand_data.size:
                    result["status"] = "FAIL_SHAPE"
                else:
                    exact = bool(np.array_equal(ref_data, cand_data))
                    result.update(_safe_metrics(ref_data, cand_data, exact))
                    finite = bool(np.isfinite(ref_data).all() and np.isfinite(cand_data).all())
                    close = bool(np.allclose(cand_data, ref_data, rtol=rtol, atol=atol, equal_nan=False))
                    result["status"] = "PASS_EXACT" if exact else ("PASS_TOLERANCE" if finite and close else ("FAIL_NAN" if not finite else "FAIL_TOLERANCE"))
                    if not exact and not close:
                        bad = np.flatnonzero(~np.isclose(cand_data, ref_data, rtol=rtol, atol=atol, equal_nan=False))
                        if bad.size:
                            flat_index = int(bad[0])
                            result.update(
                                first_bad_flat_index=flat_index,
                                first_bad_coordinate=[int(item) for item in np.unravel_index(flat_index, reference.get("shape", []))] if reference.get("shape") else [],
                                reference_value=float(ref_data[flat_index]) if np.isfinite(ref_data[flat_index]) else None,
                                candidate_value=float(cand_data[flat_index]) if np.isfinite(cand_data[flat_index]) else None,
                            )
                        diff_dir.mkdir(exist_ok=True)
                        safe = name.replace("/", "_") + f"__call_{call_index:02d}"
                        (cand_data.astype(np.float64) - ref_data.astype(np.float64)).astype("<f4").tofile(diff_dir / f"{safe}.diff.f32le.bin")
        results.append(result)
        if result["status"].startswith("PASS"):
            previous_passing = f"{name}::call_{call_index:02d}"
        elif first_failure is None:
            first_failure = dict(result)
            first_failure["previous_passing_tensor"] = previous_passing
            if stop_after_first_failure:
                break

    if not stop_after_first_failure:
        for candidate in candidate_records:
            key = semantic_key(candidate)
            if key not in reference_keys:
                results.append({"semantic_name": key[0], "call_index": key[1],
                                "candidate_trace_id": candidate["trace_id"], "status": "MISSING_REFERENCE"})
                if first_failure is None:
                    first_failure = dict(results[-1])
                    first_failure["previous_passing_tensor"] = previous_passing

    statuses: dict[str, int] = {}
    for result in results:
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
    fieldnames = sorted({key for result in results for key in result})
    with (output_dir / "compare.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "compare.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "first_divergence.txt").write_text(
        "PASS_EXACT\n" if first_failure is None else json.dumps(first_failure, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "shape_mismatches.txt").write_text("\n".join(r["semantic_name"] for r in results if r["status"] == "FAIL_SHAPE"), encoding="utf-8")
    (output_dir / "layout_mismatches.txt").write_text("\n".join(r["semantic_name"] for r in results if r["status"] == "FAIL_LAYOUT"), encoding="utf-8")
    (output_dir / "missing_in_cpp.txt").write_text("\n".join(r["semantic_name"] for r in results if r["status"] == "MISSING_CANDIDATE"), encoding="utf-8")
    (output_dir / "missing_in_python.txt").write_text("\n".join(r["semantic_name"] for r in results if r["status"] == "MISSING_REFERENCE"), encoding="utf-8")
    summary = {"status": "PASS" if first_failure is None else "FAIL", "status_counts": statuses,
               "tensor_count": len(results), "first_divergence": first_failure}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare semantic tensor traces in reference execution order")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--stop-after-first-failure", action="store_true")
    args = parser.parse_args()
    summary = compare_traces(args.reference, args.candidate, args.output, atol=args.atol, rtol=args.rtol,
                             stop_after_first_failure=args.stop_after_first_failure)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

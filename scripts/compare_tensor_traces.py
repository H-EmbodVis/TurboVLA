from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np

def load_manifest(trace_dir: Path) -> dict:
    manifest_path = trace_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return {}
    
    tensors = {}
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                name = t.get("semantic_name")
                call_index = t.get("call_index", 0)
                if name:
                    tensors[f"{name}_{call_index}"] = t
    return tensors

def get_dtype(dtype_str: str) -> np.dtype:
    mapping = {
        "float32": np.float32, "f32": np.float32,
        "float16": np.float16, "f16": np.float16,
        "bfloat16": np.uint16, "torch.bfloat16": np.uint16,
        "uint16": np.uint16,
        "int32": np.int32, "i32": np.int32,
        "int64": np.int64, "i64": np.int64,
        "bool": np.bool_
    }
    return mapping.get(dtype_str.lower(), np.float32)

def main():
    parser = argparse.ArgumentParser(description="Compare tensor traces")
    parser.add_argument("--reference", type=Path, required=True, help="Reference trace dir")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate trace dir")
    parser.add_argument("--output", type=Path, required=True, help="Output dir")
    parser.add_argument("--atol", type=float, default=0.02, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.02, help="Relative tolerance")
    parser.add_argument("--stop-after-first-failure", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False)
    args = parser.parse_args()

    ref_tensors = load_manifest(args.reference)
    cand_tensors = load_manifest(args.candidate)

    args.output.mkdir(parents=True, exist_ok=True)
    diff_dir = args.output / "diff_tensors"
    diff_dir.mkdir(parents=True, exist_ok=True)

    results = []
    
    missing_in_cand = set(ref_tensors.keys()) - set(cand_tensors.keys())
    missing_in_ref = set(cand_tensors.keys()) - set(ref_tensors.keys())
    
    with open(args.output / "missing_in_python.txt", "w") as f:
        for m in sorted(missing_in_ref):
            f.write(f"{m}\n")
            results.append({"semantic_name": m, "status": "MISSING_REFERENCE"})
            
    with open(args.output / "missing_in_cpp.txt", "w") as f:
        for m in sorted(missing_in_cand):
            f.write(f"{m}\n")
            results.append({"semantic_name": m, "status": "MISSING_CANDIDATE"})

    shape_mismatches = []
    layout_mismatches = []
    first_failure = None
    first_failure_prev = None
    prev_passing_name = None

    common_keys = sorted(set(ref_tensors.keys()) & set(cand_tensors.keys()))
    
    for key in common_keys:
        ref_t = ref_tensors[key]
        cand_t = cand_tensors[key]
        
        name = ref_t.get("semantic_name")
        
        ref_shape = ref_t.get("shape", [])
        cand_shape = cand_t.get("shape", [])
        ref_layout = ref_t.get("layout", "")
        cand_layout = cand_t.get("layout", "")
        
        shape_match = (ref_shape == cand_shape)
        layout_match = (ref_layout == cand_layout)
        numel_match = (np.prod(ref_shape) == np.prod(cand_shape))
        
        if not shape_match:
            shape_mismatches.append(name)
        if not layout_match:
            layout_mismatches.append(name)
            
        res = {
            "semantic_name": name,
            "shape_match": shape_match,
            "layout_match": layout_match,
            "numel_match": numel_match,
            "ref_nan_count": ref_t.get("nan_count", 0),
            "ref_inf_count": ref_t.get("inf_count", 0),
            "cand_nan_count": cand_t.get("nan_count", 0),
            "cand_inf_count": cand_t.get("inf_count", 0),
        }

        if not numel_match:
            res["status"] = "FAIL_SHAPE"
            results.append(res)
            continue
            
        ref_file = args.reference / ref_t.get("file", "")
        cand_file = args.candidate / cand_t.get("file", "")
        
        if not ref_file.exists() or not cand_file.exists():
            res["status"] = "MISSING_FILE"
            results.append(res)
            continue
            
        ref_data = np.fromfile(ref_file, dtype=get_dtype(ref_t.get("storage_dtype", "f32"))).astype(np.float32)
        cand_data = np.fromfile(cand_file, dtype=get_dtype(cand_t.get("storage_dtype", "f32"))).astype(np.float32)
        
        if ref_data.size != cand_data.size:
            res["status"] = "FAIL_SHAPE"
            results.append(res)
            continue
            
        exact_match = bool(np.array_equal(ref_data, cand_data))
        res["exact_match"] = exact_match
        
        abs_diff = np.abs(ref_data - cand_data)
        max_abs_error = float(np.max(abs_diff))
        mean_abs_error = float(np.mean(abs_diff))
        median_abs_error = float(np.median(abs_diff))
        rmse = float(np.sqrt(np.mean(abs_diff**2)))
        
        epsilon = 1e-8
        rel_diff = abs_diff / (np.abs(ref_data) + epsilon)
        max_relative_error = float(np.max(rel_diff))
        mean_relative_error = float(np.mean(rel_diff))
        
        try:
            if np.std(ref_data) > 0 and np.std(cand_data) > 0:
                pearson_correlation = float(np.corrcoef(ref_data.flatten(), cand_data.flatten())[0, 1])
                cosine_similarity = float(np.dot(ref_data.flatten(), cand_data.flatten()) / (np.linalg.norm(ref_data) * np.linalg.norm(cand_data)))
            else:
                pearson_correlation = 1.0 if exact_match else 0.0
                cosine_similarity = 1.0 if exact_match else 0.0
        except:
            pearson_correlation = 0.0
            cosine_similarity = 0.0
            
        res.update({
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "median_abs_error": median_abs_error,
            "rmse": rmse,
            "max_relative_error": max_relative_error,
            "mean_relative_error": mean_relative_error,
            "cosine_similarity": cosine_similarity,
            "pearson_correlation": pearson_correlation,
        })
        
        is_nan = np.isnan(cand_data).any() or np.isinf(cand_data).any()
        
        status = "PASS_EXACT"
        if not exact_match:
            if max_abs_error <= args.atol and max_relative_error <= args.rtol:
                status = "PASS_TOLERANCE"
            else:
                status = "FAIL_TOLERANCE"
                
        if is_nan:
            status = "FAIL_NAN"
            
        if not shape_match:
            status = "FAIL_SHAPE"
        elif not layout_match and status not in ["FAIL_NAN", "FAIL_TOLERANCE"]:
            status = "FAIL_LAYOUT"
            
        res["status"] = status
        
        if status in ["FAIL_TOLERANCE", "FAIL_NAN"]:
            diff_indices = np.where(~np.isclose(ref_data, cand_data, rtol=args.rtol, atol=args.atol))[0]
            if len(diff_indices) > 0:
                first_bad_flat_index = int(diff_indices[0])
                res["first_bad_flat_index"] = first_bad_flat_index
                if len(ref_shape) > 0:
                    res["first_bad_coordinate"] = str(np.unravel_index(first_bad_flat_index, ref_shape))
                else:
                    res["first_bad_coordinate"] = "()"
                res["reference_value"] = float(ref_data[first_bad_flat_index])
                res["candidate_value"] = float(cand_data[first_bad_flat_index])
                
            safe_name = name.replace("/", "_")
            (cand_data - ref_data).astype(np.float32).tofile(diff_dir / f"{safe_name}.diff.f32le.bin")
            abs_diff.astype(np.float32).tofile(diff_dir / f"{safe_name}.abs_diff.f32le.bin")
            rel_diff.astype(np.float32).tofile(diff_dir / f"{safe_name}.relative_diff.f32le.bin")
            
            if first_failure is None:
                first_failure = res
                first_failure_prev = prev_passing_name
                if args.stop_after_first_failure:
                    results.append(res)
                    break
        else:
            prev_passing_name = name
            
        results.append(res)

    with open(args.output / "shape_mismatches.txt", "w") as f:
        for s in shape_mismatches:
            f.write(f"{s}\n")
            
    with open(args.output / "layout_mismatches.txt", "w") as f:
        for s in layout_mismatches:
            f.write(f"{s}\n")
            
    def json_serializer(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    if first_failure:
        with open(args.output / "first_divergence.txt", "w") as f:
            f.write("First Failing Tensor:\n")
            f.write(json.dumps(first_failure, indent=2, default=json_serializer))
            f.write(f"\nPrevious Passing Tensor: {first_failure_prev}\n")

    valid_results = [r for r in results if "max_abs_error" in r]
    worst = sorted(valid_results, key=lambda x: x.get("max_abs_error", 0), reverse=True)[:10]
    with open(args.output / "worst_tensors.txt", "w") as f:
        for w in worst:
            f.write(f"{w['semantic_name']}: {w['max_abs_error']}\n")

    with open(args.output / "compare.json", "w") as f:
        json.dump(results, f, indent=2, default=json_serializer)

    if len(results) > 0:
        keys = set()
        for r in results:
            keys.update(r.keys())
        
        with open(args.output / "compare.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(list(keys)))
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print("=== Comparison Summary ===")
    status_counts = {}
    for r in results:
        s = r.get("status")
        status_counts[s] = status_counts.get(s, 0) + 1
        
    for s, c in status_counts.items():
        print(f"{s}: {c}")

if __name__ == '__main__':
    exit(main())

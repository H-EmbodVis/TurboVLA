from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def get_dtype_size(dtype_str: str) -> int:
    mapping = {
        "float32": 4, "f32": 4,
        "float16": 2, "f16": 2,
        "bfloat16": 2, "bf16": 2, "torch.bfloat16": 2,
        "uint16": 2,
        "int32": 4, "i32": 4,
        "int64": 8, "i64": 8,
        "bool": 1
    }
    return mapping.get(dtype_str.lower(), 4)

def calculate_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Validate trace manifest")
    parser.add_argument("trace_dir", type=Path, help="Directory containing manifest.jsonl")
    parser.add_argument("--strict", action="store_true", help="Fail on NaN/Inf")
    args = parser.parse_args()

    manifest_path = args.trace_dir / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} does not exist.")
        return 1

    tensors = []
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                tensors.append(json.loads(line))

    errors = []
    warnings = []
    seen_names = set()
    last_trace_id = -1

    for i, t in enumerate(tensors):
        name = t.get("semantic_name")
        call_index = t.get("call_index", 0)
        trace_id = t.get("trace_id", -1)
        name_key = f"{name}_{call_index}"
        
        if name_key in seen_names:
            errors.append(f"[{trace_id}] Duplicate semantic_name + call_index: {name_key}")
        seen_names.add(name_key)
        
        if trace_id <= last_trace_id:
            errors.append(f"[{trace_id}] Non-monotonic trace_id. Previous was {last_trace_id}")
        last_trace_id = trace_id
        
        shape = t.get("shape", [])
        layout = t.get("layout", "")
        
        if layout:
            layout_dims = len(layout.split(",")) if "," in layout else len(layout)
            if layout_dims != len(shape):
                errors.append(f"[{trace_id}] Layout '{layout}' (rank {layout_dims}) doesn't match shape dims {len(shape)} for {name}")
            
        numel = np.prod(shape) if shape else 1
        expected_numel = t.get("numel", numel)
        if expected_numel != numel:
            errors.append(f"[{trace_id}] Shape product {numel} != expected numel {expected_numel} for {name}")
            
        file_path = args.trace_dir / t.get("file", "")
        if not file_path.exists():
            errors.append(f"[{trace_id}] Missing file: {file_path}")
        else:
            dtype = t.get("storage_dtype", "float32")
            dtype_size = get_dtype_size(dtype)
            actual_size = file_path.stat().st_size
            expected_size = numel * dtype_size
            if actual_size != expected_size:
                errors.append(f"[{trace_id}] Size mismatch for {name}: expected {expected_size}, got {actual_size}")
            
            sha256 = t.get("sha256_f32")
            if sha256 and dtype in ["float32", "f32"]:
                actual_sha = calculate_sha256(file_path)
                if actual_sha != sha256:
                    errors.append(f"[{trace_id}] SHA256 mismatch for {name}")
                    
        nan_count = t.get("nan_count", 0)
        inf_count = t.get("inf_count", 0)
        if nan_count > 0 or inf_count > 0:
            msg = f"[{trace_id}] Found {nan_count} NaNs and {inf_count} Infs in {name}"
            if args.strict:
                errors.append(msg)
            else:
                warnings.append(msg)

    print("=== Validation Summary ===")
    if errors:
        print(f"Status: FAIL ({len(errors)} errors)")
        for e in errors:
            print(f"ERROR: {e}")
    else:
        print("Status: PASS")
        
    for w in warnings:
        print(f"WARN: {w}")

    return 1 if errors else 0

if __name__ == '__main__':
    exit(main())

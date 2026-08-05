from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import numpy as np

def get_dtype(dtype_str: str) -> np.dtype:
    mapping = {
        "float32": np.float32,
        "float16": np.float16,
        "uint16": np.uint16,
        "int32": np.int32,
        "int64": np.int64,
        "bfloat16": np.uint16,
        "torch.bfloat16": np.uint16,
    }
    return mapping.get(dtype_str, np.float32)

def dtype_size(dtype_str: str) -> int:
    return np.dtype(get_dtype(dtype_str)).itemsize

def main():
    parser = argparse.ArgumentParser(description="Inspect trace output")
    parser.add_argument("trace_dir", type=Path, help="Directory containing manifest.jsonl")
    parser.add_argument("--name", type=str, help="Specific tensor semantic_name to inspect")
    parser.add_argument("--filter", type=str, help="Filter tensors by fnmatch pattern")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    parser.add_argument("--verify", action="store_true", help="Check file sizes match shape*dtype_size")
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

    if args.filter:
        tensors = [t for t in tensors if fnmatch.fnmatch(t.get("semantic_name", ""), args.filter)]

    trace_tree_path = args.trace_dir / "trace_tree.txt"
    if trace_tree_path.exists() and not args.name:
        print("=== Trace Tree ===")
        with open(trace_tree_path, "r") as f:
            print(f.read())
        print("==================\n")

    if args.name:
        target = None
        for t in tensors:
            if t.get("semantic_name") == args.name:
                target = t
                break
        
        if not target:
            print(f"Tensor {args.name} not found.")
            return 1
            
        print("=== Tensor Details ===")
        for k, v in target.items():
            print(f"{k}: {v}")
            
        file_path = args.trace_dir / target.get("file", "")
        if file_path.exists():
            dtype = get_dtype(target.get("storage_dtype", "float32"))
            try:
                data = np.fromfile(file_path, dtype=dtype)
                shape = tuple(target.get("shape", []))
                if np.prod(shape) == data.size:
                    data = data.reshape(shape)
                print("\nData Preview:")
                print(data)
            except Exception as e:
                print(f"Failed to load data: {e}")
        else:
            print(f"File {file_path} not found.")
            
    else:
        print(f"{'Trace ID':<10} {'Semantic Name':<40} {'Shape':<20} {'Storage Dtype':<15}")
        print("-" * 90)
        for t in tensors:
            trace_id = t.get("trace_id", "")
            name = t.get("semantic_name", "")
            if len(name) > 38:
                name = name[:35] + "..."
            shape = str(t.get("shape", []))
            dtype = t.get("storage_dtype", "")
            print(f"{trace_id:<10} {name:<40} {shape:<20} {dtype:<15}")
            
            if args.stats:
                print(f"    Min: {t.get('min')} | Max: {t.get('max')} | Mean: {t.get('mean')} | Std: {t.get('std')}")
                
            if args.verify:
                file_path = args.trace_dir / t.get("file", "")
                if file_path.exists():
                    actual_size = file_path.stat().st_size
                    expected_size = np.prod(t.get("shape", [])) * dtype_size(dtype)
                    if actual_size != expected_size:
                        print(f"    [WARN] Size mismatch: expected {expected_size}, got {actual_size}")
                else:
                    print(f"    [WARN] Missing file: {file_path}")

    return 0

if __name__ == '__main__':
    exit(main())

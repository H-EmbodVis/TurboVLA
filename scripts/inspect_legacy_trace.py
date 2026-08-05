from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Inspect legacy trace output")
    parser.add_argument("trace_dir", type=Path, help="Directory containing manifest.json")
    parser.add_argument("--json", action="store_true", help="Output report in JSON format")
    args = parser.parse_args()

    manifest_path = args.trace_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} does not exist.")
        return 1

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    tensors = manifest.get("tensors", {})
    
    total_tensors = len(tensors)
    tensors_with_module_path = 0
    tensors_mapped = 0
    tensors_unmapped = 0
    missing_files = []
    shape_dtype_summary = {}

    semantic_keywords = ['vision', 'text', 'action', 'interaction']

    report_tensors = []

    for name, info in tensors.items():
        tensor_id = info.get("id", 0)
        stats = info.get("stats", {})
        shape = tuple(stats.get("shape", []))
        dtype = stats.get("dtype", "unknown")
        
        pt_file = args.trace_dir / info.get("pt_file", "")
        raw_file = args.trace_dir / info.get("raw_file", "")
        
        is_missing = False
        if info.get("pt_file") and not pt_file.exists():
            missing_files.append(str(pt_file))
            is_missing = True
        if info.get("raw_file") and not raw_file.exists():
            missing_files.append(str(raw_file))
            is_missing = True

        shape_dtype_key = f"{shape} {dtype}"
        shape_dtype_summary[shape_dtype_key] = shape_dtype_summary.get(shape_dtype_key, 0) + 1

        mapped_name = None
        for kw in semantic_keywords:
            if kw in name.lower():
                mapped_name = name
                break
        
        if mapped_name:
            tensors_mapped += 1
        else:
            mapped_name = f"legacy.unresolved.{tensor_id}"
            tensors_unmapped += 1

        has_module_path = "module_path" in info or "." in name
        if has_module_path:
            tensors_with_module_path += 1

        report_tensors.append({
            "id": tensor_id,
            "original_name": name,
            "mapped_name": mapped_name,
            "shape": shape,
            "dtype": dtype,
            "stats": stats,
            "missing_files": is_missing
        })

    report = {
        "summary": {
            "total_tensors": total_tensors,
            "tensors_with_module_path_info": tensors_with_module_path,
            "tensors_mapped_to_semantic_names": tensors_mapped,
            "tensors_unmapped": tensors_unmapped,
            "missing_files_count": len(missing_files),
            "missing_files": missing_files,
            "shape_dtype_summary": shape_dtype_summary
        },
        "tensors": report_tensors
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=== Migration Report ===")
        print(f"Total tensors: {total_tensors}")
        print(f"Tensors with module path info: {tensors_with_module_path}")
        print(f"Tensors mapped to semantic names: {tensors_mapped}")
        print(f"Tensors unmapped: {tensors_unmapped}")
        print(f"Missing files: {len(missing_files)}")
        if missing_files:
            for mf in missing_files[:5]:
                print(f"  - {mf}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")
        
        print("\nShape/Dtype Summary:")
        for sd, count in sorted(shape_dtype_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sd}: {count}")

        print("\nTensors (first 10):")
        for t in report_tensors[:10]:
            print(f"  [{t['id']}] {t['original_name']} -> {t['mapped_name']} | {t['shape']} {t['dtype']}")
        
    return 0

if __name__ == '__main__':
    exit(main())

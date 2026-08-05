#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an exhaustive TurboVLA PyTorch trace.")
    parser.add_argument("trace_dir", type=Path, help="Directory containing manifest.json")
    parser.add_argument("--filter", default="", help="Only show module/tensor names containing this text")
    parser.add_argument("--tensor", default="", help="Load tensors whose exact or partial name matches")
    parser.add_argument("--max_values", type=int, default=32, help="Maximum flattened values to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.trace_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest["metadata"], indent=2))

    print("\nModule calls:")
    for call in manifest["module_calls"]:
        label = (
            f"{call['global_call_index']:06d} "
            f"{call['module_name']}[{call['module_call_index']}] "
            f"{call['module_type']}"
        )
        if not args.filter or args.filter in label:
            print(label)

    print("\nTensors:")
    for name, entry in manifest["tensors"].items():
        if args.filter and args.filter not in name:
            continue
        stats = entry["stats"]
        print(
            f"{entry['id']:08d} {name}: shape={stats['shape']} dtype={stats['dtype']} "
            f"min={stats.get('min')} max={stats.get('max')} mean={stats.get('mean')}"
        )
        if args.tensor and (args.tensor == name or args.tensor in name):
            if "pt_file" not in entry:
                raise ValueError(f"Tensor {name!r} has no .pt file; enable embedding_dump.save_pt")
            value = torch.load(args.trace_dir / entry["pt_file"], map_location="cpu", weights_only=True)
            print(value.flatten()[: args.max_values])


if __name__ == "__main__":
    main()

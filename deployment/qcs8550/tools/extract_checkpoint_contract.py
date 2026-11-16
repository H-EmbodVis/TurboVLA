#!/usr/bin/env python3
"""Extract the small runtime contract needed by the TurboVLA QNN host adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT.parents[1] / "pretrained/TurboVLA/checkpoints/libero/object.pth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/checkpoint_contract.json")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint has no model_config")
    text = config.get("text")
    if not isinstance(text, dict):
        raise RuntimeError("checkpoint has no text configuration")
    layout = text.get("padding_length_by_instruction", {})
    if not isinstance(layout, dict):
        raise RuntimeError("padding_length_by_instruction is not a mapping")
    contract = {
        "checkpoint": str(args.checkpoint.resolve()),
        "text_padding_length": int(text["padding_length"]),
        "instruction_lengths": {str(key): int(value) for key, value in sorted(layout.items())},
        "special_token_strings": ["[CLS]", "[SEP]", ".", "?"],
        "image": {
            "height": 256,
            "width": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "libero_rotation": "flip height and width",
        },
        "state": {
            "mean": [
                -0.04190646484494209,
                0.03539437800645828,
                0.8257066607475281,
                2.908315658569336,
                -0.5562158823013306,
                -0.16649103164672852,
                0.02831534668803215,
                -0.028561558574438095,
            ],
            "std": [
                0.10743443667888641,
                0.14424759149551392,
                0.25723373889923096,
                0.34413808584213257,
                1.234430193901062,
                0.35798805952072144,
                0.013308786787092686,
                0.013174591585993767,
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "instructions": len(layout)}, indent=2))


if __name__ == "__main__":
    main()

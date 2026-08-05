from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from turbovla.debug import FixtureWriter


def main():
    parser = argparse.ArgumentParser(description="Export deterministic input fixture for TurboVLA parity testing")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/parity_traces/fixtures"), help="Root fixtures directory")
    parser.add_argument("--fixture-id", type=str, default="object_task0_episode0_step0", help="Fixture identifier")
    parser.add_argument("--instruction", type=str, default="pick up the alphabet soup and place it in the basket", help="Language instruction")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for synthetic inputs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fixture")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    writer = FixtureWriter(output_dir=args.output_dir, fixture_id=args.fixture_id, overwrite=args.overwrite)

    # 1. Instruction
    writer.write_instruction(args.instruction)

    # 2. Raw RGB Images (256x256x3 uint8)
    view_0_rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    view_1_rgb = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    writer.write_image("view_0_rgb_u8", view_0_rgb)
    writer.write_image("view_1_rgb_u8", view_1_rgb)

    # 3. Preprocessed pixel values (B=1, V=2, C=3, H=256, W=256)
    pixel_values = torch.randn(1, 2, 3, 256, 256, dtype=torch.float32)
    writer.write_tensor("pixel_values_f32", pixel_values, dtype_suffix="f32")

    # 4. State (Raw & Normalized) - 8D
    state_raw = torch.tensor([[-0.04, 0.035, 0.825, 2.9, -0.55, -0.16, 0.028, -0.028]], dtype=torch.float32)
    state_norm = torch.zeros(1, 8, dtype=torch.float32)
    writer.write_tensor("state_raw_f32", state_raw, dtype_suffix="f32")
    writer.write_tensor("state_normalized_f32", state_norm, dtype_suffix="f32")

    # 5. Token IDs & Masks (dummy placeholders for standalone replay)
    input_ids = torch.randint(100, 2000, (1, 21), dtype=torch.int64)
    attention_mask = torch.ones(1, 21, dtype=torch.uint8)
    text_self_attn_mask = torch.eye(21, dtype=torch.uint8).unsqueeze(0)
    position_ids = torch.arange(21, dtype=torch.int64).unsqueeze(0)

    writer.write_tensor("input_ids_i64", input_ids, dtype_suffix="bin")
    writer.write_tensor("attention_mask_u8", attention_mask, dtype_suffix="bin")
    writer.write_tensor("text_self_attention_mask_u8", text_self_attn_mask, dtype_suffix="bin")
    writer.write_tensor("position_ids_i64", position_ids, dtype_suffix="bin")

    metadata = {
        "fixture_version": 1,
        "fixture_id": args.fixture_id,
        "seed": args.seed,
        "instruction": args.instruction,
        "image_layout": "B,V,C,H,W",
        "state_layout": "B,D",
        "action_layout": "B,T,A",
        "model_precision": "bf16",
    }
    writer.write_metadata(metadata)

    fixture_path = writer.finalize()
    print(f"[Fixture] Successfully exported input fixture to {fixture_path}")


if __name__ == "__main__":
    main()

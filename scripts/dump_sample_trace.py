#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image
import torch
from transformers import BertConfig, BertModel, DINOv3ViTConfig, DINOv3ViTModel

from turbovla.models.configuration import TurboVLAConfig
from turbovla.models.turbovla import TurboVLA
from turbovla.models import text_encoder as text_encoder_module
from turbovla.models import vision_encoder as vision_encoder_module


DEFAULT_INSTRUCTION = "pick up the orange juice and place it in the basket"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump one reproducible exhaustive TurboVLA sample trace.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bert_path", type=Path, required=True)
    parser.add_argument("--primary_image", type=Path, required=True)
    parser.add_argument("--wrist_image", type=Path, required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--state", type=float, nargs=8, default=[0.0] * 8)
    parser.add_argument("--normalize_state", action="store_true")
    parser.add_argument("--stats_path", type=Path, default=Path("experiments/libero/configs/libero_all4_stats.json"))
    parser.add_argument("--stats_key", default="libero_all4_no_noops")
    parser.add_argument("--dataset_source", default=None)
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--primary_timestamp", type=float, default=None)
    parser.add_argument("--wrist_timestamp", type=float, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/embedding_dumps/sample"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> TurboVLA:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
    config.text.model_name_or_path = str(args.bert_path)
    config.text.local_files_only = True
    config.vision.model_name_or_path = "config-only-dinov3-vitb16"
    config.vision.local_files_only = True
    config.vision.compute_precision = "fp32"
    config.interaction.compute_precision = "fp32"
    config.embedding_dump.enabled = True
    config.embedding_dump.output_dir = str(args.output_dir)
    config.embedding_dump.max_dumps = 1
    config.embedding_dump.every_n_forwards = 1
    config.embedding_dump.rank_zero_only = True
    config.embedding_dump.print_summary = True
    config.embedding_dump.preview_values = 8
    config.embedding_dump.capture_module_io = True
    config.embedding_dump.save_pt = True
    config.embedding_dump.save_raw = True
    config.__post_init__()

    bert_config = BertConfig.from_pretrained(args.bert_path, local_files_only=True)
    text_encoder_module._load_pretrained_model = lambda _config: BertModel(bert_config)
    dinov3_config = DINOv3ViTConfig(
        image_size=config.vision.image_size,
        patch_size=16,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_register_tokens=4,
    )
    vision_encoder_module._load_pretrained_model = lambda _config: DINOv3ViTModel(dinov3_config)

    model = TurboVLA(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    return model


def prepare_sample(primary_path: Path, wrist_path: Path) -> tuple[torch.Tensor, Image.Image, Image.Image]:
    view_0 = Image.open(primary_path).convert("RGB")
    view_1 = Image.open(wrist_path).convert("RGB")
    if view_0.size != (256, 256) or view_1.size != (256, 256):
        raise ValueError(f"camera frames must both be 256x256, got {view_0.size} and {view_1.size}")
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def normalize(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return (torch.from_numpy(array).permute(2, 0, 1) - mean) / std

    pixel_values = torch.stack([normalize(view_0), normalize(view_1)], dim=0).unsqueeze(0)
    return pixel_values, view_0, view_1


def prepare_state(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    raw_state = torch.tensor([args.state], dtype=torch.float32)
    if not args.normalize_state:
        return raw_state, raw_state.clone()
    stats_payload = json.loads(args.stats_path.read_text(encoding="utf-8"))[args.stats_key]
    state_stats = stats_payload.get("proprio", stats_payload["state"])
    mean = torch.tensor(state_stats["mean"], dtype=torch.float32)
    std = torch.tensor(state_stats["std"], dtype=torch.float32)
    return raw_state, (raw_state - mean) / (std + 1e-6)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(args).to(args.device).eval()
    model.requires_grad_(False)
    pixel_values, view_0, view_1 = prepare_sample(args.primary_image, args.wrist_image)
    raw_state, state = prepare_state(args)
    with torch.inference_mode():
        actions = model(
            [args.instruction],
            {"dinov3": pixel_values.to(args.device)},
            state.to(args.device),
        ).cpu()

    trace_dir = model.embedding_dumper.last_trace_dir
    if trace_dir is None:
        raise RuntimeError("forward completed without producing a trace")
    view_0.save(trace_dir / "sample_view_0.png")
    view_1.save(trace_dir / "sample_view_1.png")
    torch.save(state, trace_dir / "sample_state.pt")
    torch.save(actions, trace_dir / "sample_actions.pt")
    (trace_dir / "sample_state_f32.bin").write_bytes(state.numpy().tobytes(order="C"))
    (trace_dir / "sample_actions_f32.bin").write_bytes(actions.float().numpy().tobytes(order="C"))
    sample = {
        "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "primary_image": str(args.primary_image.resolve()),
        "wrist_image": str(args.wrist_image.resolve()),
        "instruction": args.instruction,
        "raw_state": raw_state.tolist(),
        "normalized_state": state.tolist(),
        "state_stats": str(args.stats_path.resolve()) if args.normalize_state else None,
        "dataset_source": args.dataset_source,
        "episode_index": args.episode_index,
        "primary_timestamp": args.primary_timestamp,
        "wrist_timestamp": args.wrist_timestamp,
        "pixel_shape": list(pixel_values.shape),
        "action_shape": list(actions.shape),
        "actions": actions.tolist(),
    }
    (trace_dir / "sample.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"sample trace: {trace_dir}")
    print(f"actions: {actions}")


if __name__ == "__main__":
    main()

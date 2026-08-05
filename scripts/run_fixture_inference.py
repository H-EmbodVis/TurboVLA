from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from turbovla.debug import TraceConfig, TraceContext
from turbovla.models.turbovla import build_turbovla
from turbovla.models.configuration import TurboVLAConfig


def main():
    parser = argparse.ArgumentParser(description="Run TurboVLA inference on an exported fixture with parity tracing")
    parser.add_argument("--fixture", type=Path, required=True, help="Path to fixture directory")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to PyTorch model checkpoint (.pth)")
    parser.add_argument("--trace-root", type=Path, default=Path("outputs/parity_traces/python"), help="Output root for parity trace")
    parser.add_argument("--trace-level", type=str, default="boundary", choices=["boundary", "layer", "op"], help="Trace level")
    parser.add_argument("--trace-max-forwards", type=int, default=1, help="Max forward passes to trace")
    parser.add_argument("--trace-include", type=str, default="", help="Comma-separated include patterns")
    parser.add_argument("--trace-exclude", type=str, default="", help="Comma-separated exclude patterns")
    parser.add_argument("--trace-overwrite", action="store_true", help="Overwrite output if exists")
    parser.add_argument("--parity-mode", action="store_true", help="Enable deterministic parity mode (disable TF32, set deterministic)")
    parser.add_argument("--dump-weights-manifest", action="store_true", help="Dump weights_manifest.jsonl alongside trace")
    args = parser.parse_args()

    fixture_dir = args.fixture
    if not fixture_dir.exists():
        raise FileNotFoundError(f"Fixture directory {fixture_dir} not found")

    # Read instruction
    instruction = (fixture_dir / "instruction.txt").read_text().strip()

    # Load pixel values and state
    pixel_values_path = fixture_dir / "pixel_values_f32.npy"
    if pixel_values_path.exists():
        pixel_values = torch.from_numpy(np.load(pixel_values_path))
    else:
        # Fallback to random if standalone test
        pixel_values = torch.randn(1, 2, 3, 256, 256, dtype=torch.float32)

    state_path = fixture_dir / "state_raw_f32.npy"
    if state_path.exists():
        state = torch.from_numpy(np.load(state_path))
    else:
        state = torch.zeros(1, 8, dtype=torch.float32)

    fixture_id = fixture_dir.name

    # Build model (dummy config if checkpoint not supplied)
    bert_path = Path("pretrained/bert-base-uncased")
    dinov3_path = Path("pretrained/dinov3/dinov3-vitb16")
    if not dinov3_path.exists():
        dinov3_path = Path("pretrained/dinov3-vitb16-pretrain-lvd1689m")

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model_config = ckpt.get("model_config")
        if model_config:
            config = TurboVLAConfig.from_mapping(model_config)
            if bert_path.exists():
                config.text.model_name_or_path = str(bert_path)
            if dinov3_path.exists():
                config.vision.model_name_or_path = str(dinov3_path)
            model = build_turbovla(config)
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
            model.load_state_dict(state_dict, strict=False)
        else:
            config = TurboVLAConfig()
            if bert_path.exists():
                config.text.model_name_or_path = str(bert_path)
            if dinov3_path.exists():
                config.vision.model_name_or_path = str(dinov3_path)
            model = build_turbovla(config)
    else:
        print("[Fixture Inference] Warning: No valid checkpoint supplied. Running with pretrained backbone initialized model.")
        config = TurboVLAConfig()
        if bert_path.exists():
            config.text.model_name_or_path = str(bert_path)
        if dinov3_path.exists():
            config.vision.model_name_or_path = str(dinov3_path)
        model = build_turbovla(config)

    model.eval()

    # Deterministic parity mode (Section 13)
    if args.parity_mode:
        print("[Fixture Inference] Parity mode enabled: disabling TF32, enabling deterministic algorithms.")
        torch.manual_seed(7)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(7)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Setup TraceContext
    include_tuple = tuple(args.trace_include.split(",")) if args.trace_include else ()
    exclude_tuple = tuple(args.trace_exclude.split(",")) if args.trace_exclude else ()

    trace_config = TraceConfig(
        enabled=True,
        root_dir=args.trace_root,
        level=args.trace_level,
        max_forwards=args.trace_max_forwards,
        include=include_tuple,
        exclude=exclude_tuple,
        save_pt=True,
        save_f32=True,
        overwrite=args.trace_overwrite,
    )

    tracer = TraceContext(trace_config)
    tracer._requested_fixture_id = fixture_id
    model.set_parity_tracer(tracer)

    print(f"[Fixture Inference] Running forward pass for fixture '{fixture_id}' with trace level '{args.trace_level}'...")
    with torch.inference_mode():
        actions = model([instruction], pixel_values, state)

    print(f"[Fixture Inference] Completed! Action shape: {actions.shape}, action values:\n{actions[0, :3]}")

    # Dump parameter manifest (Section 14)
    if args.dump_weights_manifest:
        import hashlib
        trace_root = args.trace_root / fixture_id
        # Find the forward dir that was just created
        forward_dirs = sorted(trace_root.glob("forward_*"))
        if forward_dirs:
            manifest_path = forward_dirs[-1] / "weights_manifest.jsonl"
            print(f"[Fixture Inference] Writing weights manifest to {manifest_path}")
            import json as _json
            with open(manifest_path, "w") as wf:
                for name, param in model.named_parameters():
                    p = param.detach().cpu()
                    p_f32 = p.float().contiguous().numpy()
                    sha_f32 = hashlib.sha256(p_f32.tobytes()).hexdigest()
                    record = {
                        "name": name,
                        "shape": list(p.shape),
                        "dtype": str(p.dtype),
                        "numel": int(p.numel()),
                        "min": float(p_f32.min()),
                        "max": float(p_f32.max()),
                        "mean": float(p_f32.mean()),
                        "std": float(p_f32.std()),
                        "sha256_f32": sha_f32,
                    }
                    wf.write(_json.dumps(record) + "\n")
            print(f"[Fixture Inference] Wrote {sum(1 for _ in model.parameters())} parameter entries.")


if __name__ == "__main__":
    main()

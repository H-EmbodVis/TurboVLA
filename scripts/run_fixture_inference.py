from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from turbovla.debug import FixtureReader, TraceConfig, TraceContext, deterministic_context
from turbovla.debug.weights_writer import write_weights_and_buffers
from turbovla.evaluation.model_loader import load_turbovla_for_inference
from turbovla.evaluation.policy import rotate_libero_image
from turbovla.evaluation.replay import ExactInputs, load_suite_statistics, run_exact_replay


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly replay an authoritative TurboVLA fixture")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov3-path", required=True)
    parser.add_argument("--bert-path", required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--stats-key", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("bf16", "fp32"), required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--trace-level", choices=("boundary", "layer", "op", "exhaustive"), default="exhaustive")
    parser.add_argument("--trace-overwrite", action="store_true")
    parser.add_argument("--parity-mode", action="store_true")
    parser.add_argument("--dump-weights-manifest", action="store_true")
    args = parser.parse_args()

    loaded = load_turbovla_for_inference(
        checkpoint_path=args.checkpoint, dinov3_path=args.dinov3_path, bert_path=args.bert_path,
        device=args.device, precision=args.precision, strict=True, deterministic=args.parity_mode,
    )
    reader = FixtureReader(args.fixture)
    raw = np.stack([reader.load_numpy(f"raw/view_{index}_rgb_u8.npy") for index in range(2)])
    bf16 = loaded.model_dtype == torch.bfloat16
    exact = ExactInputs(
        raw_views=raw, rotated_views=np.stack([rotate_libero_image(item) for item in raw]),
        pixel_values_f32=reader.load_tensor("exact_inputs/pixel_values_f32.npy").float(),
        pixel_values_model_dtype=reader.load_tensor("exact_inputs/pixel_values_model_dtype.npy", bf16_bits=bf16).to(loaded.device),
        state_raw_f32=reader.load_tensor("raw/state_raw_f32.npy").float(),
        state_normalized_f32=reader.load_tensor("exact_inputs/state_normalized_f32.npy").float(),
        state_model_dtype=reader.load_tensor("exact_inputs/state_model_dtype.npy", bf16_bits=bf16).to(loaded.device),
        instruction=reader.instruction,
    )
    tracer = TraceContext(TraceConfig(
        enabled=True, root_dir=args.trace_root, level=args.trace_level, overwrite=args.trace_overwrite,
        save_pt=False, save_f32=True, save_native=True, fail_on_nan=True,
    ))
    stats = load_suite_statistics(args.stats_path, args.stats_key)
    with deterministic_context(7, enabled=args.parity_mode):
        result = run_exact_replay(loaded=loaded, exact=exact, statistics=stats, tracer=tracer,
                                  processor_config=reader.metadata["preprocessing"])
    if args.dump_weights_manifest:
        write_weights_and_buffers(loaded.model, args.trace_root / "weights")
    print(result.normalized_action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

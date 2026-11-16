r"""Benchmark LIBERO policy latency from GPU-ready inputs to a GPU action chunk.

Run from the repository root (no simulator or dataset is required)::

    python scripts/libero/benchmark_latency.py \
        --checkpoint pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth \
        --dinov3-path /path/to/dinov3-vitb --bert-path /path/to/bert-base-uncased \
        --device cuda:0 --cpu-threads 1 --output-dir outputs/latency

The default ``stats-bf16`` postprocessing uses BF16 action denormalization with
action ranges from the statistics JSON. The ``policy-fp32`` mode uses the public
CPU policy's constants, converts predictions to FP32 and sanitizes NaN/Inf
before action conversion. The selected mode is recorded in the output.

Timing includes online tokenization/token transfer, BERT, DINOv3, interaction,
ACT and GPU action postprocessing. Image/state preparation and transfer, action
return to CPU, environment execution, network and initialization are excluded.
Each call reuses fixed synthetic GPU inputs and a real instruction. No text
embedding cache, compilation or CUDA Graph is used. This is not a rollout or
a measurement of task success. Reported latency depends on hardware and runtime.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = "pick up the black bowl next to the cookie box and place it on the plate"
# Import torch only after argument parsing and thread/environment configuration.
torch: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov3-path", type=Path, required=True)
    parser.add_argument("--bert-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, default=ROOT / "experiments/libero/configs/libero_all4_stats.json")
    parser.add_argument("--stats-key", default="libero_all4_no_noops")
    parser.add_argument("--device", default="cuda:0", help="Logical CUDA device; respects CUDA_VISIBLE_DEVICES")
    parser.add_argument("--cpu-threads", type=int, default=1, help="PyTorch intra-op, OMP and MKL threads (default: 1)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument("--postprocess", choices=("stats-bf16", "policy-fp32"), default="stats-bf16")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latency"))
    parser.add_argument("--allow-busy-gpu", action="store_true", help="Allow detected competing GPU processes; record the count")
    args = parser.parse_args()
    for name in ("cpu_threads", "repeats", "blocks"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if not args.instruction.strip():
        parser.error("--instruction must not be empty")
    for path in (args.checkpoint, args.dinov3_path, args.bert_path):
        if not path.exists():
            parser.error(f"asset does not exist: {path}")
    if args.postprocess == "stats-bf16" and not args.stats_path.is_file():
        parser.error(f"statistics file does not exist: {args.stats_path}")
    return args


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def compute_processes() -> list[tuple[str, int]] | None:
    text = command_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    if text is None:
        return None
    rows = []
    for row in csv.reader(text.splitlines()):
        if len(row) == 2 and row[1].strip().isdigit():
            rows.append((row[0].strip(), int(row[1].strip())))
    return rows


def selected_gpu_uuid(device: Any) -> str | None:
    # Creating one CUDA context before this query identifies the selected device
    # by this process's PID, without guessing indices from CUDA_VISIBLE_DEVICES.
    value = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if value:
        return str(value)
    rows = compute_processes()
    own = {uuid for uuid, pid in rows or [] if pid == os.getpid()}
    return next(iter(own)) if len(own) == 1 else None


def gpu_snapshot(uuid: str | None) -> dict[str, Any]:
    fields = ("index", "uuid", "name", "driver_version", "memory.total", "memory.used",
              "utilization.gpu", "clocks.sm", "temperature.gpu", "power.draw", "power.limit")
    text = command_output(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if text is not None and uuid is not None:
        for row in csv.reader(text.splitlines()):
            if len(row) == len(fields) and row[1].strip() == uuid:
                return dict(zip(fields, (value.strip() for value in row)))
    return {"status": "unavailable", "reason": "nvidia-smi metadata or selected GPU UUID unavailable"}


def check_idle(uuid: str | None, allow_busy: bool) -> dict[str, Any]:
    rows = compute_processes()
    if rows is None or uuid is None:
        return {"status": "unverified", "other_compute_process_count": None}
    count = sum(gpu == uuid and pid != os.getpid() for gpu, pid in rows)
    if count and not allow_busy:
        raise RuntimeError(f"Selected GPU has {count} other compute process(es); use another device or --allow-busy-gpu")
    return {"status": "checked", "other_compute_process_count": count}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q / 100
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def summarize(values: list[float]) -> dict[str, float]:
    return dict(median_ms=statistics.median(values), p90_ms=percentile(values, 90),
                mean_ms=statistics.fmean(values), min_ms=min(values), max_ms=max(values))


def main() -> None:
    args = parse_args()
    os.environ.update(OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
        USE_TORCH="1", USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    global torch
    import torch as torch_module
    import numpy as np

    torch = torch_module
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device and a CUDA-enabled PyTorch build")
    torch.cuda.set_device(device)
    context_probe = torch.empty(1, device=device)
    uuid = selected_gpu_uuid(device)
    occupancy_before = check_idle(uuid, args.allow_busy_gpu)
    gpu_before = gpu_snapshot(uuid)
    del context_probe
    sys.path.insert(0, str(ROOT))
    from turbovla.evaluation.policy import (
        ACTION_MAX, ACTION_MIN, TurboVLAPolicy, normalized_action_to_env_action, sanitize_pred_chunk,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(args.checkpoint)
    policy = TurboVLAPolicy(ckpt_path=str(args.checkpoint), dinov3_path=str(args.dinov3_path),
        bert_path=str(args.bert_path), device=device, precision="bf16", allow_hf_download=False)
    model, cfg = policy.model, policy.model.config
    actual = (cfg.vision.num_views, cfg.vision.image_size, cfg.action.state_dim,
              cfg.action.horizon, cfg.action.action_dim, cfg.interaction.num_layers)
    if actual != (2, 256, 8, 12, 7, 6):
        raise ValueError(f"Expected released LIBERO configuration (2,256,8,12,7,6), got {actual}")
    if {p.dtype for p in model.parameters() if p.is_floating_point()} != {torch.bfloat16}:
        raise RuntimeError("All floating model parameters must be BF16")
    images = {"dinov3": torch.randn(1, 2, 3, 256, 256, device=device, dtype=torch.bfloat16)}
    states = torch.randn(1, 8, device=device, dtype=torch.bfloat16)
    if args.postprocess == "stats-bf16":
        stats = json.loads(args.stats_path.read_text(encoding="utf-8"))[args.stats_key]["action"]
        lower, upper = stats["min"], stats["max"]
    else:
        lower, upper = ACTION_MIN.tolist(), ACTION_MAX.tolist()
    if len(lower) != 7 or len(upper) != 7 or not all(math.isfinite(x) for x in (*lower, *upper)):
        raise ValueError("Action min/max must contain seven finite values each")
    post_dtype = torch.bfloat16 if args.postprocess == "stats-bf16" else torch.float32
    amin = torch.tensor(lower, device=device, dtype=post_dtype)
    amax = torch.tensor(upper, device=device, dtype=post_dtype)

    def postprocess(normalized: Any) -> Any:
        if args.postprocess == "policy-fp32":
            normalized = torch.nan_to_num(normalized.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        arm = 0.5 * (normalized[..., :6] + 1.0)
        arm = arm * (amax[:6] - amin[:6]) + amin[:6]
        grip = normalized[..., 6:7]
        grip = grip.new_ones(grip.shape).masked_fill(grip < 0, -1.0)
        return torch.cat((arm, grip), dim=-1)

    def infer() -> Any:
        return postprocess(model([args.instruction], images, states))

    enhancer = model.vision_language_interaction
    def validate() -> dict[str, Any]:
        counts = {name: 0 for name in ("fusion", "text_self_attention", "visual_values", "text_values", "bert", "vision", "act")}

        def hook(name: str) -> Any:
            def count(*_: Any) -> None:
                counts[name] += 1
            return count

        handles = []
        for block, text_layer in zip(enhancer.fusion_layers, enhancer.text_layers):
            for module, name in ((block, "fusion"), (text_layer, "text_self_attention"),
                                 (block.attn.values_v_proj, "visual_values"), (block.attn.values_l_proj, "text_values")):
                handles.append(module.register_forward_hook(hook(name)))
        for module, name in ((model.text_encoder.bert, "bert"), (model.vision_encoder.backbone, "vision"), (model.action_head, "act")):
            handles.append(module.register_forward_hook(hook(name)))
        try:
            raw = model([args.instruction], images, states)
            output = postprocess(raw)
            torch.cuda.synchronize(device)
        finally:
            for handle in handles:
                handle.remove()
        expected = dict(fusion=6, text_self_attention=6, visual_values=6, text_values=6,
            bert=1, vision=2 if cfg.vision.encode_views_separately else 1, act=1)
        if counts != expected or tuple(output.shape) != (1, 12, 7) or not output.is_cuda:
            raise RuntimeError(f"Invalid execution path/output: counts={counts}, expected={expected}")
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Non-finite action output")
        validation: dict[str, Any] = {"execution_counts": counts, "output_shape": list(output.shape),
            "output_dtype": str(output.dtype), "output_device": str(output.device)}
        if args.postprocess == "policy-fp32":
            edge = torch.tensor([float("nan"), float("inf"), -float("inf"), -1.0, -0.0, 0.0, 1.0],
                                device=device, dtype=torch.bfloat16).view(1, 7, 1).expand(1, 7, 7)
            for value in (raw, edge):
                clean = sanitize_pred_chunk(value[0].detach().float().cpu().numpy())
                cpu_reference = np.stack([normalized_action_to_env_action(row) for row in clean])
                np.testing.assert_allclose(postprocess(value)[0].cpu().numpy(), cpu_reference, rtol=0, atol=1e-6)
            validation["official_cpu_postprocess_check"] = "passed on model output and nonfinite/boundary fixture"
        return validation

    with torch.inference_mode():
        validation = validate()
        occupancy_start = check_idle(uuid, args.allow_busy_gpu)
        blocks = []
        for block_index in range(args.blocks):
            for _ in range(args.warmup):
                output = infer()
                del output
            torch.cuda.synchronize(device)
            times = []
            for _ in range(args.repeats):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                output = infer()
                torch.cuda.synchronize(device)
                times.append((time.perf_counter() - start) * 1000)
                del output
            blocks.append({"block": block_index + 1, **summarize(times), "raw_ms": times})
        row = dict(mode="bidirectional", blocks=blocks, validation=validation, occupancy_start=occupancy_start,
            occupancy_end=check_idle(uuid, args.allow_busy_gpu),
            median_of_block_medians_ms=statistics.median(block["median_ms"] for block in blocks))
        results = [row]
        print(f"bidirectional: median of block medians = {row['median_of_block_medians_ms']:.3f} ms", flush=True)

    packages = {}
    for name in ("torch", "transformers", "tokenizers", "timm", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    files = [Path(__file__).resolve(), *sorted((ROOT / "turbovla").rglob("*.py"))]
    token_length = cfg.text.padding_length_by_instruction.get(args.instruction, cfg.text.padding_length)
    tokens = model.text_encoder.tokenizer([args.instruction], padding="max_length" if token_length else "longest",
        truncation=True, max_length=token_length or cfg.text.max_length, return_tensors="pt")
    metadata = dict(status="completed", timestamp_utc=datetime.now(timezone.utc).isoformat(),
        repository=dict(commit=command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            dirty=command_output(["git", "-C", str(ROOT), "status", "--porcelain"])),
        checkpoint=dict(name=args.checkpoint.name, sha256=checkpoint_hash, initialization="strict loaded checkpoint"),
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in files},
        environment=dict(python=sys.version, platform=platform.platform(), packages=packages,
            torch_build=torch.__version__, cuda_runtime=torch.version.cuda,
            cpu_model=command_output(["lscpu"]), cpu_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(), omp_threads=os.environ["OMP_NUM_THREADS"],
            mkl_threads=os.environ["MKL_NUM_THREADS"], cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
        gpu_before=gpu_before, gpu_after=gpu_snapshot(uuid), occupancy_before=occupancy_before,
        model_config=cfg.to_dict(),
        input=dict(shape=list(images["dinov3"].shape), state_shape=list(states.shape), dtype="bfloat16",
            instruction=args.instruction, nonpadding_tokens=int(tokens.attention_mask.sum()),
            padded_tokens=int(tokens.input_ids.shape[1]), seed=args.seed,
            semantics="same fixed synthetic GPU-ready image/state tensors reused for every call"),
        protocol=dict(warmup=args.warmup, repeats=args.repeats, blocks=args.blocks,
            timer="perf_counter; CUDA synchronization before and after every timed call",
            scope="GPU-ready images/state to complete GPU action chunk; online tokenization and token H2D included",
            excluded=["image/state preparation and H2D", "action D2H", "environment", "network", "initialization"],
            postprocess=args.postprocess, action_min=lower, action_max=upper,
            stats_sha256=sha256(args.stats_path) if args.postprocess == "stats-bf16" else None,
            statistics_source=args.stats_key if args.postprocess == "stats-bf16" else "public policy ACTION_MIN/ACTION_MAX",
            compile=False, cuda_graph=False, text_embedding_cache=False),
        notes=["Measures the released full bidirectional model with synthetic inputs, not rollout success rate.",
               "GPU occupancy checks are discrete, not continuous monitoring.",
               "stats-bf16 and policy-fp32 use different constants/precision; compare matching contracts.",
               "Latency depends on hardware and runtime."], results=results)
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "raw_latency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "block", "iteration", "latency_ms"))
        for row in results:
            for block in row["blocks"]:
                writer.writerows((row["mode"], block["block"], index + 1, value) for index, value in enumerate(block["raw_ms"]))
    print(f"Saved {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

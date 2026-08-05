from __future__ import annotations

import hashlib
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .tensor_record import TensorRecord


def compute_tensor_stats(tensor: torch.Tensor, num_first_last: int = 4) -> dict[str, Any]:
    """Compute comprehensive stats for a tensor, handling NaN/Inf/empty gracefully."""
    if tensor.numel() == 0:
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "abs_max": 0.0,
            "l1_norm": 0.0, "l2_norm": 0.0, "zero_count": 0, "zero_ratio": 0.0,
            "nan_count": 0, "inf_count": 0, "first_values": [], "last_values": [],
        }

    t_f32 = tensor.detach().cpu().float()

    nan_mask = torch.isnan(t_f32)
    inf_mask = torch.isinf(t_f32)
    valid_mask = ~(nan_mask | inf_mask)

    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())

    valid_t = t_f32[valid_mask]
    if valid_t.numel() > 0:
        t_min = float(valid_t.min().item())
        t_max = float(valid_t.max().item())
        t_mean = float(valid_t.mean().item())
        t_std = float(valid_t.std(unbiased=False).item()) if valid_t.numel() > 1 else 0.0
        t_abs_max = float(valid_t.abs().max().item())
        t_l1 = float(valid_t.abs().sum().item())
        t_l2 = float(valid_t.norm(p=2).item())
    else:
        t_min = t_max = t_mean = t_std = t_abs_max = t_l1 = t_l2 = 0.0

    zero_count = int((t_f32 == 0).sum().item())
    zero_ratio = float(zero_count) / tensor.numel()

    flat_t = t_f32.flatten()
    n_elems = min(num_first_last, flat_t.numel())
    first_values = flat_t[:n_elems].tolist()
    last_values = flat_t[-n_elems:].tolist() if n_elems > 0 else []

    return {
        "min": t_min, "max": t_max, "mean": t_mean, "std": t_std, "abs_max": t_abs_max,
        "l1_norm": t_l1, "l2_norm": t_l2, "zero_count": zero_count, "zero_ratio": zero_ratio,
        "nan_count": nan_count, "inf_count": inf_count,
        "first_values": first_values, "last_values": last_values,
    }


def sha256_of_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


# Map the first segment of a semantic name to a numbered prefix folder.
_STAGE_MAP: dict[str, str] = {
    "input": "00_input",
    "text": "10_text_encoder",
    "text_encoder": "10_text_encoder",
    "vision": "20_vision_encoder",
    "vision_encoder": "20_vision_encoder",
    "vision_projection": "30_vision_projection",
    "interaction": "40_interaction",
    "condition": "45_condition",
    "state_projection": "50_state_projection",
    "state": "50_state_projection",
    "action": "60_action_decoder",
    "action_decoder": "60_action_decoder",
    "output": "70_output",
}


def stage_prefix(stage: str) -> str:
    """Map stage name to numbered prefix folder name."""
    return _STAGE_MAP.get(stage.lower(), stage)


def semantic_name_to_subdir(semantic_name: str) -> Path:
    """Convert a semantic name to a directory path under the tensors/ folder.

    Example:
        "interaction.layer_02.cross.visual_to_text.probs"
        -> "tensors/40_interaction/layer_02"
    """
    parts = semantic_name.split(".")
    stage = parts[0]
    prefix = stage_prefix(stage)

    # Build subdirectory from the layer/block parts, not the final tensor name.
    # We include up to the layer index level for organization.
    subdirs = [prefix]
    for part in parts[1:]:
        # Stop adding subdirs once we hit something that looks like an op/variant
        # Keep going for parts that look like structural components (layer_NN, block_NN, view_N)
        if part.startswith(("layer_", "block_", "view_", "decoder", "encoder")):
            subdirs.append(part)
        else:
            break

    return Path("tensors") / Path(*subdirs)


def canonical_f32_bytes(tensor: torch.Tensor) -> bytes:
    """Convert tensor to canonical float32 little-endian contiguous bytes."""
    arr = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
    return arr.tobytes()


def raw_bf16_bytes(tensor: torch.Tensor) -> bytes:
    """Convert BF16 tensor to raw uint16 little-endian bytes."""
    if tensor.dtype != torch.bfloat16:
        return b""
    raw = (
        tensor.detach()
        .cpu()
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype("<u2", copy=False)
    )
    return raw.tobytes()


def build_trace_tree(records: list[TensorRecord]) -> str:
    """Build a hierarchical trace tree text from a list of records.

    Produces output like:
        00 input
          input.pixel_values
          input.input_ids
        10 text_encoder
          text.bert.embeddings
          ...
    """
    if not records:
        return "(empty trace)\n"

    # Group records by stage prefix
    stage_records: dict[str, list[TensorRecord]] = defaultdict(list)
    for record in records:
        parts = record.semantic_name.split(".")
        stage = parts[0]
        prefix = stage_prefix(stage)
        stage_records[prefix].append(record)

    lines: list[str] = []
    for prefix in sorted(stage_records.keys()):
        lines.append(f"\n{prefix}")
        for record in stage_records[prefix]:
            shape_str = "x".join(map(str, record.shape))
            dtype_short = record.source_dtype.replace("torch.", "")
            lines.append(f"  {record.trace_id:04d} {record.semantic_name}  [{shape_str}] {dtype_short}")

    return "\n".join(lines) + "\n"


def build_environment_json() -> dict[str, Any]:
    """Collect Python/torch/numpy/CUDA/platform versions."""
    import numpy as np

    env: dict[str, Any] = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "byte_order": sys.byteorder,
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
    return env

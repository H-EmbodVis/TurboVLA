from __future__ import annotations

import hashlib
import math
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .tensor_record import TensorRecord


def compute_tensor_stats(tensor: torch.Tensor, num_first_last: int = 4) -> dict[str, Any]:
    if tensor.numel() == 0:
        return {key: 0 for key in ("min", "max", "mean", "std", "abs_max", "l1_norm", "l2_norm",
                                    "zero_count", "zero_ratio", "nan_count", "inf_count")} | {
            "first_values": [], "last_values": []}
    value = tensor.detach().cpu().float()
    nan_mask, inf_mask = torch.isnan(value), torch.isinf(value)
    valid = value[~(nan_mask | inf_mask)]
    if valid.numel():
        valid64 = valid.double()
        stats = {
            "min": float(valid64.min()), "max": float(valid64.max()), "mean": float(valid64.mean()),
            "std": float(valid64.std(unbiased=False)) if valid.numel() > 1 else 0.0,
            "abs_max": float(valid.abs().max()), "l1_norm": float(valid64.abs().sum()),
            "l2_norm": float(valid64.norm(p=2)),
        }
    else:
        stats = {key: 0.0 for key in ("min", "max", "mean", "std", "abs_max", "l1_norm", "l2_norm")}
    flat = value.flatten()
    count = min(num_first_last, flat.numel())
    stats.update(
        zero_count=int((value == 0).sum()), zero_ratio=float((value == 0).sum()) / value.numel(),
        nan_count=int(nan_mask.sum()), inf_count=int(inf_mask.sum()),
        first_values=[value if math.isfinite(value) else None for value in flat[:count].tolist()],
        last_values=[value if math.isfinite(value) else None for value in flat[-count:].tolist()],
    )
    return stats


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


_STAGE_MAP = {
    "input": "00_input", "text": "10_text_encoder", "vision": "20_vision_encoder",
    "vision_projection": "30_vision_projection", "interaction": "40_interaction",
    "condition": "45_condition", "state": "50_state_projection", "state_projection": "50_state_projection",
    "action": "60_action_decoder", "output": "70_output",
}


def stage_prefix(stage: str) -> str:
    return _STAGE_MAP.get(stage.lower(), stage)


def semantic_name_to_subdir(semantic_name: str, *, native: bool = False) -> Path:
    parts = semantic_name.split(".")
    subdirs = [stage_prefix(parts[0])]
    for part in parts[1:]:
        if part.startswith(("layer_", "block_", "view_")):
            subdirs.append(part)
        else:
            break
    return Path("tensors_native" if native else "tensors", *subdirs)


def sanitize_filename(value: str) -> str:
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", value)


def canonical_bytes(tensor: torch.Tensor) -> tuple[bytes, str, str]:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.bool:
        return cpu.to(torch.uint8).numpy().tobytes(), "uint8", "u8"
    if not cpu.is_floating_point():
        if cpu.dtype in (torch.uint8, torch.int8):
            array = cpu.numpy().astype("u1" if cpu.dtype == torch.uint8 else "i1", copy=False)
            return array.tobytes(), "uint8" if cpu.dtype == torch.uint8 else "int8", "u8" if cpu.dtype == torch.uint8 else "i8"
        mapping = {torch.int16: ("<i2", "int16", "i16"), torch.int32: ("<i4", "int32", "i32"),
                   torch.int64: ("<i8", "int64", "i64")}
        if cpu.dtype not in mapping:
            raise TypeError(f"unsupported integer trace dtype {cpu.dtype}")
        np_dtype, storage, suffix = mapping[cpu.dtype]
        return cpu.numpy().astype(np_dtype, copy=False).tobytes(), storage, suffix
    data = cpu.float().numpy().astype("<f4", copy=False).tobytes()
    return data, "float32", "f32le"


def native_bytes(tensor: torch.Tensor) -> tuple[bytes, str, str]:
    cpu = tensor.detach().cpu().contiguous()
    if cpu.dtype == torch.bfloat16:
        return cpu.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(), "bfloat16", "bf16le"
    if cpu.dtype == torch.float16:
        return cpu.numpy().astype("<f2", copy=False).tobytes(), "float16", "f16le"
    canonical, storage, suffix = canonical_bytes(cpu)
    return canonical, storage, suffix


def canonical_f32_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False).tobytes()


def raw_bf16_bytes(tensor: torch.Tensor) -> bytes:
    return native_bytes(tensor)[0] if tensor.dtype == torch.bfloat16 else b""


def build_trace_tree(records: list[TensorRecord]) -> str:
    grouped: dict[str, list[TensorRecord]] = defaultdict(list)
    for record in records:
        grouped[stage_prefix(record.stage or record.semantic_name.split(".")[0])].append(record)
    lines: list[str] = []
    for stage in sorted(grouped):
        lines.append(stage)
        lines.extend(f"  {r.trace_id:06d} {r.semantic_name} call={r.call_index} [{','.join(map(str, r.shape))}]" for r in grouped[stage])
    return "\n".join(lines) + ("\n" if lines else "")


def build_environment_json() -> dict[str, Any]:
    info = {"python_version": sys.version, "torch_version": torch.__version__, "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "platform": platform.platform(), "byte_order": sys.byteorder}
    if torch.cuda.is_available():
        info.update(gpu_name=torch.cuda.get_device_name(0), gpu_count=torch.cuda.device_count())
    return info

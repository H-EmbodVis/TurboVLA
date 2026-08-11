from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from ..models.configuration import TurboVLAConfig
from ..models.turbovla import TurboVLA, build_turbovla


@dataclass(frozen=True)
class LoadedTurboVLA:
    model: TurboVLA
    config: TurboVLAConfig
    checkpoint_sha256: str
    config_sha256: str
    device: torch.device
    model_dtype: torch.dtype
    state_tensor_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint).__name__}")
    for key in ("model_state_dict", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(isinstance(item, torch.Tensor) for item in value.values()):
            return dict(value)
    raise KeyError("checkpoint does not contain a non-empty model_state_dict/model/state_dict tensor mapping")


def _strip_distributed_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for name, tensor in state.items():
        cleaned = name.removeprefix("module.")
        if cleaned in result:
            raise KeyError(f"duplicate checkpoint key after removing module prefix: {cleaned}")
        result[cleaned] = tensor
    return result


def _normalized_config_hash(config: TurboVLAConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_turbovla_for_inference(
    *,
    checkpoint_path: Path,
    dinov3_path: str,
    bert_path: str,
    device: str | torch.device,
    precision: Literal["bf16", "fp32"],
    strict: bool = True,
    deterministic: bool = False,
) -> LoadedTurboVLA:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not strict:
        raise ValueError("the shared inference loader requires strict=True")
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be 'bf16' or 'fp32'")
    for label, value in (("DINOv3", dinov3_path), ("BERT", bert_path)):
        if not value:
            raise ValueError(f"{label} path must be explicit")
        if not Path(value).expanduser().exists():
            raise FileNotFoundError(f"local {label} model path not found: {value}")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_config"), dict):
        raise KeyError("checkpoint must contain the exact model_config mapping")
    config = TurboVLAConfig.from_mapping(checkpoint["model_config"])
    config.text.model_name_or_path = str(Path(bert_path).expanduser().resolve())
    config.text.local_files_only = True
    config.vision.model_name_or_path = str(Path(dinov3_path).expanduser().resolve())
    config.vision.local_files_only = True
    config.vision.compute_precision = "bf16" if precision == "bf16" else "fp32"
    config.embedding_dump.enabled = False
    if deterministic:
        config.text.attention_implementation = "eager"
        config.vision.attention_implementation = "eager"
        config.interaction.attention_backend = "manual"
        config.interaction.compute_precision = "fp32" if precision == "fp32" else config.interaction.compute_precision
    config.__post_init__()

    model = build_turbovla(config)
    state = _strip_distributed_prefix(checkpoint_state_dict(checkpoint))
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"strict checkpoint load unexpectedly returned missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model.to(device=target_device, dtype=model_dtype)
    model.eval()
    model.requires_grad_(False)
    bad_parameters = [(name, str(value.dtype)) for name, value in model.named_parameters()
                      if value.is_floating_point() and value.dtype != model_dtype]
    bad_buffers = [(name, str(value.dtype)) for name, value in model.named_buffers()
                   if value.is_floating_point() and value.dtype != model_dtype]
    if bad_parameters or bad_buffers:
        raise RuntimeError(f"model precision verification failed: parameters={bad_parameters}, buffers={bad_buffers}")
    return LoadedTurboVLA(
        model=model, config=config, checkpoint_sha256=sha256_file(checkpoint_path),
        config_sha256=_normalized_config_hash(config), device=target_device,
        model_dtype=model_dtype, state_tensor_count=len(state),
    )

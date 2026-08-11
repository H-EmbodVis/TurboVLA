"""TurboVLA policy adapter for aligned LIBERO evaluation.

This module keeps the released LIBERO evaluation protocol local to its adapter:
256px DINOv3 preprocessing, state normalization, hard min/max
action denormalization, and the original gripper sign rule.
"""

from __future__ import annotations

import json
import math
import os
import random
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
import torch

from .model_loader import load_turbovla_for_inference


EXPECTED_IMAGE_SIZE = 256
DINO_PATCH_SIZE = 16
ACTION_CHUNK_SIZE = 12
ACTION_DIM = 7
STATE_DIM = 8

DEFAULT_DINOV3_PATH = ""
DEFAULT_BERT_PATH = ""

ACTION_MIN = np.asarray(
    [
        -0.9375,
        -0.9375,
        -0.9375,
        -0.23642857372760773,
        -0.3053571283817291,
        -0.3675000071525574,
        -1.0,
    ],
    dtype=np.float32,
)

ACTION_MAX = np.asarray(
    [
        0.9375,
        0.9375,
        0.9375,
        0.30000001192092896,
        0.29357144236564636,
        0.375,
        1.0,
    ],
    dtype=np.float32,
)

PROPRIO_MEAN = np.asarray(
    [
        -0.04190646484494209,
        0.03539437800645828,
        0.8257066607475281,
        2.908315658569336,
        -0.5562158823013306,
        -0.16649103164672852,
        0.02831534668803215,
        -0.028561558574438095,
    ],
    dtype=np.float32,
)

PROPRIO_STD = np.asarray(
    [
        0.10743443667888641,
        0.14424759149551392,
        0.25723373889923096,
        0.34413808584213257,
        1.234430193901062,
        0.35798805952072144,
        0.013308786787092686,
        0.013174591585993767,
    ],
    dtype=np.float32,
)


def configure_transformers_offline(allow_hf_download: bool = False) -> None:
    if allow_hf_download:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_preprocessor_config(local_model_path: str) -> dict[str, Any]:
    cfg_path = os.path.join(local_model_path, "preprocessor_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_manual_rgb_normalizer(
    image_mean: Sequence[float],
    image_std: Sequence[float],
    rescale_factor: float,
    expected_size: int,
    patch_size: int,
    backbone_name: str,
):
    mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
    rescale_factor = float(rescale_factor)

    def process_one(img: Image.Image | np.ndarray) -> torch.Tensor:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")

        width, height = img.size
        if height != expected_size or width != expected_size:
            raise ValueError(
                f"{backbone_name} expects pre-rotated {expected_size}x{expected_size} RGB input, "
                f"but got {height}x{width}. Do not apply StarVLA/OpenVLA 224px resize here."
            )
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"{backbone_name} input size {(height, width)} must be divisible by patch size {patch_size}."
            )

        arr = np.asarray(img, dtype=np.float32) * rescale_factor
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return (x - mean) / std

    def processor(images: Image.Image | np.ndarray | Sequence[Image.Image | np.ndarray]) -> dict[str, torch.Tensor]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        pixel_values = torch.stack([process_one(im) for im in images], dim=0)
        return {"pixel_values": pixel_values}

    return processor


def build_dinov3_manual_processor(local_dinov3_path: str):
    cfg = load_preprocessor_config(local_dinov3_path)
    return _build_manual_rgb_normalizer(
        image_mean=cfg.get("image_mean", [0.485, 0.456, 0.406]),
        image_std=cfg.get("image_std", [0.229, 0.224, 0.225]),
        rescale_factor=cfg.get("rescale_factor", 1.0 / 255.0),
        expected_size=EXPECTED_IMAGE_SIZE,
        patch_size=DINO_PATCH_SIZE,
        backbone_name="DINOv3",
    )


def rotate_libero_image(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - float(quat[3]) * float(quat[3])))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def state_from_libero_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def normalize_state(state_np: np.ndarray) -> torch.Tensor:
    state = np.asarray(state_np, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"TurboVLA state must have {STATE_DIM} dims, got {state.shape}")
    return torch.from_numpy((state - PROPRIO_MEAN) / (PROPRIO_STD + 1e-6)).float()


def denormalize_arm_action(action_norm_np: np.ndarray) -> np.ndarray:
    action = np.asarray(action_norm_np, dtype=np.float32).reshape(-1)
    if action.shape[0] < 6:
        raise ValueError(f"TurboVLA action must have at least 6 dims, got {action.shape}")
    action = action[:6].copy()
    return 0.5 * (action + 1.0) * (ACTION_MAX[:6] - ACTION_MIN[:6]) + ACTION_MIN[:6]


def gripper_command_from_norm(gripper_norm: float, deadband: float = 0.0) -> float:
    if gripper_norm > deadband:
        return 1.0
    if gripper_norm < -deadband:
        return -1.0
    return 1.0


def normalized_action_to_env_action(
    action_norm_np: np.ndarray,
    gripper_deadband: float = 0.0,
) -> np.ndarray:
    action_norm_np = np.asarray(action_norm_np, dtype=np.float32).reshape(-1)
    arm_action = denormalize_arm_action(action_norm_np)
    gripper_source = float(action_norm_np[6])
    gripper = np.asarray([gripper_command_from_norm(gripper_source, gripper_deadband)], dtype=np.float32)
    return np.concatenate([arm_action, gripper], axis=0).astype(np.float32)


def sanitize_pred_chunk(pred_chunk: np.ndarray) -> np.ndarray:
    pred_chunk = np.asarray(pred_chunk, dtype=np.float32)
    if pred_chunk.ndim == 1:
        pred_chunk = pred_chunk[None, :]
    elif pred_chunk.ndim > 2:
        pred_chunk = pred_chunk.reshape(pred_chunk.shape[0], -1)

    valid_actions = []
    for row in pred_chunk:
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.size < ACTION_DIM:
            continue
        if row.size > ACTION_DIM:
            row = row[:ACTION_DIM]
        row = np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=-1.0)
        valid_actions.append(row.astype(np.float32))

    if not valid_actions:
        return np.zeros((1, ACTION_DIM), dtype=np.float32)
    return np.stack(valid_actions, axis=0)


def get_libero_dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def _make_model_args(
    dinov3_path: str,
    bert_path: str,
    hidden_dim: int,
    nheads: int,
    dim_feedforward: int,
    max_text_len: int,
    vla_feature_enhancer_layers: int,
    enhancer_inner_dim: int,
    action_dim: int,
    chunk_size: int,
    state_dim: int,
    num_state_tokens: int,
    text_dropout: float,
    fusion_dropout: float,
    fusion_droppath: float,
    sub_sentence_present: bool,
    text_padding_length: int,
    precision: str,
    allow_hf_download: bool,
    dump_embeddings: bool,
    embedding_dump_dir: str,
    embedding_dump_max: int,
    embedding_dump_every: int,
    embedding_dump_print_summary: bool,
    embedding_dump_preview_values: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        dinov3_path=dinov3_path,
        bert_path=bert_path,
        hidden_dim=hidden_dim,
        nheads=nheads,
        dim_feedforward=dim_feedforward,
        max_text_len=max_text_len,
        text_padding_length=text_padding_length,
        sub_sentence_present=sub_sentence_present,
        vla_feature_enhancer_layers=vla_feature_enhancer_layers,
        enhancer_inner_dim=enhancer_inner_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        state_dim=state_dim,
        num_state_tokens=num_state_tokens,
        text_dropout=text_dropout,
        fusion_dropout=fusion_dropout,
        fusion_droppath=fusion_droppath,
        local_files_only=not allow_hf_download,
        freeze_text_encoder=True,
        freeze_vision_encoder=True,
        dinov3_precision="bf16" if precision == "bf16" else "bf16_autocast",
        num_views=2,
        image_size=EXPECTED_IMAGE_SIZE,
        position_embedding="view",
        encode_views_separately=True,
        padding_strategy="key_padding_mask",
        dump_embeddings=dump_embeddings,
        embedding_dump_dir=embedding_dump_dir,
        embedding_dump_max=embedding_dump_max,
        embedding_dump_every=embedding_dump_every,
        embedding_dump_print_summary=embedding_dump_print_summary,
        embedding_dump_preview_values=embedding_dump_preview_values,
    )


def load_turbovla_builder():
    from ..models.turbovla import build_turbovla

    return build_turbovla, "turbovla.models.turbovla"


class TurboVLAPolicy:
    def __init__(
        self,
        ckpt_path: str,
        dinov3_path: str = DEFAULT_DINOV3_PATH,
        bert_path: str = DEFAULT_BERT_PATH,
        device: str | torch.device | None = None,
        allow_hf_download: bool = False,
        hidden_dim: int = 256,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        max_text_len: int = 256,
        text_padding_length: int = 21,
        vla_feature_enhancer_layers: int = 6,
        enhancer_inner_dim: int = 1024,
        action_dim: int = ACTION_DIM,
        chunk_size: int = ACTION_CHUNK_SIZE,
        state_dim: int = STATE_DIM,
        num_state_tokens: int = 2,
        text_dropout: float = 0.0,
        fusion_dropout: float = 0.0,
        fusion_droppath: float = 0.1,
        sub_sentence_present: bool = True,
        precision: str = "bf16",
        dinov3_output_hidden_states: bool = True,
        verbose: bool = True,
        dump_embeddings: bool = False,
        embedding_dump_dir: str = "embedding_dumps",
        embedding_dump_max: int = 1,
        embedding_dump_every: int = 1,
        embedding_dump_print_summary: bool = True,
        embedding_dump_preview_values: int = 8,
    ) -> None:
        configure_transformers_offline(allow_hf_download=allow_hf_download)

        self.ckpt_path = str(ckpt_path)
        self.dinov3_path = str(dinov3_path)
        self.bert_path = str(bert_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.precision = str(precision).lower()
        self.model_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        self.dinov3_output_hidden_states = bool(dinov3_output_hidden_states)
        self.verbose = bool(verbose)
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported precision={precision!r}; expected 'bf16' or 'fp32'.")

        if not self.dinov3_path:
            raise ValueError("dinov3_path must point to a local DINOv3 checkpoint or a HF model id")
        if not self.bert_path:
            raise ValueError("bert_path must point to BERT base uncased or a compatible model id")
        if not allow_hf_download and not os.path.isdir(self.dinov3_path):
            raise FileNotFoundError(f"local DINOv3 directory not found: {self.dinov3_path}")
        if not allow_hf_download and not os.path.isdir(self.bert_path):
            raise FileNotFoundError(f"local BERT directory not found: {self.bert_path}")

        loaded_model_path = "turbovla.evaluation.model_loader.load_turbovla_for_inference"
        if self.verbose:
            print(f"[TurboVLAPolicy] model source: {loaded_model_path}", flush=True)
            print(
                "[TurboVLAPolicy] "
                f"precision={self.precision}, "
                f"dinov3_output_hidden_states={self.dinov3_output_hidden_states}",
                flush=True,
            )
        if allow_hf_download:
            raise ValueError("production inference requires explicit local BERT and DINO paths")
        loaded = load_turbovla_for_inference(
            checkpoint_path=Path(self.ckpt_path), dinov3_path=self.dinov3_path,
            bert_path=self.bert_path, device=self.device, precision=self.precision,
            strict=True, deterministic=False,
        )
        self.loaded_model = loaded
        self.model = loaded.model
        self.chunk_size = loaded.config.action.horizon
        self.action_dim = loaded.config.action.action_dim
        if dump_embeddings:
            warnings.warn(
                "EmbeddingDumper is legacy-only and is not valid for parity output",
                DeprecationWarning, stacklevel=2,
            )
            self.model.embedding_dumper.config.enabled = True
        self._verify_model_precision()
        self.dinov3_processor = build_dinov3_manual_processor(self.dinov3_path)

    def _set_eval_precision(self) -> None:
        if self.precision == "bf16":
            self.model.to(dtype=torch.bfloat16)
        else:
            # FP32 evaluation keeps all parameters FP32 while autocasting only
            # the DINOv3 forward pass to BF16.
            self.model.float()

    def _verify_model_precision(self) -> None:
        floating_dtypes = {
            param.dtype
            for param in self.model.parameters()
            if param.is_floating_point()
        }
        expected = {self.model_dtype}
        if floating_dtypes != expected:
            raise RuntimeError(
                f"precision={self.precision} expected model parameter dtypes {expected}, "
                f"got {floating_dtypes}"
            )
        if self.verbose:
            dtype_name = str(self.model_dtype).removeprefix("torch.")
            print(
                f"[TurboVLAPolicy] precision={self.precision}, model_parameter_dtype={dtype_name}",
                flush=True,
            )

    def _load_checkpoint(self) -> None:
        source_state = _strip_module_prefix(_checkpoint_state_dict(self._checkpoint))
        self.model.load_state_dict(source_state, strict=True)
        if self.verbose:
            print(
                f"[TurboVLAPolicy] strict checkpoint load: {self.ckpt_path} ({len(source_state)} tensors)",
                flush=True,
            )
        del self._checkpoint

    def _build_batch(
        self,
        primary_images: Sequence[np.ndarray],
        wrist_images: Sequence[np.ndarray],
        states: Sequence[np.ndarray],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        flat_images: list[np.ndarray] = []
        for primary, wrist in zip(primary_images, wrist_images):
            flat_images.extend([primary, wrist])

        dinov3_pixel_values = self.dinov3_processor(flat_images)["pixel_values"]
        batch_size = len(primary_images)
        samples = {
            "dinov3": dinov3_pixel_values.view(batch_size, 2, *dinov3_pixel_values.shape[1:]).to(self.device),
        }
        state_tensors = torch.stack([normalize_state(state) for state in states], dim=0).to(self.device)
        return samples, state_tensors

    def _prepare_model_inputs(
        self,
        samples: dict[str, torch.Tensor],
        states: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        samples = {
            key: value.to(dtype=self.model_dtype) if value.is_floating_point() else value
            for key, value in samples.items()
        }
        return samples, states.to(dtype=self.model_dtype)

    def predict_normalized_action_chunk(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
    ) -> np.ndarray:
        if isinstance(state_or_obs, dict):
            state = state_from_libero_obs(state_or_obs)
        else:
            state = np.asarray(state_or_obs, dtype=np.float32)

        samples, states = self._build_batch([primary_image], [wrist_image], [state])
        samples, states = self._prepare_model_inputs(samples, states)
        tracer = getattr(self.model, "_parity_tracer", None)
        started = False
        if tracer is not None and not tracer.active:
            started = tracer.begin(forward_index=0, fixture_id=getattr(tracer, "requested_fixture_id", ""))
        try:
            with torch.inference_mode():
                pred = self.model([instruction], samples, states)
            if tracer is not None and tracer.active:
                tracer.tensor("action.normalized", pred, layout="B,T,A", operation="tanh", required_level="boundary")
        except Exception as error:
            if started:
                tracer.finish(error=error)
            raise
        if pred.dtype != self.model_dtype:
            raise RuntimeError(
                f"precision={self.precision} expected forward output dtype {self.model_dtype}, got {pred.dtype}"
            )
        result = sanitize_pred_chunk(pred.detach().float().cpu().numpy()[0])
        # Lifecycle remains open for environment-action postprocessing.
        return result

    def set_parity_tracer(self, tracer) -> None:
        if hasattr(self.model, "set_parity_tracer"):
            self.model.set_parity_tracer(tracer)

    def predict_env_action_chunk(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
        execute_steps: int | None = None,
    ) -> np.ndarray:
        pred_norm = self.predict_normalized_action_chunk(primary_image, wrist_image, instruction, state_or_obs)
        env_actions = np.stack(
            [
                normalized_action_to_env_action(row)
                for row in pred_norm
            ],
            axis=0,
        )
        tracer = getattr(self.model, "_parity_tracer", None)
        if tracer is not None and tracer.active:
            tracer.tensor("action.denormalized", torch.from_numpy(env_actions), layout="T,A", operation="denormalize", required_level="boundary")
        if execute_steps is not None:
            env_actions = env_actions[: int(execute_steps)]
        if tracer is not None and tracer.active:
            tracer.tensor("action.first_step", torch.from_numpy(env_actions[:1]), layout="T,A", operation="slice", required_level="boundary")
            tracer.tensor("action.executed_steps", torch.from_numpy(env_actions), layout="T,A", operation="slice", required_level="boundary")
            tracer.finish()
        return env_actions.astype(np.float32)

    def predict_env_action_chunk_from_obs(
        self,
        obs: dict[str, Any],
        instruction: str,
        execute_steps: int | None = None,
    ) -> np.ndarray:
        primary = rotate_libero_image(obs["agentview_image"])
        wrist = rotate_libero_image(obs["robot0_eye_in_hand_image"])
        return self.predict_env_action_chunk(primary, wrist, instruction, obs, execute_steps=execute_steps)


def batched(iterable: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]

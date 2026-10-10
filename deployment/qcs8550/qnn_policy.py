"""Host-side preprocessing and action decoding for the persistent TurboVLA QNN service."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


QCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = QCS_ROOT.parents[1]
if str(QCS_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(QCS_ROOT / "tools"))

from native_client import run_request


DEFAULT_CONTRACT = QCS_ROOT / "artifacts/checkpoint_contract.json"
DEFAULT_BERT = REPO_ROOT / "pretrained/bert-base-uncased"
ACTION_MIN = np.asarray(
    (-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3675000071525574),
    dtype=np.float32,
)
ACTION_MAX = np.asarray(
    (0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375),
    dtype=np.float32,
)


def rotate_libero_image(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - float(quat[3]) ** 2))
    if np.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(float(quat[3])) / denominator).astype(np.float32)


def _special_token_masks(input_ids: np.ndarray, special_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match TurboVLA generate_masks_with_special_tokens for batch size one."""
    ids = np.asarray(input_ids, dtype=np.int32)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"expected one tokenized instruction, got {ids.shape}")
    length = ids.shape[1]
    attention = np.eye(length, dtype=np.bool_)[None]
    position_ids = np.zeros((1, length), dtype=np.int32)
    previous = 0
    for column in np.flatnonzero(np.isin(ids[0], special_ids)):
        if column == 0 or column == length - 1:
            attention[0, column, column] = True
            position_ids[0, column] = 0
        else:
            attention[0, previous + 1 : column + 1, previous + 1 : column + 1] = True
            position_ids[0, previous + 1 : column + 1] = np.arange(0, column - previous, dtype=np.int32)
        previous = int(column)
    return attention, position_ids


class TurboVLAQnnPolicy:
    """TurboVLA LIBERO policy with DINO/BERT/policy-core execution on QCS8550."""

    def __init__(
        self,
        *_,
        service_host: str = os.environ.get("TURBOVLA_QNN_HOST", "127.0.0.1"),
        service_port: int = int(os.environ.get("TURBOVLA_QNN_PORT", "10092")),
        contract_path: str | Path = DEFAULT_CONTRACT,
        bert_path: str | Path = DEFAULT_BERT,
        timeout: float = 120.0,
        **__,
    ) -> None:
        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
        self.instruction_lengths = {str(key): int(value) for key, value in contract["instruction_lengths"].items()}
        self.output_length = int(contract["text_padding_length"])
        self.mean = np.asarray(contract["image"]["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(contract["image"]["std"], dtype=np.float32)[:, None, None]
        self.proprio_mean = np.asarray(contract["state"]["mean"], dtype=np.float32)
        self.proprio_std = np.asarray(contract["state"]["std"], dtype=np.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(str(bert_path), local_files_only=True, use_fast=True)
        self.special_ids = np.asarray(
            self.tokenizer.convert_tokens_to_ids(contract["special_token_strings"]), dtype=np.int32
        )
        self.service_host = service_host
        self.service_port = int(service_port)
        self.timeout = float(timeout)
        self.last_metrics: dict[str, object] = {}

    def _text_inputs(self, instruction: str) -> dict[str, np.ndarray]:
        if instruction not in self.instruction_lengths:
            raise KeyError(f"instruction is not in the checkpoint static-length contract: {instruction!r}")
        length = self.instruction_lengths[instruction]
        tokens = self.tokenizer(
            [instruction], padding="max_length", truncation=True, max_length=length, return_tensors="np"
        )
        input_ids = np.ascontiguousarray(tokens["input_ids"], dtype=np.int32)
        token_type_ids = np.ascontiguousarray(tokens["token_type_ids"], dtype=np.int32)
        token_attention = np.asarray(tokens["attention_mask"], dtype=np.bool_)
        bert_attention_mask, position_ids = _special_token_masks(input_ids, self.special_ids)
        key_padding = np.ones((1, self.output_length), dtype=np.bool_)
        key_padding[:, :length] = ~token_attention
        policy_attention = np.eye(self.output_length, dtype=np.bool_)[None]
        policy_attention[:, :length, :length] = bert_attention_mask
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "bert_attention_mask": bert_attention_mask,
            "position_ids": position_ids,
            "text_key_padding_mask": key_padding,
            "text_self_attention_mask": policy_attention,
        }

    def _state(self, state_or_obs: np.ndarray | dict[str, Any]) -> np.ndarray:
        if isinstance(state_or_obs, dict):
            state = np.concatenate(
                (
                    np.asarray(state_or_obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                    quat2axisangle(state_or_obs["robot0_eef_quat"]),
                    np.asarray(state_or_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
                )
            )
        else:
            state = np.asarray(state_or_obs, dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"TurboVLA requires an 8-D state, got {state.shape}")
        return np.ascontiguousarray(((state - self.proprio_mean) / (self.proprio_std + 1e-6))[None], dtype=np.float32)

    def _pixels(self, primary: np.ndarray, wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        views = []
        for image in (primary, wrist):
            rgb = np.asarray(image, dtype=np.float32)
            if rgb.shape != (256, 256, 3):
                raise ValueError(f"TurboVLA requires a 256x256 RGB image, got {rgb.shape}")
            views.append(np.ascontiguousarray((np.transpose(rgb / 255.0, (2, 0, 1)) - self.mean) / self.std))
        return views[0][None], views[1][None]

    def request_arrays(
        self,
        primary: np.ndarray,
        wrist: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
    ) -> dict[str, np.ndarray]:
        view0, view1 = self._pixels(primary, wrist)
        return {
            "pixels_view0": view0,
            "pixels_view1": view1,
            **self._text_inputs(instruction),
            "state": self._state(state_or_obs),
        }

    def predict_normalized_action_chunk(
        self,
        primary: np.ndarray,
        wrist: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
    ) -> np.ndarray:
        action, self.last_metrics = run_request(
            self.service_host,
            self.service_port,
            self.request_arrays(primary, wrist, instruction, state_or_obs),
            self.timeout,
        )
        return np.nan_to_num(action[0], nan=0.0, posinf=1.0, neginf=-1.0).clip(-1.0, 1.0)

    def predict_env_action_chunk(
        self,
        primary: np.ndarray,
        wrist: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
        execute_steps: int | None = None,
    ) -> np.ndarray:
        normalized = self.predict_normalized_action_chunk(primary, wrist, instruction, state_or_obs)
        arm = 0.5 * (normalized[:, :6] + 1.0) * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
        gripper = np.where(normalized[:, 6:7] >= 0.0, 1.0, -1.0).astype(np.float32)
        actions = np.concatenate((arm, gripper), axis=1).astype(np.float32)
        return actions if execute_steps is None else actions[: int(execute_steps)]

    def predict_env_action_chunk_from_obs(
        self,
        obs: dict[str, Any],
        instruction: str,
        execute_steps: int | None = None,
    ) -> np.ndarray:
        return self.predict_env_action_chunk(
            rotate_libero_image(obs["agentview_image"]),
            rotate_libero_image(obs["robot0_eye_in_hand_image"]),
            instruction,
            obs,
            execute_steps,
        )

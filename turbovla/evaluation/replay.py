from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..debug.model_trace import emit
from .policy import rotate_libero_image, state_from_libero_obs


@dataclass(frozen=True)
class SuiteStatistics:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    stats_path: str
    stats_key: str


@dataclass
class ExactInputs:
    raw_views: np.ndarray
    rotated_views: np.ndarray
    pixel_values_f32: torch.Tensor
    pixel_values_model_dtype: torch.Tensor
    state_raw_f32: torch.Tensor
    state_normalized_f32: torch.Tensor
    state_model_dtype: torch.Tensor
    instruction: str


@dataclass
class ReplayResult:
    normalized_action: np.ndarray
    env_action: np.ndarray
    first_step: np.ndarray
    executed_steps: np.ndarray


def capture_libero_observation(*, libero_root: Path | str, task_suite: str, task_id: int,
                               episode: int, step: int, seed: int) -> tuple[dict[str, Any], str, dict[str, Any]]:
    import os
    import sys

    libero_root = Path(libero_root).expanduser().resolve()
    if not libero_root.is_dir():
        raise FileNotFoundError(f"LIBERO root not found: {libero_root}")
    sys.path.insert(0, str(libero_root)) if str(libero_root) not in sys.path else None
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    suites = benchmark.get_benchmark_dict()
    if task_suite not in suites:
        raise KeyError(f"unknown LIBERO suite {task_suite!r}; available={sorted(suites)}")
    suite = suites[task_suite]()
    if not 0 <= task_id < suite.n_tasks:
        raise IndexError(f"task_id {task_id} outside [0,{suite.n_tasks})")
    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    if not 0 <= episode < len(states):
        raise IndexError(f"episode {episode} outside [0,{len(states)})")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(states[episode])
        dummy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        for _ in range(step):
            obs, _, done, _ = env.step(dummy)
            if done:
                raise RuntimeError(f"LIBERO episode terminated before requested fixture step {step}")
        frozen = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in obs.items()}
    finally:
        env.close()
    metadata = {
        "task_suite": task_suite, "task_id": task_id, "task_name": getattr(task, "name", ""),
        "episode": episode, "step": step, "seed": seed, "bddl_file": str(bddl),
        "initial_state_count": len(states),
    }
    return frozen, str(task.language), metadata


def load_suite_statistics(path: Path | str, key: str) -> SuiteStatistics:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key not in payload or not isinstance(payload[key], dict):
        raise KeyError(f"stats key {key!r} not found in {path}")
    selected = payload[key]
    state_section = selected.get("proprio", selected.get("state"))
    if not isinstance(state_section, dict) or not isinstance(selected.get("action"), dict):
        raise KeyError("stats must contain proprio/state and action mappings")
    return SuiteStatistics(
        state_mean=np.asarray(state_section["mean"], dtype=np.float32),
        state_std=np.asarray(state_section["std"], dtype=np.float32),
        action_min=np.asarray(selected["action"]["min"], dtype=np.float32),
        action_max=np.asarray(selected["action"]["max"], dtype=np.float32),
        stats_path=str(path), stats_key=key,
    )


def derive_exact_inputs(*, obs: dict[str, Any], instruction: str, processor, statistics: SuiteStatistics,
                        model_dtype: torch.dtype, device: torch.device) -> ExactInputs:
    raw_views = np.stack([
        np.ascontiguousarray(obs["agentview_image"], dtype=np.uint8),
        np.ascontiguousarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8),
    ])
    rotated_views = np.stack([rotate_libero_image(view) for view in raw_views])
    pixel_values = processor(list(rotated_views))["pixel_values"].view(1, 2, 3, 256, 256).contiguous()
    state = state_from_libero_obs(obs)
    normalized = (state - statistics.state_mean) / (statistics.state_std + 1e-6)
    state_raw = torch.from_numpy(state).view(1, -1).float()
    state_normalized = torch.from_numpy(normalized).view(1, -1).float()
    return ExactInputs(
        raw_views=raw_views, rotated_views=rotated_views,
        pixel_values_f32=pixel_values.float(),
        pixel_values_model_dtype=pixel_values.to(device=device, dtype=model_dtype),
        state_raw_f32=state_raw,
        state_normalized_f32=state_normalized,
        state_model_dtype=state_normalized.to(device=device, dtype=model_dtype),
        instruction=instruction,
    )


def trace_preprocessing(tracer, exact: ExactInputs, processor_config: dict, statistics: SuiteStatistics) -> None:
    device = exact.pixel_values_model_dtype.device
    dtype = exact.pixel_values_model_dtype.dtype
    mean = torch.tensor(processor_config.get("image_mean", [0.485, 0.456, 0.406]), device=device, dtype=torch.float32)
    std = torch.tensor(processor_config.get("image_std", [0.229, 0.224, 0.225]), device=device, dtype=torch.float32)
    factor = torch.tensor(float(processor_config.get("rescale_factor", 1 / 255)), device=device)
    for index in range(2):
        raw = torch.from_numpy(exact.raw_views[index]).to(device)
        rotated = torch.from_numpy(exact.rotated_views[index]).to(device)
        rescaled = rotated.float() * factor
        normalized = (rescaled.permute(2, 0, 1) - mean[:, None, None]) / std[:, None, None]
        emit(tracer, f"input.raw.view_{index}_rgb_u8", raw, "H,W,C", "libero_observation", "boundary")
        emit(tracer, f"input.rotated.view_{index}_rgb_u8", rotated, "H,W,C", "rotate_180", "boundary")
        emit(tracer, f"input.rescaled.view_{index}", rescaled, "H,W,C", "multiply", "exhaustive")
        emit(tracer, f"input.normalized.view_{index}", normalized, "C,H,W", "normalize", "op")
    emit(tracer, "input.rescale.factor", factor, "", "constant", "exhaustive")
    emit(tracer, "input.normalize.mean", mean, "C", "constant", "op")
    emit(tracer, "input.normalize.std", std, "C", "constant", "op")
    emit(tracer, "input.pixel_values_f32", exact.pixel_values_f32.to(device), "B,V,C,H,W", "stack", "boundary")
    emit(tracer, "input.pixel_values_model_dtype", exact.pixel_values_model_dtype, "B,V,C,H,W", "cast", "boundary")
    state_raw = exact.state_raw_f32.to(device)
    state_mean = torch.from_numpy(statistics.state_mean).to(device)
    state_std = torch.from_numpy(statistics.state_std).to(device)
    centered = state_raw - state_mean
    emit(tracer, "state.raw", state_raw, "B,D", "libero_state", "boundary")
    emit(tracer, "state.normalize.mean", state_mean, "D", "constant", "op")
    emit(tracer, "state.normalize.std", state_std, "D", "constant", "op")
    emit(tracer, "state.centered", centered, "B,D", "subtract", "exhaustive")
    emit(tracer, "state.normalized_f32", exact.state_normalized_f32.to(device), "B,D", "divide", "boundary")
    emit(tracer, "state.normalized_model_dtype", exact.state_model_dtype, "B,D", "cast", "boundary")
    instruction_bytes = torch.tensor(list(exact.instruction.encode("utf-8")), dtype=torch.uint8, device=device)
    emit(tracer, "text.instruction_utf8_bytes", instruction_bytes, "BYTES", "utf8_encode", "boundary")


def denormalize_actions(action: torch.Tensor, statistics: SuiteStatistics, tracer=None, execute_steps: int | None = None) -> ReplayResult:
    action = action.float()
    device = action.device
    minimum = torch.from_numpy(statistics.action_min[:6]).to(device)
    maximum = torch.from_numpy(statistics.action_max[:6]).to(device)
    source = action[..., 6]
    plus_one = action[..., :6] + 1.0
    action_range = maximum - minimum
    scaled = 0.5 * plus_one * action_range
    arm = scaled + minimum
    deadband = torch.tensor(0.0, device=device)
    positive, negative = source > deadband, source < -deadband
    gripper = torch.where(positive, torch.ones_like(source), torch.where(negative, -torch.ones_like(source), torch.ones_like(source)))
    env = torch.cat([arm, gripper.unsqueeze(-1)], dim=-1)
    emit(tracer, "action.denormalize.action_min", minimum, "A6", "constant", "op")
    emit(tracer, "action.denormalize.action_max", maximum, "A6", "constant", "op")
    emit(tracer, "action.denormalize.input", action[..., :6], "B,T,A6", "slice", "op")
    emit(tracer, "action.denormalize.plus_one", plus_one, "B,T,A6", "add", "exhaustive")
    emit(tracer, "action.denormalize.range", action_range, "A6", "subtract", "exhaustive")
    emit(tracer, "action.denormalize.scaled", scaled, "B,T,A6", "multiply", "exhaustive")
    emit(tracer, "action.denormalize.arm_output", arm, "B,T,A6", "add", "op")
    emit(tracer, "action.gripper.source", source, "B,T", "slice", "op")
    emit(tracer, "action.gripper.deadband", deadband, "", "constant", "exhaustive")
    emit(tracer, "action.gripper.positive_mask", positive, "B,T", "greater", "exhaustive")
    emit(tracer, "action.gripper.negative_mask", negative, "B,T", "less", "exhaustive")
    emit(tracer, "action.gripper.output", gripper, "B,T", "where", "op")
    emit(tracer, "action.denormalized", env, "B,T,A", "concat", "boundary")
    steps = env if execute_steps is None else env[:, :execute_steps]
    emit(tracer, "action.first_step", steps[:, :1], "B,1,A", "slice", "boundary")
    emit(tracer, "action.executed_steps", steps, "B,E,A", "slice", "boundary")
    return ReplayResult(
        normalized_action=action.detach().cpu().numpy(), env_action=env.detach().cpu().numpy(),
        first_step=steps[:, :1].detach().cpu().numpy(), executed_steps=steps.detach().cpu().numpy(),
    )


def run_exact_replay(*, loaded, exact: ExactInputs, statistics: SuiteStatistics, tracer=None,
                     processor_config: dict | None = None, execute_steps: int | None = None) -> ReplayResult:
    started = False
    if tracer is not None and not tracer.active:
        started = tracer.begin(0)
    try:
        if tracer is not None:
            trace_preprocessing(tracer, exact, processor_config or {}, statistics)
        loaded.model.set_parity_tracer(tracer)
        with torch.inference_mode():
            normalized = loaded.model(
                [exact.instruction], {"dinov3": exact.pixel_values_model_dtype}, exact.state_model_dtype
            )
        result = denormalize_actions(normalized, statistics, tracer=tracer, execute_steps=execute_steps)
    except Exception as error:
        if started:
            tracer.finish(error=error)
        raise
    finally:
        loaded.model.set_parity_tracer(None)
    if started:
        tracer.finish()
    return result

"""VLA-Adapter-derived LIBERO evaluation for a TurboVLA checkpoint.

This script intentionally reuses the VLA-Adapter task/episode protocol shape,
but bypasses OpenVLA/Prismatic preprocessing and action unnormalization. The
policy adapter keeps TurboVLA's 256px DINOv3 inputs, proprio stats, hard
action min/max, and gripper sign rule.

VLA-Adapter is MIT-licensed; see ../LICENSES/VLA-Adapter.txt.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, fields
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional

import imageio
import numpy as np
import tqdm


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

LIBERO_SUITES = tuple(TASK_MAX_STEPS.keys())


@dataclass
class GenerateConfig:
    ckpt_path: str = ""
    libero_root: str = ""
    dinov3_path: str = ""
    bert_path: str = ""
    stats_path: str = "experiments/libero/configs/libero_all4_stats.json"
    stats_key: str = "libero_all4_no_noops"
    normalize_binary_gripper: str = "auto"
    allow_hf_download: bool = False

    task_suite_name: str = "libero_10"
    task_ids: str = ""
    num_trials_per_task: int = 50
    max_tasks: int = -1
    num_steps_wait: int = 10
    num_open_loop_steps: int = 12
    env_img_res: int = 256
    seed: int = 7
    control_mode: str = "relative"
    mujoco_gl: str = "osmesa"
    pyopengl_platform: str = "osmesa"
    osmesa_library: str = ""

    save_video: bool = False
    video_out_path: str = "outputs/evaluation"
    result_json_path: str = ""
    log_path: str = ""
    dry_run_model_load: bool = False

    hidden_dim: int = 256
    nheads: int = 8
    dim_feedforward: int = 2048
    max_text_len: int = 256
    text_padding_length: int = 21
    vla_feature_enhancer_layers: int = 6
    enhancer_inner_dim: int = 1024
    action_dim: int = 7
    chunk_size: int = 12
    state_dim: int = 8
    num_state_tokens: int = 2
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    sub_sentence_present: bool = True
    precision: str = "bf16"
    dinov3_output_hidden_states: bool = True
    dump_embeddings: bool = False
    embedding_dump_dir: str = "embedding_dumps"
    embedding_dump_max: int = 1
    embedding_dump_every: int = 1
    embedding_dump_print_summary: bool = True
    embedding_dump_preview_values: int = 8


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> GenerateConfig:
    parser = argparse.ArgumentParser(
        description="VLA-Adapter-derived aligned LIBERO evaluation for TurboVLA checkpoints."
    )
    for field in fields(GenerateConfig):
        name = field.name
        default = field.default
        arg_name = f"--{name}"
        dashed_arg_name = f"--{name.replace('_', '-')}"
        kwargs = {"default": default, "help": f"default: {default}"}
        if isinstance(default, bool):
            parser.add_argument(arg_name, dashed_arg_name, type=_parse_bool, nargs="?", const=True, **kwargs)
            parser.add_argument(f"--no_{name}", f"--no-{name.replace('_', '-')}", dest=name, action="store_false")
        elif name == "precision":
            parser.add_argument(arg_name, dashed_arg_name, type=str, choices=("bf16", "fp32"), **kwargs)
        elif isinstance(default, int):
            parser.add_argument(arg_name, dashed_arg_name, type=int, **kwargs)
        elif isinstance(default, float):
            parser.add_argument(arg_name, dashed_arg_name, type=float, **kwargs)
        else:
            parser.add_argument(arg_name, dashed_arg_name, type=str, **kwargs)
    return GenerateConfig(**vars(parser.parse_args()))


def _import_turbovla_adapter():
    from turbovla.evaluation.suite_policy import (
        TurboVLAPolicy,
        get_libero_dummy_action,
        rotate_libero_image,
        set_seed_everywhere,
    )

    return TurboVLAPolicy, get_libero_dummy_action, rotate_libero_image, set_seed_everywhere


def _ensure_libero_import_path(cfg: GenerateConfig) -> None:
    if cfg.mujoco_gl:
        os.environ.setdefault("MUJOCO_GL", cfg.mujoco_gl)
    if cfg.pyopengl_platform:
        os.environ.setdefault("PYOPENGL_PLATFORM", cfg.pyopengl_platform)
    if cfg.osmesa_library:
        os.environ.setdefault("OSMESA_LIBRARY", cfg.osmesa_library)
        render_lib_dir = str(Path(cfg.osmesa_library).resolve().parent)
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        ld_parts = [item for item in current_ld_path.split(":") if item]
        if render_lib_dir not in ld_parts:
            os.environ["LD_LIBRARY_PATH"] = ":".join([render_lib_dir, *ld_parts])

    candidates = []
    if cfg.libero_root:
        candidates.append(Path(cfg.libero_root))
    for path in candidates:
        if path.exists():
            resolved = str(path.resolve())
            while resolved in sys.path:
                sys.path.remove(resolved)
            sys.path.insert(0, resolved)
            # A local experiments/libero package may already have claimed the
            # top-level name before the external benchmark path was selected.
            sys.modules.pop("libero", None)
            return


def _setup_logging(cfg: GenerateConfig) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.log_path:
        Path(cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def _try_set_control_mode(env, control_mode: str) -> bool:
    candidates = [
        env,
        getattr(env, "env", None),
        getattr(env, "unwrapped", None),
        getattr(getattr(env, "env", None), "env", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        if hasattr(obj, "control_mode"):
            try:
                setattr(obj, "control_mode", control_mode)
                return True
            except Exception:
                pass
        if hasattr(obj, "set_control_mode"):
            try:
                obj.set_control_mode(control_mode)
                return True
            except Exception:
                pass
    return False


def _make_libero_env(task, cfg: GenerateConfig):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": cfg.env_img_res,
        "camera_widths": cfg.env_img_res,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(cfg.seed)
    _try_set_control_mode(env, cfg.control_mode)
    return env, task_description


def _save_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [np.asarray(x) for x in frames], fps=fps)


def _result_path(cfg: GenerateConfig) -> Path:
    if cfg.result_json_path:
        return Path(cfg.result_json_path)
    ckpt_tag = Path(cfg.ckpt_path).stem
    return Path(cfg.video_out_path) / f"{ckpt_tag}_{cfg.task_suite_name}_results.json"


def _run_episode(
    cfg: GenerateConfig,
    env,
    policy,
    task_description: str,
    initial_state: np.ndarray,
    get_libero_dummy_action,
    rotate_libero_image,
) -> tuple[bool, list[np.ndarray], list[list[float]]]:
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue: deque[np.ndarray] = deque()
    replay_images: list[np.ndarray] = []
    executed_actions: list[list[float]] = []
    dummy_action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    for t in range(max_steps + cfg.num_steps_wait):
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(dummy_action.tolist())
            continue

        primary = rotate_libero_image(obs["agentview_image"])
        wrist = rotate_libero_image(obs["robot0_eye_in_hand_image"])
        replay_images.append(np.concatenate([primary, wrist], axis=1))

        if not action_queue:
            env_actions = policy.predict_env_action_chunk(
                primary,
                wrist,
                task_description,
                obs,
                execute_steps=cfg.num_open_loop_steps,
            )
            action_queue.extend(np.asarray(row, dtype=np.float32) for row in env_actions)

        action = action_queue.popleft() if action_queue else dummy_action
        executed_actions.append(action.tolist())
        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break

    return success, replay_images, executed_actions


def eval_libero(cfg: GenerateConfig) -> float:
    _setup_logging(cfg)
    if not cfg.ckpt_path:
        raise ValueError("--ckpt_path is required for TurboVLA evaluation.")
    if not Path(cfg.ckpt_path).exists():
        raise FileNotFoundError(f"TurboVLA checkpoint not found: {cfg.ckpt_path}")
    if not cfg.dinov3_path:
        raise ValueError("--dinov3_path is required")
    if not cfg.bert_path:
        raise ValueError("--bert_path is required")
    if not Path(cfg.stats_path).is_file():
        raise FileNotFoundError(f"stats file not found: {cfg.stats_path}")
    if cfg.task_suite_name not in LIBERO_SUITES:
        raise ValueError(f"Unknown task suite {cfg.task_suite_name}; choose from {LIBERO_SUITES}")

    if cfg.num_open_loop_steps != cfg.chunk_size:
        logging.warning(
            "num_open_loop_steps=%s differs from chunk_size=%s; this is an ablation, not the default aligned protocol.",
            cfg.num_open_loop_steps,
            cfg.chunk_size,
        )

    (
        TurboVLAPolicy,
        get_libero_dummy_action,
        rotate_libero_image,
        set_seed_everywhere,
    ) = _import_turbovla_adapter()
    set_seed_everywhere(cfg.seed)

    logging.info("Loading TurboVLA policy from %s", cfg.ckpt_path)
    policy = TurboVLAPolicy(
        ckpt_path=cfg.ckpt_path,
        dinov3_path=cfg.dinov3_path,
        bert_path=cfg.bert_path,
        stats_path=cfg.stats_path,
        stats_key=cfg.stats_key,
        normalize_binary_gripper=cfg.normalize_binary_gripper,
        allow_hf_download=cfg.allow_hf_download,
        hidden_dim=cfg.hidden_dim,
        nheads=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        max_text_len=cfg.max_text_len,
        text_padding_length=cfg.text_padding_length,
        vla_feature_enhancer_layers=cfg.vla_feature_enhancer_layers,
        enhancer_inner_dim=cfg.enhancer_inner_dim,
        action_dim=cfg.action_dim,
        chunk_size=cfg.chunk_size,
        state_dim=cfg.state_dim,
        num_state_tokens=cfg.num_state_tokens,
        text_dropout=cfg.text_dropout,
        fusion_dropout=cfg.fusion_dropout,
        fusion_droppath=cfg.fusion_droppath,
        sub_sentence_present=cfg.sub_sentence_present,
        precision=cfg.precision,
        dinov3_output_hidden_states=cfg.dinov3_output_hidden_states,
        dump_embeddings=cfg.dump_embeddings,
        embedding_dump_dir=cfg.embedding_dump_dir,
        embedding_dump_max=cfg.embedding_dump_max,
        embedding_dump_every=cfg.embedding_dump_every,
        embedding_dump_print_summary=cfg.embedding_dump_print_summary,
        embedding_dump_preview_values=cfg.embedding_dump_preview_values,
    )

    if cfg.dry_run_model_load:
        logging.info("dry_run_model_load=True; exiting before LIBERO rollout.")
        return 0.0

    _ensure_libero_import_path(cfg)
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    n_tasks = task_suite.n_tasks if cfg.max_tasks <= 0 else min(cfg.max_tasks, task_suite.n_tasks)
    if cfg.task_ids:
        task_ids = [int(item.strip()) for item in cfg.task_ids.split(",") if item.strip()]
        if not task_ids:
            raise ValueError("--task_ids did not contain any task indices")
        invalid = [task_id for task_id in task_ids if task_id < 0 or task_id >= task_suite.n_tasks]
        if invalid:
            raise ValueError(f"task ids out of range for {cfg.task_suite_name}: {invalid}")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("--task_ids cannot contain duplicates")
    else:
        task_ids = list(range(n_tasks))

    summary = {
        "script": "turbovla_libero_evaluation",
        "ckpt_path": cfg.ckpt_path,
        "task_suite_name": cfg.task_suite_name,
        "requested_task_ids": task_ids,
        "num_trials_per_task": cfg.num_trials_per_task,
        "num_open_loop_steps": cfg.num_open_loop_steps,
        "seed": cfg.seed,
        "precision": cfg.precision,
        "dinov3_output_hidden_states": cfg.dinov3_output_hidden_states,
        "tasks": [],
        "total_episodes": 0,
        "total_successes": 0,
        "overall_success_rate": 0.0,
    }

    total_episodes = 0
    total_successes = 0
    video_root = Path(cfg.video_out_path) / Path(cfg.ckpt_path).stem / cfg.task_suite_name
    video_root.mkdir(parents=True, exist_ok=True)

    for task_id in tqdm.tqdm(task_ids, desc="tasks"):
        # The released 500-episode results evaluated each task in an isolated
        # process. Resetting here preserves that exact seed contract when a
        # single invocation evaluates multiple task IDs.
        set_seed_everywhere(cfg.seed)
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if cfg.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"{cfg.task_suite_name} task {task_id} has only {len(initial_states)} initial states, "
                f"but num_trials_per_task={cfg.num_trials_per_task}"
            )

        env, task_description = _make_libero_env(task, cfg)
        task_successes = 0
        task_episodes = 0
        try:
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"task {task_id}", leave=False):
                success, replay_images, executed_actions = _run_episode(
                    cfg,
                    env,
                    policy,
                    task_description,
                    initial_states[episode_idx],
                    get_libero_dummy_action,
                    rotate_libero_image,
                )
                task_episodes += 1
                total_episodes += 1
                if success:
                    task_successes += 1
                    total_successes += 1

                suffix = "success" if success else "failure"
                logging.info(
                    "task=%d episode=%d success=%s running=%d/%d %.1f%%",
                    task_id,
                    episode_idx,
                    success,
                    total_successes,
                    total_episodes,
                    100.0 * total_successes / max(total_episodes, 1),
                )
                if cfg.save_video:
                    _save_video(video_root / f"task{task_id:02d}_ep{episode_idx:03d}_{suffix}.mp4", replay_images)
        finally:
            env.close()

        task_rate = task_successes / max(task_episodes, 1)
        summary["tasks"].append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_rate,
            }
        )
        logging.info("task=%d success_rate=%.4f (%d/%d)", task_id, task_rate, task_successes, task_episodes)

    final_rate = total_successes / max(total_episodes, 1)
    summary["total_episodes"] = total_episodes
    summary["total_successes"] = total_successes
    summary["overall_success_rate"] = final_rate

    result_path = _result_path(cfg)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Final success rate: %.4f (%d/%d)", final_rate, total_successes, total_episodes)
    logging.info("Saved result json: %s", result_path)
    return final_rate


def main():
    eval_libero(parse_args())


if __name__ == "__main__":
    main()

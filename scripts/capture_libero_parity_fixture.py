from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turbovla.evaluation.replay import capture_libero_observation


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one real deterministic LIBERO observation")
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    obs, instruction, metadata = capture_libero_observation(
        libero_root=args.libero_root, task_suite=args.task_suite, task_id=args.task_id,
        episode=args.episode, step=args.step, seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(args.output / "agentview_image.npy", obs["agentview_image"], allow_pickle=False)
    np.save(args.output / "robot0_eye_in_hand_image.npy", obs["robot0_eye_in_hand_image"], allow_pickle=False)
    for name in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        np.save(args.output / f"{name}.npy", obs[name], allow_pickle=False)
    (args.output / "instruction.txt").write_text(instruction, encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

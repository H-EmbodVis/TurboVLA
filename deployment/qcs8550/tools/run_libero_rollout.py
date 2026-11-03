#!/usr/bin/env python3
"""Run the upstream LIBERO rollout protocol with QCS8550 TurboVLA execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parents[1]
for path in (ROOT, SOURCE_ROOT, SOURCE_ROOT / "third_party/vla_adapter"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qnn_policy import TurboVLAQnnPolicy, rotate_libero_image
from vla_adapter import rollout


def qnn_policy_import():
    from turbovla.evaluation.policy import get_libero_dummy_action, set_seed_everywhere

    return TurboVLAQnnPolicy, get_libero_dummy_action, rotate_libero_image, set_seed_everywhere


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rollout._import_turbovla_adapter = qnn_policy_import
    rollout.main()

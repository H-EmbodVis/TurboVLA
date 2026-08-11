from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class FixtureReader:
    def __init__(self, fixture_dir: Path | str) -> None:
        self.fixture_dir = Path(fixture_dir)
        if not self.fixture_dir.is_dir():
            raise FileNotFoundError(f"fixture directory not found: {self.fixture_dir}")
        self.metadata = json.loads((self.fixture_dir / "metadata.json").read_text(encoding="utf-8"))

    @property
    def instruction(self) -> str:
        return (self.fixture_dir / "instruction.txt").read_text(encoding="utf-8")

    def load_numpy(self, relative: str) -> np.ndarray:
        return np.load(self.fixture_dir / relative, allow_pickle=False)

    def load_tensor(self, relative: str, *, bf16_bits: bool = False) -> torch.Tensor:
        result = torch.from_numpy(self.load_numpy(relative))
        return result.view(torch.bfloat16) if bf16_bits else result

    def path(self, relative: str) -> Path:
        return self.fixture_dir / relative

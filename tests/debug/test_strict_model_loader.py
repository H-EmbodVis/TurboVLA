from pathlib import Path

import pytest

from turbovla.evaluation.model_loader import load_turbovla_for_inference


def test_strict_loader_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_turbovla_for_inference(checkpoint_path=tmp_path / "missing.pth", dinov3_path=str(tmp_path),
                                    bert_path=str(tmp_path), device="cpu", precision="fp32")


def test_strict_loader_rejects_non_strict_mode(tmp_path):
    checkpoint = tmp_path / "empty.pth"
    checkpoint.write_bytes(b"not loaded")
    with pytest.raises(ValueError, match="strict=True"):
        load_turbovla_for_inference(checkpoint_path=checkpoint, dinov3_path=str(tmp_path),
                                    bert_path=str(tmp_path), device="cpu", precision="fp32", strict=False)

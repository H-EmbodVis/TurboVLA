from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch
import pytest

from turbovla.debug import TraceConfig, TraceContext, TensorWriter, TensorRecord


def test_tensor_writer_roundtrip():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        config = TraceConfig(
            enabled=True,
            root_dir=temp_dir,
            save_pt=True,
            save_f32=True,
            save_raw_bf16=True,
            overwrite=True,
        )
        writer = TensorWriter(temp_dir, config)

        # Create test float32 & bf16 tensors
        t_f32 = torch.randn(2, 3, 4, dtype=torch.float32)
        t_bf16 = torch.randn(2, 3, 4, dtype=torch.bfloat16)

        r1 = writer.write(
            semantic_name="test.f32_tensor",
            tensor=t_f32,
            layout="B,C,H",
            operation="test_op",
        )
        r2 = writer.write(
            semantic_name="test.bf16_tensor",
            tensor=t_bf16,
            layout="B,C,H",
            operation="test_op",
        )

        writer.close()

        assert r1 is not None
        assert r2 is not None
        assert (temp_dir / "manifest.jsonl").exists()
        assert (temp_dir / "summary.csv").exists()

        # Check manifest contents
        lines = (temp_dir / "manifest.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

        rec1 = json.loads(lines[0])
        assert rec1["semantic_name"] == "test.f32_tensor"
        assert rec1["shape"] == [2, 3, 4]

        # Verify f32 binary exact roundtrip
        bin_path = temp_dir / rec1["file"]
        read_f32 = np.fromfile(bin_path, dtype="<f4").reshape(2, 3, 4)
        np.testing.assert_allclose(read_f32, t_f32.numpy(), rtol=1e-6)

        # Verify bf16 raw binary exact roundtrip
        rec2 = json.loads(lines[1])
        raw_bf16_path = temp_dir / rec2["raw_bf16_file"]
        read_u16 = np.fromfile(raw_bf16_path, dtype="<u2")
        expected_u16 = t_bf16.view(torch.uint16).numpy().flatten()
        np.testing.assert_equal(read_u16, expected_u16)

    finally:
        shutil.rmtree(temp_dir)

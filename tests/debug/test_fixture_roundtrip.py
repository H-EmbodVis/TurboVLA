from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from turbovla.debug import FixtureWriter


def test_fixture_writer_roundtrip():
    temp_dir = Path(tempfile.mkdtemp())
    fixture_id = "test_fixture_001"
    try:
        writer = FixtureWriter(output_dir=temp_dir, fixture_id=fixture_id, overwrite=True)
        writer.write_instruction("pick up the block")
        
        rgb_img = np.zeros((256, 256, 3), dtype=np.uint8)
        writer.write_image("view_0_rgb_u8", rgb_img)

        pix_val = torch.randn(1, 2, 3, 256, 256, dtype=torch.float32)
        writer.write_tensor("pixel_values_f32", pix_val, dtype_suffix="f32")

        meta = {"version": 1, "fixture_id": fixture_id}
        writer.write_metadata(meta)

        fixture_path = writer.finalize()
        assert fixture_path.exists()
        assert (fixture_path / "metadata.json").exists()
        assert (fixture_path / "instruction.txt").read_text() == "pick up the block"
        assert (fixture_path / "pixel_values_f32.npy").exists()

        # Check overwrite safety
        with pytest.raises(FileExistsError):
            FixtureWriter(output_dir=temp_dir, fixture_id=fixture_id, overwrite=False)

    finally:
        shutil.rmtree(temp_dir)

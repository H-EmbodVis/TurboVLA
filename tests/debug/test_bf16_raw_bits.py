from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch

from turbovla.debug import TraceConfig, TensorWriter


def test_bf16_raw_bits_roundtrip():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        config = TraceConfig(
            enabled=True,
            root_dir=temp_dir,
            save_pt=False,
            save_f32=True,
            save_raw_bf16=True,
            overwrite=True,
        )
        writer = TensorWriter(temp_dir, config)

        t_bf16 = torch.tensor([1.0, -2.5, 3.14159], dtype=torch.bfloat16)

        record = writer.write(semantic_name="bf16.raw_test", tensor=t_bf16, layout="D")
        writer.close()

        assert record is not None
        assert record.raw_bf16_file is not None

        raw_path = temp_dir / record.raw_bf16_file
        assert raw_path.exists()

        # Read back raw uint16 bits
        read_u16 = np.fromfile(raw_path, dtype="<u2")
        expected_u16 = t_bf16.view(torch.uint16).numpy()

        np.testing.assert_equal(read_u16, expected_u16)

    finally:
        shutil.rmtree(temp_dir)

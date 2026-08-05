from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
import torch

from turbovla.debug import TraceConfig, TensorWriter


def test_manifest_schema_and_validation():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        config = TraceConfig(enabled=True, root_dir=temp_dir, save_f32=True, save_pt=True, overwrite=True)
        writer = TensorWriter(temp_dir, config)

        t1 = torch.ones((2, 4), dtype=torch.float32)
        writer.write(semantic_name="test.tensor_1", tensor=t1, layout="B,D", operation="ones")
        writer.close()

        manifest_path = temp_dir / "manifest.jsonl"
        assert manifest_path.exists()

        lines = manifest_path.read_text().strip().split("\n")
        assert len(lines) == 1

        rec = json.loads(lines[0])
        required_fields = [
            "trace_id", "semantic_name", "operation", "shape", "layout",
            "source_dtype", "storage_dtype", "endianness", "contiguous", "numel",
            "file", "min", "max", "mean", "std", "abs_max", "nan_count", "inf_count", "sha256_f32"
        ]
        for field in required_fields:
            assert field in rec, f"Missing field {field} in manifest record"

        assert rec["trace_id"] == 0
        assert rec["numel"] == 8
        assert (temp_dir / rec["file"]).exists()

    finally:
        shutil.rmtree(temp_dir)

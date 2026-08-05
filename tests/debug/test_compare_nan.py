from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import pytest
import torch

from turbovla.debug import TraceConfig, TensorWriter


def test_compare_nan_handling():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        ref_dir = temp_dir / "ref"
        cand_dir = temp_dir / "cand"
        out_dir = temp_dir / "out"

        config_ref = TraceConfig(enabled=True, root_dir=ref_dir, overwrite=True)
        config_cand = TraceConfig(enabled=True, root_dir=cand_dir, overwrite=True)

        writer_ref = TensorWriter(ref_dir, config_ref)
        writer_cand = TensorWriter(cand_dir, config_cand)

        t_ref = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        t_cand = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float32)

        writer_ref.write(semantic_name="nan.tensor", tensor=t_ref, layout="D")
        writer_cand.write(semantic_name="nan.tensor", tensor=t_cand, layout="D")

        writer_ref.close()
        writer_cand.close()

        cmd = [
            "/home/linh/anaconda3/envs/turbovla-libero/bin/python",
            "scripts/compare_tensor_traces.py",
            "--reference", str(ref_dir),
            "--candidate", str(cand_dir),
            "--output", str(out_dir),
        ]
        subprocess.run(cmd, capture_output=True, text=True)

        assert (out_dir / "compare.json").exists()
        compare_data = json.loads((out_dir / "compare.json").read_text())

        status_map = {item["semantic_name"]: item["status"] for item in compare_data}
        assert status_map["nan.tensor"] == "FAIL_NAN"

        # Also test fail_on_nan config
        config_fail_nan = TraceConfig(enabled=True, root_dir=temp_dir / "fail_nan", fail_on_nan=True, overwrite=True)
        writer_fail = TensorWriter(temp_dir / "fail_nan", config_fail_nan)
        with pytest.raises(ValueError, match="NaN detected"):
            writer_fail.write(semantic_name="nan.tensor", tensor=t_cand, layout="D")

    finally:
        shutil.rmtree(temp_dir)

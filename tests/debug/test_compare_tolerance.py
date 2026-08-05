from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch

from turbovla.debug import TraceConfig, TensorWriter


def test_compare_tolerance():
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
        t_pass_tol = torch.tensor([1.005, 2.005, 3.005], dtype=torch.float32)  # Diff within 0.02
        t_fail_tol = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)      # Large diff

        writer_ref.write(semantic_name="tensor.pass_tol", tensor=t_ref, layout="D")
        writer_ref.write(semantic_name="tensor.fail_tol", tensor=t_ref, layout="D")

        writer_cand.write(semantic_name="tensor.pass_tol", tensor=t_pass_tol, layout="D")
        writer_cand.write(semantic_name="tensor.fail_tol", tensor=t_fail_tol, layout="D")

        writer_ref.close()
        writer_cand.close()

        cmd = [
            "/home/linh/anaconda3/envs/turbovla-libero/bin/python",
            "scripts/compare_tensor_traces.py",
            "--reference", str(ref_dir),
            "--candidate", str(cand_dir),
            "--output", str(out_dir),
            "--atol", "0.02",
            "--rtol", "0.02",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)

        assert (out_dir / "compare.json").exists()
        compare_data = json.loads((out_dir / "compare.json").read_text())

        status_map = {item["semantic_name"]: item["status"] for item in compare_data}
        assert status_map["tensor.pass_tol"] == "PASS_TOLERANCE"
        assert status_map["tensor.fail_tol"] == "FAIL_TOLERANCE"

    finally:
        shutil.rmtree(temp_dir)

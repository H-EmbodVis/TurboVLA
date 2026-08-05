from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch

from turbovla.debug import TraceConfig, TensorWriter


def test_compare_layout_mismatch():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        ref_dir = temp_dir / "ref"
        cand_dir = temp_dir / "cand"
        out_dir = temp_dir / "out"

        config_ref = TraceConfig(enabled=True, root_dir=ref_dir, overwrite=True)
        config_cand = TraceConfig(enabled=True, root_dir=cand_dir, overwrite=True)

        writer_ref = TensorWriter(ref_dir, config_ref)
        writer_cand = TensorWriter(cand_dir, config_cand)

        t_ref = torch.ones((2, 3), dtype=torch.float32)
        t_cand = torch.ones((2, 3), dtype=torch.float32)

        writer_ref.write(semantic_name="layout.tensor", tensor=t_ref, layout="B,N")
        writer_cand.write(semantic_name="layout.tensor", tensor=t_cand, layout="B,D")

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

        assert (out_dir / "layout_mismatches.txt").exists()
        layout_text = (out_dir / "layout_mismatches.txt").read_text()
        assert "layout.tensor" in layout_text

    finally:
        shutil.rmtree(temp_dir)

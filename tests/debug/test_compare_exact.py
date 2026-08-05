from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import torch

from turbovla.debug import TraceConfig, TensorWriter


def test_compare_tensors():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        ref_dir = temp_dir / "ref"
        cand_dir = temp_dir / "cand"
        out_dir = temp_dir / "out"

        config_ref = TraceConfig(enabled=True, root_dir=ref_dir, overwrite=True)
        config_cand = TraceConfig(enabled=True, root_dir=cand_dir, overwrite=True)

        writer_ref = TensorWriter(ref_dir, config_ref)
        writer_cand = TensorWriter(cand_dir, config_cand)

        t1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        t2_ref = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        t2_cand = torch.tensor([1.0, 2.05, 3.0], dtype=torch.float32)  # Slight diff

        writer_ref.write(semantic_name="layer1.output", tensor=t1, layout="D")
        writer_ref.write(semantic_name="layer2.output", tensor=t2_ref, layout="D")

        writer_cand.write(semantic_name="layer1.output", tensor=t1, layout="D")
        writer_cand.write(semantic_name="layer2.output", tensor=t2_cand, layout="D")

        writer_ref.close()
        writer_cand.close()

        import subprocess
        cmd = [
            "/home/linh/anaconda3/bin/python",
            "scripts/compare_tensor_traces.py",
            "--reference", str(ref_dir),
            "--candidate", str(cand_dir),
            "--output", str(out_dir),
            "--atol", "0.01",
            "--rtol", "0.01",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        assert (out_dir / "compare.csv").exists()
        assert (out_dir / "compare.json").exists()
        assert (out_dir / "first_divergence.txt").exists()

        div_text = (out_dir / "first_divergence.txt").read_text()
        assert "layer2.output" in div_text

    finally:
        shutil.rmtree(temp_dir)

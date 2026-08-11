import subprocess
import sys

import torch

from turbovla.debug import TensorWriter, TraceConfig


def test_allclose_accepts_either_atol_or_relative_term(tmp_path):
    for name, value in (("ref", 100.0), ("cand", 101.5)):
        writer = TensorWriter(tmp_path / name, TraceConfig(enabled=True, overwrite=True))
        writer.write(semantic_name="value", tensor=torch.tensor([value]), layout="D")
        writer.close()
    result = subprocess.run([sys.executable, "scripts/compare_tensor_traces.py", "--reference", str(tmp_path / "ref"),
                             "--candidate", str(tmp_path / "cand"), "--output", str(tmp_path / "out"),
                             "--atol", "0", "--rtol", "0.02"])
    assert result.returncode == 0

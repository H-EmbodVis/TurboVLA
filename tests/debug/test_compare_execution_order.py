import json
import subprocess
import sys

import torch

from turbovla.debug import TensorWriter, TraceConfig


def test_first_divergence_uses_reference_execution_order(tmp_path):
    for directory, second, tenth in ((tmp_path / "ref", 2.0, 10.0), (tmp_path / "cand", 3.0, 11.0)):
        writer = TensorWriter(directory, TraceConfig(enabled=True, overwrite=True))
        writer.write(semantic_name="z_second", tensor=torch.tensor([second]), layout="D")
        writer.write(semantic_name="a_tenth", tensor=torch.tensor([tenth]), layout="D")
        writer.close()
    subprocess.run([sys.executable, "scripts/compare_tensor_traces.py", "--reference", str(tmp_path / "ref"),
                    "--candidate", str(tmp_path / "cand"), "--output", str(tmp_path / "out")], check=False)
    divergence = json.loads((tmp_path / "out/first_divergence.txt").read_text())
    assert divergence["semantic_name"] == "z_second"

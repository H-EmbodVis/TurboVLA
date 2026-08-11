import json
import subprocess
import sys

import torch

from turbovla.debug import TensorWriter, TraceConfig


def test_deliberate_perturbation_reports_coordinate(tmp_path):
    for name, value in (("ref", [1.0, 2.0, 3.0]), ("cand", [1.0, 9.0, 3.0])):
        writer = TensorWriter(tmp_path / name, TraceConfig(enabled=True, overwrite=True))
        writer.write(semantic_name="tiny", tensor=torch.tensor(value), layout="D")
        writer.close()
    subprocess.run([sys.executable, "scripts/compare_tensor_traces.py", "--reference", str(tmp_path / "ref"),
                    "--candidate", str(tmp_path / "cand"), "--output", str(tmp_path / "out")], check=False)
    result = json.loads((tmp_path / "out/first_divergence.txt").read_text())
    assert result["first_bad_flat_index"] == 1
    assert result["first_bad_coordinate"] == [1]

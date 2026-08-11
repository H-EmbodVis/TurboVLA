import json
import numpy as np
import torch

from turbovla.debug import TraceConfig, TraceContext


def test_bf16_has_canonical_and_native_files(tmp_path):
    value = torch.tensor([1.5, -2.0], dtype=torch.bfloat16)
    tracer = TraceContext(TraceConfig(enabled=True, root_dir=tmp_path, overwrite=True))
    tracer.begin(0)
    tracer.tensor("value", value)
    trace = tracer.finish()
    record = json.loads((trace / "manifest.jsonl").read_text())
    np.testing.assert_array_equal(np.fromfile(trace / record["native_file"], dtype="<u2"), value.view(torch.uint16).numpy())
    np.testing.assert_allclose(np.fromfile(trace / record["file"], dtype="<f4"), value.float().numpy())

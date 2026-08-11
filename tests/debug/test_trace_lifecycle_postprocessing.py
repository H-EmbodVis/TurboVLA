import torch

from turbovla.debug import TraceConfig, TraceContext


def test_lifecycle_can_include_postprocessing(tmp_path):
    tracer = TraceContext(TraceConfig(enabled=True, root_dir=tmp_path, overwrite=True))
    tracer.begin(0)
    tracer.tensor("action.normalized", torch.zeros(1, 1, 7))
    assert tracer.active
    tracer.tensor("action.denormalized", torch.zeros(1, 1, 7))
    trace = tracer.finish()
    assert "action.denormalized" in (trace / "manifest.jsonl").read_text()

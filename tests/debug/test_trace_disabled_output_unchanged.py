import torch

from turbovla.debug import TraceConfig, TraceContext
from turbovla.models.turbovla import VisionProjection


def test_disabled_trace_does_not_create_output_or_change_value(tmp_path):
    module = VisionProjection(4, 4, 8, 0.0).eval()
    value = torch.randn(1, 2, 4)
    tracer = TraceContext(TraceConfig(enabled=False, root_dir=tmp_path))
    assert not tracer.begin(0)
    torch.testing.assert_close(module(value), module(value, tracer=tracer), rtol=0, atol=0)
    assert not list(tmp_path.rglob("*"))

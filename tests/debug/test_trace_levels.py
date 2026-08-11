import torch

from turbovla.debug import TraceConfig, TraceContext


def test_trace_levels_filter_before_write(tmp_path):
    tracer = TraceContext(TraceConfig(enabled=True, root_dir=tmp_path, level="layer", overwrite=True))
    assert tracer.begin(0)
    tracer.tensor("boundary", torch.ones(1), required_level="boundary")
    tracer.tensor("layer", torch.ones(1), required_level="layer")
    tracer.tensor("op", torch.ones(1), required_level="op")
    assert [record.semantic_name for record in tracer.records] == ["boundary", "layer"]
    tracer.finish()

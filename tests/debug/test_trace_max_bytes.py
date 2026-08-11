import pytest
import torch

from turbovla.debug import TraceByteLimitExceeded, TraceConfig, TraceContext


def test_byte_limit_fails_instead_of_truncating(tmp_path):
    tracer = TraceContext(TraceConfig(enabled=True, root_dir=tmp_path, max_bytes=3, overwrite=True))
    tracer.begin(0)
    with pytest.raises(TraceByteLimitExceeded):
        tracer.tensor("too_large", torch.ones(1))

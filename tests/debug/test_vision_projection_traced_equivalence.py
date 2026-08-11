import torch

from turbovla.models.turbovla import VisionProjection


def test_vision_projection_output_unchanged_without_active_trace():
    torch.manual_seed(1)
    module = VisionProjection(8, 4, 16, 0.0).eval()
    value = torch.randn(2, 3, 5, 8)
    torch.testing.assert_close(module(value), module(value, tracer=None), rtol=0, atol=0)

import torch

from turbovla.models.action_head import StateProjection
from turbovla.models.configuration import ActionHeadConfig


def test_state_projection_output_unchanged_without_active_trace():
    torch.manual_seed(4)
    module = StateProjection(ActionHeadConfig(state_dim=8, num_state_tokens=2, state_hidden_dim=16, dropout=0.0), 8).eval()
    value = torch.randn(2, 8)
    torch.testing.assert_close(module(value), module(value, tracer=None), rtol=0, atol=0)

import torch
import torch.nn.functional as F

from turbovla.models.components.utils import MLP


def test_action_mlp_decomposition_matches_module():
    torch.manual_seed(6)
    module = MLP(8, 16, 3, 3).eval()
    value = torch.randn(2, 4, 8)
    manual = value
    for index, layer in enumerate(module.layers):
        manual = layer(manual)
        if index < len(module.layers) - 1:
            manual = F.relu(manual)
    torch.testing.assert_close(manual, module(value), rtol=0, atol=0)

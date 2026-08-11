import torch

from turbovla.models.components.fusion import BiAttentionBlock


def test_fusion_output_unchanged_without_active_trace():
    torch.manual_seed(2)
    module = BiAttentionBlock(8, 8, 16, 2, dropout=0.0, drop_path=0.0).eval()
    visual, text = torch.randn(1, 5, 8), torch.randn(1, 3, 8)
    expected = module(visual, text)
    actual = module(visual, text, tracer=None, prefix="interaction.layer_00")
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

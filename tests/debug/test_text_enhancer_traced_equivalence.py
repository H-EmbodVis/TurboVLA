import torch

from turbovla.models.components.transformer import TransformerEncoderLayer


def test_text_enhancer_none_mask_guard_preserves_output():
    torch.manual_seed(3)
    module = TransformerEncoderLayer(8, 2, 16, dropout=0.0).eval()
    value = torch.randn(4, 1, 8)
    expected = module(value, src_mask=None)
    actual = module(value, src_mask=None, tracer=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

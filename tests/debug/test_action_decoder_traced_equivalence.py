import torch

from turbovla.models.action_head import ACTDecoder
from turbovla.models.configuration import ActionHeadConfig


def test_manual_decoder_loop_matches_stock_transformer_decoder():
    torch.manual_seed(5)
    module = ACTDecoder(ActionHeadConfig(horizon=3, num_layers=2, action_dim=2, mlp_hidden_dim=8, dropout=0.0), 8, 2, 16).eval()
    memory = torch.randn(1, 7, 8)
    queries = module.action_queries.weight.unsqueeze(0)
    stock = module.decoder(queries, memory)
    manual = queries
    for layer in module.decoder.layers:
        manual = layer(manual, memory)
    if module.decoder.norm is not None:
        manual = module.decoder.norm(manual)
    torch.testing.assert_close(manual, stock, rtol=0, atol=0)

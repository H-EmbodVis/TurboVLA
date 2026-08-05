from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from .components.utils import MLP
from .configuration import ActionHeadConfig


class StateProjection(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(config.num_state_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_hidden_dim, self.num_tokens * self.hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state: torch.Tensor, tracer: Any | None = None) -> torch.Tensor:
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2:
            raise ValueError(f"state must be [B,D] or [B,T,D], got {tuple(state.shape)}")
        if tracer is not None and tracer.active:
            tracer.tensor("state_projection.input_normalized", state, layout="B,D", operation="input")
        projected = self.net(state)
        tokens = projected.view(state.shape[0], self.num_tokens, self.hidden_dim)
        if tracer is not None and tracer.active:
            tracer.tensor("state_projection.reshape", tokens, layout="B,S,D", operation="reshape")
            tracer.tensor("state_projection.position_embedding", self.position, layout="1,S,D", operation="parameter")
        output = self.output_norm(tokens + self.position.to(device=tokens.device, dtype=tokens.dtype))
        if tracer is not None and tracer.active:
            tracer.tensor("state_projection.output", output, layout="B,S,D", operation="layer_norm")
        return output


class ACTDecoder(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.horizon = int(config.horizon)
        self.action_queries = nn.Embedding(self.horizon, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.action_projection = MLP(hidden_dim, config.mlp_hidden_dim, config.action_dim, 3)

    def forward(
        self,
        memory: torch.Tensor,
        dump: Callable[[str, torch.Tensor], None] | None = None,
        tracer: Any | None = None,
    ) -> torch.Tensor:
        queries = self.action_queries.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        if dump is not None:
            dump("action.query_embeddings", queries)
        if tracer is not None and tracer.active:
            tracer.tensor("action.queries", queries, layout="B,T,D", operation="embedding")
        hidden = self.decoder(tgt=queries, memory=memory)
        if dump is not None:
            dump("action.decoder_hidden", hidden)
        if tracer is not None and tracer.active:
            tracer.tensor("action.decoder_hidden", hidden, layout="B,T,D", operation="transformer_decoder")
        raw_actions = self.action_projection(hidden)
        if dump is not None:
            dump("action.before_tanh", raw_actions)
        if tracer is not None and tracer.active:
            tracer.tensor("action.before_tanh", raw_actions, layout="B,T,A", operation="mlp")
        actions = torch.tanh(raw_actions)
        if dump is not None:
            dump("action.output", actions)
        return actions


class TurboVLAActionHead(nn.Module):
    def __init__(self, config: ActionHeadConfig, hidden_dim: int, nheads: int, dim_feedforward: int) -> None:
        super().__init__()
        self.state_projection = StateProjection(config, hidden_dim)
        self.decoder = ACTDecoder(config, hidden_dim, nheads, dim_feedforward)

    def forward(
        self,
        vision_language_tokens: torch.Tensor,
        state: torch.Tensor,
        dump: Callable[[str, torch.Tensor], None] | None = None,
        tracer: Any | None = None,
    ) -> torch.Tensor:
        state = state.to(device=vision_language_tokens.device, dtype=vision_language_tokens.dtype)
        if dump is not None:
            dump("action.state_input", state)
        state_tokens = self.state_projection(state, tracer=tracer)
        memory = torch.cat([vision_language_tokens, state_tokens], dim=1)
        if dump is not None:
            dump("action.state_tokens", state_tokens)
            dump("action.decoder_memory", memory)
        if tracer is not None and tracer.active:
            tracer.tensor("action.memory.condition", vision_language_tokens, layout="B,VxN+N,D", operation="split")
            tracer.tensor("action.memory.state_tokens", state_tokens, layout="B,S,D", operation="state_projection")
            tracer.tensor("action.memory.concatenated", memory, layout="B,VxN+N+S,D", operation="concat")
        return self.decoder(memory, dump=dump, tracer=tracer)

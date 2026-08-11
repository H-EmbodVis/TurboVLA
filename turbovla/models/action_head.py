from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from .components.utils import MLP
from .configuration import ActionHeadConfig
from ..debug.model_trace import emit, tensor_axis, trace_layer_norm, trace_linear, trace_softmax


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
        trace_layer_norm(tracer, "state_projection.norm", state, self.net[0], layout="B,D")
        normalized = self.net[0](state)
        linear_1 = trace_linear(tracer, "state_projection.linear_1", normalized, self.net[1], layout="B,F")
        gelu = self.net[2](linear_1)
        emit(tracer, "state_projection.gelu.input", linear_1, "B,F", "gelu", "exhaustive")
        emit(tracer, "state_projection.gelu.output", gelu, "B,F", "gelu", "op")
        trace_linear(tracer, "state_projection.linear_2", gelu, self.net[4], layout="B,SxD")
        projected = self.net(state)
        emit(tracer, "state_projection.before_reshape", projected, "B,SxD", "identity", "exhaustive")
        tokens = projected.view(state.shape[0], self.num_tokens, self.hidden_dim)
        if tracer is not None and tracer.active:
            tracer.tensor("state_projection.after_reshape", tokens, layout="B,S,D", operation="reshape", required_level="op")
            tracer.tensor("state_projection.position_embedding", self.position, layout="1,S,D", operation="parameter")
            tracer.tensor("state_projection.position.before_add", tokens, layout="B,S,D", operation="identity", required_level="exhaustive")
        positioned = tokens + self.position.to(device=tokens.device, dtype=tokens.dtype)
        emit(tracer, "state_projection.position.after_add", positioned, "B,S,D", "add", "op")
        trace_layer_norm(tracer, "state_projection.output_norm", positioned, self.output_norm, layout="B,S,D")
        output = self.output_norm(positioned)
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

    @staticmethod
    def _trace_attention(tracer, prefix, query, key_value, module, *, mask=None):
        dim = query.shape[-1]
        weight, bias = module.in_proj_weight, module.in_proj_bias
        q = F.linear(query, weight[:dim], None if bias is None else bias[:dim])
        k = F.linear(key_value, weight[dim:2 * dim], None if bias is None else bias[dim:2 * dim])
        v = F.linear(key_value, weight[2 * dim:], None if bias is None else bias[2 * dim:])
        heads, head_dim = module.num_heads, dim // module.num_heads
        qh = q.view(q.shape[0], q.shape[1], heads, head_dim).transpose(1, 2)
        kh = k.view(k.shape[0], k.shape[1], heads, head_dim).transpose(1, 2)
        vh = v.view(v.shape[0], v.shape[1], heads, head_dim).transpose(1, 2)
        for name, linear, split in (("q", q, qh), ("k", k, kh), ("v", v, vh)):
            emit(tracer, f"{prefix}.{name}_linear", linear, "B,N,D", "linear", "op")
            emit(tracer, f"{prefix}.{name}_heads", split, "B,H,N,Dh", "reshape_transpose", "exhaustive")
        logits = torch.matmul(qh, kh.transpose(-1, -2)) / (head_dim ** 0.5)
        if mask is None:
            mask = torch.zeros_like(logits, dtype=torch.bool)
        masked = logits.masked_fill(mask, float("-inf")) if mask.dtype == torch.bool else logits + mask
        emit(tracer, f"{prefix}.logits", logits, "B,H,Q,K", "matmul_scale", "op")
        emit(tracer, f"{prefix}.mask", mask, "B,H,Q,K", "mask", "exhaustive")
        emit(tracer, f"{prefix}.masked_logits", masked, "B,H,Q,K", "masked_fill", "op")
        probs = trace_softmax(tracer, f"{prefix}.softmax", masked, layout="B,H,Q,K", output_dtype=query.dtype)
        context_heads = torch.matmul(probs, vh)
        context = context_heads.transpose(1, 2).reshape(query.shape[0], query.shape[1], dim)
        output = module.out_proj(context)
        emit(tracer, f"{prefix}.context_heads", context_heads, "B,H,Q,Dh", "matmul", "op")
        emit(tracer, f"{prefix}.context_merged", context, "B,Q,D", "reshape", "op")
        emit(tracer, f"{prefix}.output_projection", output, "B,Q,D", "linear", "op")
        return output

    def _trace_decoder_layer(self, tracer, index, layer, value, memory):
        prefix = f"action.decoder.layer_{index:02d}"
        emit(tracer, f"{prefix}.input", value, "B,T,D", "layer_input", "layer")
        trace_layer_norm(tracer, f"{prefix}.self_attn.norm", value, layer.norm1, layout="B,T,D")
        emit(tracer, f"{prefix}.self_attn.norm_input", value, "B,T,D", "identity", "op")
        normalized = layer.norm1(value)
        self_delta = self._trace_attention(tracer, f"{prefix}.self_attn", normalized, normalized, layer.self_attn)
        for suffix, tensor in (("left", value), ("right", self_delta), ("sum", value + self_delta)):
            emit(tracer, f"{prefix}.self_attn.residual.{suffix}", tensor, "B,T,D", "add" if suffix == "sum" else "identity", "op")
        after_self = value + self_delta
        trace_layer_norm(tracer, f"{prefix}.cross_attn.norm", after_self, layer.norm2, layout="B,T,D")
        emit(tracer, f"{prefix}.cross_attn.norm_input", after_self, "B,T,D", "identity", "op")
        cross_delta = self._trace_attention(tracer, f"{prefix}.cross_attn", layer.norm2(after_self), memory, layer.multihead_attn)
        for suffix, tensor in (("left", after_self), ("right", cross_delta), ("sum", after_self + cross_delta)):
            emit(tracer, f"{prefix}.cross_attn.residual.{suffix}", tensor, "B,T,D", "add" if suffix == "sum" else "identity", "op")
        after_cross = after_self + cross_delta
        trace_layer_norm(tracer, f"{prefix}.ffn.norm", after_cross, layer.norm3, layout="B,T,D")
        emit(tracer, f"{prefix}.ffn.norm_input", after_cross, "B,T,D", "identity", "op")
        normed = layer.norm3(after_cross)
        linear_1 = trace_linear(tracer, f"{prefix}.ffn.linear_1", normed, layer.linear1, layout="B,T,F")
        activated = layer.activation(linear_1)
        emit(tracer, f"{prefix}.ffn.activation.input", linear_1, "B,T,F", "activation", "exhaustive")
        emit(tracer, f"{prefix}.ffn.activation.output", activated, "B,T,F", "activation", "op")
        linear_2 = trace_linear(tracer, f"{prefix}.ffn.linear_2", activated, layer.linear2, layout="B,T,D")
        for suffix, tensor in (("left", after_cross), ("right", linear_2), ("sum", after_cross + linear_2)):
            emit(tracer, f"{prefix}.ffn.residual.{suffix}", tensor, "B,T,D", "add" if suffix == "sum" else "identity", "op")

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
            tracer.tensor("action.queries.weight", self.action_queries.weight, layout="T,D", operation="parameter", required_level="op")
            tracer.tensor("action.queries.expanded", queries, layout="B,T,D", operation="expand", required_level="op")
        hidden = queries
        for index, layer in enumerate(self.decoder.layers):
            self._trace_decoder_layer(tracer, index, layer, hidden, memory)
            hidden = layer(hidden, memory)
            emit(tracer, f"action.decoder.layer_{index:02d}.output", hidden, "B,T,D", "layer_output", "layer")
        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        if dump is not None:
            dump("action.decoder_hidden", hidden)
        emit(tracer, "action.mlp.input", hidden, "B,T,D", "input", "layer")
        mlp_value = hidden
        for index, layer in enumerate(self.action_projection.layers):
            linear = trace_linear(tracer, f"action.mlp.linear_{index}", mlp_value, layer, layout="B,T,D")
            if index < len(self.action_projection.layers) - 1:
                emit(tracer, f"action.mlp.relu_{index}.input", linear, "B,T,D", "relu", "exhaustive")
                mlp_value = F.relu(linear)
                emit(tracer, f"action.mlp.relu_{index}.output", mlp_value, "B,T,D", "relu", "op")
            else:
                mlp_value = linear
        raw_actions = self.action_projection(hidden)
        if dump is not None:
            dump("action.before_tanh", raw_actions)
        if tracer is not None and tracer.active:
            tracer.tensor("action.before_tanh", raw_actions, layout="B,T,A", operation="mlp")
        actions = torch.tanh(raw_actions)
        emit(tracer, "action.normalized", actions, "B,T,A", "tanh", "boundary")
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
            tracer.tensor("action.memory.concat_axis", tensor_axis(1, memory.device), layout="", operation="constant", required_level="exhaustive")
            tracer.tensor("action.memory.concatenated", memory, layout="B,VxN+N+S,D", operation="concat")
        return self.decoder(memory, dump=dump, tracer=tracer)

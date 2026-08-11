# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# Modified for TurboVLA.
# ------------------------------------------------------------------------
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .utils import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    sigmoid_focal_loss,
)
from ...debug.model_trace import emit, trace_layer_norm, trace_linear, trace_softmax


class TextTransformer(nn.Module):
    def __init__(self, num_layers, d_model=256, nheads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.norm = None

        single_encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.layers = _get_clones(single_encoder_layer, num_layers)

    def forward(self, memory_text: torch.Tensor, text_attention_mask: torch.Tensor):
        """

        Args:
            text_attention_mask: bs, num_token
            memory_text: bs, num_token, d_model

        Raises:
            RuntimeError: _description_

        Returns:
            output: bs, num_token, d_model
        """

        output = memory_text.transpose(0, 1)

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=text_attention_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output.transpose(0, 1)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        tracer=None,
        prefix: str = "interaction.layer_00.text_enhancer",
    ):
        # repeat attn mask
        emit(tracer, f"{prefix}.input", src.transpose(0, 1), "B,N,D", "layer_input", "layer")
        if src_mask is not None:
            emit(tracer, f"{prefix}.mask.original", src_mask, "B,N,N", "mask", "op")
        if src_mask is not None and src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            # bs, num_q, num_k
            src_mask = src_mask.repeat(self.nhead, 1, 1)
        trace_mask = src_mask
        if trace_mask is None:
            trace_mask = torch.zeros((src.shape[1] * self.nhead, src.shape[0], src.shape[0]), dtype=torch.bool, device=src.device)
        emit(tracer, f"{prefix}.mask.repeated", trace_mask, "BH,N,N", "repeat", "op")

        q = k = self.with_pos_embed(src, pos)

        # Trace nn.MultiheadAttention's packed projection without changing its output path.
        d_model = src.shape[-1]
        packed_w, packed_b = self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        projected = F.linear(src, packed_w, packed_b)
        q_linear, k_linear, v_linear = projected.chunk(3, dim=-1)
        batch, length, head_dim = src.shape[1], src.shape[0], d_model // self.nhead
        q_heads = q_linear.transpose(0, 1).view(batch, length, self.nhead, head_dim).transpose(1, 2)
        k_heads = k_linear.transpose(0, 1).view(batch, length, self.nhead, head_dim).transpose(1, 2)
        v_heads = v_linear.transpose(0, 1).view(batch, length, self.nhead, head_dim).transpose(1, 2)
        for name, value, heads in (("q", q_linear, q_heads), ("k", k_linear, k_heads), ("v", v_linear, v_heads)):
            emit(tracer, f"{prefix}.attn.{name}_linear", value.transpose(0, 1), "B,N,D", "linear", "op")
            emit(tracer, f"{prefix}.attn.{name}_heads", heads, "B,H,N,Dh", "reshape_transpose", "exhaustive")
        logits = torch.matmul(q_heads, k_heads.transpose(-1, -2)) / (head_dim ** 0.5)
        mask_heads = trace_mask.view(batch, self.nhead, length, length)
        masked_logits = logits.masked_fill(mask_heads, float("-inf")) if mask_heads.dtype == torch.bool else logits + mask_heads
        emit(tracer, f"{prefix}.attn.logits", logits, "B,H,N,N", "matmul_scale", "op")
        emit(tracer, f"{prefix}.attn.masked_logits", masked_logits, "B,H,N,N", "masked_fill", "op")
        probs = trace_softmax(tracer, f"{prefix}.attn.softmax", masked_logits, layout="B,H,N,N", output_dtype=src.dtype)
        context_heads = torch.matmul(probs, v_heads)
        context = context_heads.transpose(1, 2).reshape(batch, length, d_model)
        emit(tracer, f"{prefix}.attn.context_heads", context_heads, "B,H,N,Dh", "matmul", "op")
        emit(tracer, f"{prefix}.attn.context_merged", context, "B,N,D", "reshape", "op")
        projected_out = F.linear(context, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias)
        emit(tracer, f"{prefix}.attn.output_projection", projected_out, "B,N,D", "linear", "op")

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        residual_left, residual_right = src, self.dropout1(src2)
        emit(tracer, f"{prefix}.residual_1.left", residual_left.transpose(0, 1), "B,N,D", "identity", "exhaustive")
        emit(tracer, f"{prefix}.residual_1.right", residual_right.transpose(0, 1), "B,N,D", "dropout", "exhaustive")
        src = residual_left + residual_right
        emit(tracer, f"{prefix}.residual_1.sum", src.transpose(0, 1), "B,N,D", "add", "op")
        trace_layer_norm(tracer, f"{prefix}.norm_1", src.transpose(0, 1), self.norm1)
        src = self.norm1(src)
        linear_1 = trace_linear(tracer, f"{prefix}.ffn.linear_1", src.transpose(0, 1), self.linear1)
        activation = self.activation(linear_1)
        emit(tracer, f"{prefix}.ffn.activation.input", linear_1, "B,N,F", "activation", "exhaustive")
        emit(tracer, f"{prefix}.ffn.activation.output", activation, "B,N,F", "activation", "op")
        trace_linear(tracer, f"{prefix}.ffn.linear_2", activation, self.linear2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        residual_left, residual_right = src, self.dropout2(src2)
        emit(tracer, f"{prefix}.residual_2.left", residual_left.transpose(0, 1), "B,N,D", "identity", "exhaustive")
        emit(tracer, f"{prefix}.residual_2.right", residual_right.transpose(0, 1), "B,N,D", "dropout", "exhaustive")
        src = residual_left + residual_right
        emit(tracer, f"{prefix}.residual_2.sum", src.transpose(0, 1), "B,N,D", "add", "op")
        trace_layer_norm(tracer, f"{prefix}.norm_2", src.transpose(0, 1), self.norm2)
        src = self.norm2(src)
        emit(tracer, f"{prefix}.output", src.transpose(0, 1), "B,N,D", "layer_output", "layer")
        return src

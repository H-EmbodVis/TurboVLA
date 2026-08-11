# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# Modified for TurboVLA.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from ...debug.model_trace import emit, trace_layer_norm, trace_linear, trace_softmax


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None, attention_backend="manual"):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout
        self.attention_backend = attention_backend

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    @staticmethod
    def _padding_mask_to_sdpa(mask, dtype, target_len):
        if mask is None:
            return None
        additive_mask = torch.zeros(
            (mask.shape[0], 1, target_len, mask.shape[1]),
            device=mask.device,
            dtype=dtype,
        )
        return additive_mask.masked_fill(mask[:, None, None, :], float("-inf"))

    def _forward_sdpa(self, v, l, attention_mask_v=None, attention_mask_l=None):
        batch_size, target_len, _ = v.size()
        source_len = l.size(1)
        query_v = self._shape(self.v_proj(v), target_len, batch_size)
        key_l = self._shape(self.l_proj(l), source_len, batch_size)
        value_v = self._shape(self.values_v_proj(v), target_len, batch_size)
        value_l = self._shape(self.values_l_proj(l), source_len, batch_size)
        dropout_p = self.dropout if self.training else 0.0
        output_v = F.scaled_dot_product_attention(
            query_v,
            key_l,
            value_l,
            attn_mask=self._padding_mask_to_sdpa(attention_mask_l, query_v.dtype, target_len),
            dropout_p=dropout_p,
        )
        output_l = F.scaled_dot_product_attention(
            key_l,
            query_v,
            value_v,
            attn_mask=self._padding_mask_to_sdpa(attention_mask_v, key_l.dtype, source_len),
            dropout_p=dropout_p,
        )
        output_v = output_v.transpose(1, 2).reshape(batch_size, target_len, self.embed_dim)
        output_l = output_l.transpose(1, 2).reshape(batch_size, source_len, self.embed_dim)
        return self.out_v_proj(output_v), self.out_l_proj(output_l)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None, tracer=None, prefix="interaction.layer_00"):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        if self.attention_backend == "sdpa" and hasattr(F, "scaled_dot_product_attention"):
            return self._forward_sdpa(v, l, attention_mask_v, attention_mask_l)

        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size()

        q_unscaled = self.v_proj(v)
        k_linear = self.l_proj(l)
        vv_linear = self.values_v_proj(v)
        vl_linear = self.values_l_proj(l)
        trace_linear(tracer, f"{prefix}.cross.q_visual", v, self.v_proj)
        trace_linear(tracer, f"{prefix}.cross.k_text", l, self.l_proj)
        trace_linear(tracer, f"{prefix}.cross.v_visual", v, self.values_v_proj)
        trace_linear(tracer, f"{prefix}.cross.v_text", l, self.values_l_proj)
        query_states = q_unscaled * self.scale
        key_states = self._shape(k_linear, -1, bsz)
        value_v_states = self._shape(vv_linear, -1, bsz)
        value_l_states = self._shape(vl_linear, -1, bsz)
        for name, linear, heads in (("q_visual", q_unscaled, self._shape(q_unscaled, tgt_len, bsz)),
                                    ("k_text", k_linear, key_states), ("v_visual", vv_linear, value_v_states),
                                    ("v_text", vl_linear, value_l_states)):
            emit(tracer, f"{prefix}.cross.{name}.reshape", linear.view(bsz, -1, self.num_heads, self.head_dim), "B,N,H,Dh", "reshape", "exhaustive")
            emit(tracer, f"{prefix}.cross.{name}.transpose", heads, "B,H,N,Dh", "transpose", "exhaustive")

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt
        emit(tracer, f"{prefix}.cross.visual_to_text.qk_matmul", attn_weights, "BH,V,L", "bmm", "op")
        emit(tracer, f"{prefix}.cross.visual_to_text.logits_raw", attn_weights, "BH,V,L", "identity", "exhaustive")

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            global_max = attn_weights.max()
            emit(tracer, f"{prefix}.cross.visual_to_text.global_max", global_max, "", "max", "exhaustive")
            attn_weights = attn_weights - global_max
            emit(tracer, f"{prefix}.cross.visual_to_text.logits_global_shifted", attn_weights, "BH,V,L", "subtract", "exhaustive")

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range
        emit(tracer, f"{prefix}.cross.visual_to_text.logits_clamped", attn_weights, "BH,V,L", "clamp", "exhaustive")

        attn_weights_T = attn_weights.transpose(1, 2)
        emit(tracer, f"{prefix}.cross.text_to_visual.logits_transposed", attn_weights_T, "BH,L,V", "transpose", "exhaustive")
        row_max = torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        emit(tracer, f"{prefix}.cross.text_to_visual.row_max", row_max, "BH,L,1", "max", "exhaustive")
        attn_weights_l = attn_weights_T - row_max
        emit(tracer, f"{prefix}.cross.text_to_visual.logits_shifted", attn_weights_l, "BH,L,V", "subtract", "exhaustive")
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range
        emit(tracer, f"{prefix}.cross.text_to_visual.logits_clamped", attn_weights_l, "BH,L,V", "clamp", "exhaustive")

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        mask_v = attention_mask_v if attention_mask_v is not None else torch.zeros_like(attn_weights_l, dtype=torch.bool)
        emit(tracer, f"{prefix}.cross.text_to_visual.mask", mask_v, "BH,L,V", "mask", "exhaustive")
        emit(tracer, f"{prefix}.cross.text_to_visual.masked_logits", attn_weights_l, "BH,L,V", "masked_fill", "op")

        trace_softmax(tracer, f"{prefix}.cross.text_to_visual.softmax", attn_weights_l, layout="BH,L,V", output_dtype=attn_weights_l.dtype)
        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        mask_l = attention_mask_l if attention_mask_l is not None else torch.zeros_like(attn_weights, dtype=torch.bool)
        emit(tracer, f"{prefix}.cross.visual_to_text.mask", mask_l, "BH,V,L", "mask", "exhaustive")
        emit(tracer, f"{prefix}.cross.visual_to_text.masked_logits", attn_weights, "BH,V,L", "masked_fill", "op")
        attn_weights_v = attn_weights.softmax(dim=-1)
        trace_softmax(tracer, f"{prefix}.cross.visual_to_text.softmax", attn_weights, layout="BH,V,L", output_dtype=attn_weights_v.dtype)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)
        emit(tracer, f"{prefix}.cross.visual_to_text.context_heads", attn_output_v, "BH,V,Dh", "bmm", "op")
        emit(tracer, f"{prefix}.cross.text_to_visual.context_heads", attn_output_l, "BH,L,Dh", "bmm", "op")

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        emit(tracer, f"{prefix}.cross.visual_to_text.context_transpose", attn_output_v, "B,V,H,Dh", "transpose", "exhaustive")
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)
        emit(tracer, f"{prefix}.cross.visual_to_text.context_merged", attn_output_v, "B,V,D", "reshape", "op")

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        emit(tracer, f"{prefix}.cross.text_to_visual.context_transpose", attn_output_l, "B,L,H,Dh", "transpose", "exhaustive")
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)
        emit(tracer, f"{prefix}.cross.text_to_visual.context_merged", attn_output_l, "B,L,D", "reshape", "op")

        trace_linear(tracer, f"{prefix}.cross.visual_to_text.output", attn_output_v, self.out_v_proj)
        trace_linear(tracer, f"{prefix}.cross.text_to_visual.output", attn_output_l, self.out_l_proj)
        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)
        emit(tracer, f"{prefix}.cross.visual_to_text.output", attn_output_v, "B,V,D", "linear", "layer")
        emit(tracer, f"{prefix}.cross.text_to_visual.output", attn_output_l, "B,L,D", "linear", "layer")

        return attn_output_v, attn_output_l


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
        residual_style="normalized",
        attention_backend="manual",
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_backend=attention_backend,
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)
        self.residual_style = residual_style

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None, tracer=None, prefix="interaction.layer_00"):
        residual_v, residual_l = v, l
        emit(tracer, f"{prefix}.visual.residual_source", residual_v, "B,V,D", "identity", "op")
        emit(tracer, f"{prefix}.text.residual_source", residual_l, "B,L,D", "identity", "op")
        trace_layer_norm(tracer, f"{prefix}.visual.norm", v, self.layer_norm_v)
        trace_layer_norm(tracer, f"{prefix}.text.norm", l, self.layer_norm_l)
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l,
            tracer=tracer, prefix=prefix,
        )
        if self.residual_style == "pre_norm":
            v, l = residual_v, residual_l
        scaled_v, scaled_l = self.gamma_v * delta_v, self.gamma_l * delta_l
        dropped_v, dropped_l = self.drop_path(scaled_v), self.drop_path(scaled_l)
        for side, gamma, delta, scaled, dropped in (("visual", self.gamma_v, delta_v, scaled_v, dropped_v),
                                                    ("text", self.gamma_l, delta_l, scaled_l, dropped_l)):
            emit(tracer, f"{prefix}.{side}.gamma", gamma, "D", "parameter", "op")
            emit(tracer, f"{prefix}.{side}.delta", delta, "B,N,D", "attention", "op")
            emit(tracer, f"{prefix}.{side}.scaled_delta", scaled, "B,N,D", "multiply", "exhaustive")
            emit(tracer, f"{prefix}.{side}.drop_path_output", dropped, "B,N,D", "drop_path", "exhaustive")
        v = v + dropped_v
        l = l + dropped_l
        emit(tracer, f"{prefix}.visual.residual_sum", v, "B,V,D", "add", "op")
        emit(tracer, f"{prefix}.text.residual_sum", l, "B,L,D", "add", "op")
        emit(tracer, f"{prefix}.visual.after_fusion", v, "B,V,D", "identity", "layer")
        emit(tracer, f"{prefix}.text.after_fusion", l, "B,L,D", "identity", "layer")
        return v, l

    # def forward(self, v:List[torch.Tensor], l, attention_mask_v=None, attention_mask_l=None)

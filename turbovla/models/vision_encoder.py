from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from .configuration import VisionEncoderConfig
from ..debug.model_trace import emit, trace_layer_norm, trace_linear, trace_softmax


def _load_pretrained_model(config: VisionEncoderConfig):
    kwargs = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": False,
    }
    if config.attention_implementation:
        try:
            return AutoModel.from_pretrained(
                config.model_name_or_path,
                attn_implementation=config.attention_implementation,
                **kwargs,
            )
        except Exception as error:
            if config.attention_implementation == "flash_attention_2":
                try:
                    return AutoModel.from_pretrained(
                        config.model_name_or_path,
                        attn_implementation="sdpa",
                        **kwargs,
                    )
                except Exception:
                    pass
            print(
                f"[TurboVLA] vision attention backend {config.attention_implementation!r} unavailable; "
                f"using model default ({type(error).__name__}).",
                flush=True,
            )
    return AutoModel.from_pretrained(config.model_name_or_path, **kwargs)


class DINOv3VisionEncoder(nn.Module):
    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _load_pretrained_model(config)
        self.hidden_size = self._hidden_size(self.backbone.config)
        self.patch_size = self._patch_size(self.backbone.config)
        self.prefix_tokens = self._prefix_tokens(self.backbone.config, default=5)
        self.num_patches = (config.image_size // self.patch_size) ** 2

        embeddings = getattr(self.backbone, "embeddings", None)
        mask_token = getattr(embeddings, "mask_token", None)
        if mask_token is not None:
            mask_token.requires_grad_(False)
        if config.frozen:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _hidden_size(config) -> int:
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "hidden_size"):
            return int(config.vision_config.hidden_size)
        raise AttributeError("cannot infer DINOv3 hidden size")

    @staticmethod
    def _patch_size(config) -> int:
        if hasattr(config, "patch_size"):
            return int(config.patch_size)
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "patch_size"):
            return int(config.vision_config.patch_size)
        raise AttributeError("cannot infer DINOv3 patch size")

    @staticmethod
    def _prefix_tokens(config, default: int = 0) -> int:
        if hasattr(config, "num_register_tokens"):
            return int(config.num_register_tokens) + 1
        if hasattr(config, "vision_config") and hasattr(config.vision_config, "num_register_tokens"):
            return int(config.vision_config.num_register_tokens) + 1
        return int(default)

    def set_compute_precision(self, precision: str) -> None:
        if precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError(f"unsupported DINOv3 precision: {precision}")
        self.config.compute_precision = precision
        if precision == "bf16":
            self.backbone.to(dtype=torch.bfloat16)
        else:
            self.backbone.float()

    def _encode_images(self, pixel_values: torch.Tensor, tracer=None, view_index: int = 0) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv3 input size {(height, width)} must be divisible by patch size {self.patch_size}"
            )
        expected_patches = (height // self.patch_size) * (width // self.patch_size)
        precision = self.config.compute_precision
        if precision == "bf16":
            pixel_values = pixel_values.to(dtype=torch.bfloat16)
        autocast_context = nullcontext()
        if precision == "bf16_autocast" and pixel_values.device.type == "cuda":
            autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        with grad_context:
            with autocast_context:
                outputs = self.backbone(
                    pixel_values=pixel_values, output_hidden_states=True,
                    output_attentions=False,
                )
        tokens = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        if tracer is not None and tracer.active:
            self._trace_backbone_math(tracer, view_index, pixel_values, outputs)
        if tokens.shape[1] == expected_patches + self.prefix_tokens:
            emit(tracer, f"vision.view_{view_index}.tokens_with_prefix_final", tokens, "B,P+N,D", "encoder_output", "boundary")
            emit(tracer, f"vision.view_{view_index}.prefix_tokens", tokens[:, :self.prefix_tokens], "B,P,D", "slice", "op")
            emit(tracer, f"vision.view_{view_index}.patch_tokens", tokens[:, self.prefix_tokens:], "B,N,D", "slice", "boundary")
            return tokens[:, self.prefix_tokens :, :]
        if tokens.shape[1] == expected_patches:
            emit(tracer, f"vision.view_{view_index}.tokens_with_prefix_final", tokens, "B,N,D", "encoder_output", "boundary")
            emit(tracer, f"vision.view_{view_index}.prefix_tokens", tokens[:, :0], "B,0,D", "slice", "op")
            emit(tracer, f"vision.view_{view_index}.patch_tokens", tokens, "B,N,D", "slice", "boundary")
            return tokens
        raise RuntimeError(
            f"DINOv3 produced {tokens.shape[1]} tokens; expected {expected_patches} patches "
            f"or {expected_patches + self.prefix_tokens} tokens including prefixes"
        )

    def _trace_backbone_math(self, tracer, view_index: int, pixel_values: torch.Tensor, outputs) -> None:
        prefix = f"vision.view_{view_index}"
        backbone = self.backbone
        emit(tracer, f"{prefix}.input", pixel_values, "B,C,H,W", "input", "boundary")
        patch = backbone.embeddings.patch_embeddings
        conv = patch(pixel_values.to(patch.weight.dtype))
        flattened = conv.flatten(2)
        transposed = flattened.transpose(1, 2)
        emit(tracer, f"{prefix}.patch_embed.input", pixel_values, "B,C,H,W", "input", "op")
        emit(tracer, f"{prefix}.patch_embed.conv.weight", patch.weight, "OUT,IN,KH,KW", "parameter", "exhaustive")
        if patch.bias is not None:
            emit(tracer, f"{prefix}.patch_embed.conv.bias", patch.bias, "OUT", "parameter", "exhaustive")
        else:
            emit(tracer, f"{prefix}.patch_embed.conv.bias", torch.zeros(patch.out_channels, device=conv.device, dtype=conv.dtype), "OUT", "constant", "exhaustive")
        emit(tracer, f"{prefix}.patch_embed.conv.output", conv, "B,D,PH,PW", "conv2d", "op")
        emit(tracer, f"{prefix}.patch_embed.flatten", flattened, "B,D,N", "flatten", "exhaustive")
        emit(tracer, f"{prefix}.patch_embed.transpose", transposed, "B,N,D", "transpose", "op")
        batch = pixel_values.shape[0]
        cls = backbone.embeddings.cls_token.expand(batch, -1, -1)
        registers = backbone.embeddings.register_tokens.expand(batch, -1, -1)
        before_position = torch.cat([cls, registers, transposed], dim=1)
        cos, sin = backbone.rope_embeddings(pixel_values)
        position = torch.stack([cos, sin], dim=0)
        emit(tracer, f"{prefix}.cls_token", cls, "B,1,D", "parameter_expand", "op")
        emit(tracer, f"{prefix}.register_tokens", registers, "B,R,D", "parameter_expand", "op")
        emit(tracer, f"{prefix}.position_embedding", position, "2,N,Dh", "rope", "op")
        emit(tracer, f"{prefix}.tokens_before_position", before_position, "B,P+N,D", "concat", "exhaustive")
        emit(tracer, f"{prefix}.tokens_after_position", before_position, "B,P+N,D", "rope_deferred", "exhaustive")
        emit(tracer, f"{prefix}.tokens_with_prefix", before_position, "B,P+N,D", "embedding_output", "layer")

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            return
        try:
            from transformers.models.dinov3_vit.modeling_dinov3_vit import apply_rotary_pos_emb
        except ImportError:
            apply_rotary_pos_emb = None
        for index, layer in enumerate(backbone.layer):
            base = f"{prefix}.block_{index:02d}"
            hidden = hidden_states[index]
            actual_output = hidden_states[index + 1]
            emit(tracer, f"{base}.input", hidden, "B,N,D", "layer_input", "layer")
            trace_layer_norm(tracer, f"{base}.norm_1", hidden, layer.norm1)
            norm_1 = layer.norm1(hidden)
            attn = layer.attention
            q = attn.q_proj(norm_1)
            k = attn.k_proj(norm_1)
            v = attn.v_proj(norm_1)
            weights = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
            biases = torch.cat([
                attn.q_proj.bias if attn.q_proj.bias is not None else torch.zeros(attn.embed_dim, device=hidden.device, dtype=hidden.dtype),
                attn.k_proj.bias if attn.k_proj.bias is not None else torch.zeros(attn.embed_dim, device=hidden.device, dtype=hidden.dtype),
                attn.v_proj.bias if attn.v_proj.bias is not None else torch.zeros(attn.embed_dim, device=hidden.device, dtype=hidden.dtype),
            ])
            qkv_matmul = torch.matmul(norm_1, weights.transpose(-1, -2))
            emit(tracer, f"{base}.attn.qkv_linear.matmul", qkv_matmul, "B,N,3D", "matmul", "exhaustive")
            emit(tracer, f"{base}.attn.qkv_linear.bias_add", qkv_matmul + biases, "B,N,3D", "add", "exhaustive")
            for name, item in (("q", q), ("k", k), ("v", v)):
                emit(tracer, f"{base}.attn.{name}", item, "B,N,D", "linear", "op")
            heads, head_dim = attn.num_heads, attn.head_dim
            qh = q.view(q.shape[0], q.shape[1], heads, head_dim).transpose(1, 2)
            kh = k.view(k.shape[0], k.shape[1], heads, head_dim).transpose(1, 2)
            vh = v.view(v.shape[0], v.shape[1], heads, head_dim).transpose(1, 2)
            if apply_rotary_pos_emb is not None:
                qh, kh = apply_rotary_pos_emb(qh, kh, cos, sin)
            for name, item in (("q_heads", qh), ("k_heads", kh), ("v_heads", vh)):
                emit(tracer, f"{base}.attn.{name}", item, "B,H,N,Dh", "reshape_rope", "op")
            logits = torch.matmul(qh, kh.transpose(-1, -2))
            scale = torch.tensor(attn.scaling, device=hidden.device, dtype=hidden.dtype)
            scaled = logits * scale
            emit(tracer, f"{base}.attn.logits", logits, "B,H,N,N", "matmul", "op")
            emit(tracer, f"{base}.attn.scale", scale, "", "constant", "exhaustive")
            emit(tracer, f"{base}.attn.scaled_logits", scaled, "B,H,N,N", "multiply", "op")
            probs = trace_softmax(tracer, f"{base}.attn.softmax", scaled, layout="B,H,N,N", output_dtype=hidden.dtype)
            context_heads = torch.matmul(probs, vh)
            context = context_heads.transpose(1, 2).reshape(hidden.shape)
            projected = attn.o_proj(context)
            emit(tracer, f"{base}.attn.context_heads", context_heads, "B,H,N,Dh", "matmul", "op")
            emit(tracer, f"{base}.attn.context_merged", context, "B,N,D", "reshape", "op")
            emit(tracer, f"{base}.attn.output_projection", projected, "B,N,D", "linear", "op")
            scaled_1 = layer.layer_scale1(projected)
            emit(tracer, f"{base}.layer_scale_1.input", projected, "B,N,D", "identity", "exhaustive")
            emit(tracer, f"{base}.layer_scale_1.output", scaled_1, "B,N,D", "multiply", "op")
            residual_1 = hidden + scaled_1
            for suffix, item in (("left", hidden), ("right", scaled_1), ("sum", residual_1)):
                emit(tracer, f"{base}.residual_1.{suffix}", item, "B,N,D", "add" if suffix == "sum" else "identity", "op")
            trace_layer_norm(tracer, f"{base}.norm_2", residual_1, layer.norm2)
            norm_2 = layer.norm2(residual_1)
            up = trace_linear(tracer, f"{base}.mlp.linear_1", norm_2, layer.mlp.up_proj)
            activated = layer.mlp.act_fn(up)
            emit(tracer, f"{base}.mlp.activation.input", up, "B,N,F", "activation", "exhaustive")
            emit(tracer, f"{base}.mlp.activation.output", activated, "B,N,F", "activation", "op")
            down = trace_linear(tracer, f"{base}.mlp.linear_2", activated, layer.mlp.down_proj)
            scaled_2 = layer.layer_scale2(down)
            emit(tracer, f"{base}.layer_scale_2.input", down, "B,N,D", "identity", "exhaustive")
            emit(tracer, f"{base}.layer_scale_2.output", scaled_2, "B,N,D", "multiply", "op")
            for suffix, item in (("left", residual_1), ("right", scaled_2), ("sum", residual_1 + scaled_2)):
                emit(tracer, f"{base}.residual_2.{suffix}", item, "B,N,D", "add" if suffix == "sum" else "identity", "op")
            emit(tracer, f"{base}.output", actual_output, "B,N,D", "layer_output", "layer")

    def forward(
        self,
        pixel_values: torch.Tensor,
        dump: Callable[[str, torch.Tensor], None] | None = None,
        tracer=None,
    ) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(f"pixel_values must be [B,V,3,H,W], got {tuple(pixel_values.shape)}")
        batch_size, num_views = pixel_values.shape[:2]
        if num_views != self.config.num_views:
            raise ValueError(f"expected {self.config.num_views} views, got {num_views}")
        if self.config.encode_views_separately:
            tokens = torch.stack(
                [self._encode_images(pixel_values[:, view_idx], tracer=tracer, view_index=view_idx) for view_idx in range(num_views)],
                dim=1,
            )
        else:
            flat = pixel_values.flatten(0, 1)
            tokens = self._encode_images(flat, tracer=tracer, view_index=0)
            tokens = tokens.view(batch_size, num_views, tokens.shape[1], tokens.shape[2])
        if dump is not None:
            dump("vision.dinov3_patch_tokens", tokens)
        emit(tracer, "vision.patch_tokens_stacked", tokens, "B,V,N,D", "stack", "boundary")
        return tokens

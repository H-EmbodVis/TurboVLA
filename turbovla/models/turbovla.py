from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .action_head import TurboVLAActionHead
from .components.fusion import BiAttentionBlock
from .components.transformer import TransformerEncoderLayer
from .components.utils import _get_clones
from .configuration import (
    ActionHeadConfig,
    EmbeddingDumpConfig,
    InteractionConfig,
    TextEncoderConfig,
    TurboVLAConfig,
    VisionEncoderConfig,
)
from .embedding_dump import EmbeddingDumper
from .text_encoder import TurboVLATextEncoder
from .vision_encoder import DINOv3VisionEncoder
from ..debug import TraceConfig, TraceContext
from ..debug.model_trace import emit, tensor_axis, trace_layer_norm, trace_linear


class VisionProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, tokens: torch.Tensor, tracer: Any | None = None) -> torch.Tensor:
        emit(tracer, "vision_projection.input", tokens, "B,V,N,Din", "input", "boundary")
        normalized = trace_layer_norm(tracer, "vision_projection.input_norm", tokens, self.input_norm, layout="B,V,N,Din")
        skip = torch.matmul(tokens, self.skip.weight.transpose(-1, -2))
        emit(tracer, "vision_projection.skip.matmul", skip, "B,V,N,D", "matmul", "exhaustive")
        emit(tracer, "vision_projection.skip.output", self.skip(tokens), "B,V,N,D", "linear", "op")
        mlp_1 = trace_linear(tracer, "vision_projection.mlp.linear_1", normalized, self.mlp[0], layout="B,V,N,F")
        gelu = self.mlp[1](mlp_1)
        emit(tracer, "vision_projection.mlp.gelu.input", mlp_1, "B,V,N,F", "gelu", "exhaustive")
        emit(tracer, "vision_projection.mlp.gelu.output", gelu, "B,V,N,F", "gelu", "op")
        mlp_2 = trace_linear(tracer, "vision_projection.mlp.linear_2", gelu, self.mlp[3], layout="B,V,N,D")
        actual_skip, actual_mlp = self.skip(tokens), self.mlp(self.input_norm(tokens))
        emit(tracer, "vision_projection.residual.left", actual_skip, "B,V,N,D", "identity", "exhaustive")
        emit(tracer, "vision_projection.residual.right", actual_mlp, "B,V,N,D", "identity", "exhaustive")
        summed = actual_skip + actual_mlp
        emit(tracer, "vision_projection.residual.sum", summed, "B,V,N,D", "add", "op")
        trace_layer_norm(tracer, "vision_projection.output_norm", summed, self.output_norm, layout="B,V,N,D")
        return self.output_norm(summed)


class VisionLanguageInteraction(nn.Module):
    def __init__(self, config: InteractionConfig) -> None:
        super().__init__()
        text_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=max(1, config.nheads // 2),
            dim_feedforward=config.enhancer_inner_dim,
            dropout=config.text_dropout,
        )
        fusion_layer = BiAttentionBlock(
            v_dim=config.hidden_dim,
            l_dim=config.hidden_dim,
            embed_dim=config.enhancer_inner_dim,
            num_heads=max(1, config.nheads // 2),
            dropout=config.fusion_dropout,
            drop_path=config.fusion_droppath,
            residual_style=config.residual_style,
            attention_backend=config.attention_backend,
        )
        self.text_layers = _get_clones(text_layer, config.num_layers)
        self.fusion_layers = _get_clones(fusion_layer, config.num_layers)
        self.padding_strategy = config.padding_strategy

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_self_attention_masks: torch.Tensor | None,
        dump: Callable[[str, torch.Tensor], None] | None = None,
        tracer: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_fill = self.padding_strategy == "zero_fill"
        if zero_fill:
            text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)

        for layer_index, (fusion_layer, text_layer) in enumerate(zip(self.fusion_layers, self.text_layers)):
            if tracer is not None and tracer.active:
                tracer.tensor(f"interaction.layer_{layer_index:02d}.visual.input", visual_tokens, layout="B,VxN,D", operation="layer_input")
                tracer.tensor(f"interaction.layer_{layer_index:02d}.text.input", text_tokens, layout="B,N,D", operation="layer_input")
            visual_tokens, text_tokens = fusion_layer(
                v=visual_tokens,
                l=text_tokens,
                attention_mask_v=None,
                attention_mask_l=text_key_padding_mask,
                tracer=tracer,
                prefix=f"interaction.layer_{layer_index:02d}",
            )
            if dump is not None:
                dump(f"interaction.layer_{layer_index}.visual_after_fusion", visual_tokens)
                dump(f"interaction.layer_{layer_index}.text_after_fusion", text_tokens)
            source_mask = None if text_self_attention_masks is None else ~text_self_attention_masks
            text_tokens = text_layer(
                src=text_tokens.transpose(0, 1),
                src_mask=source_mask,
                src_key_padding_mask=None if zero_fill else text_key_padding_mask,
                pos=None,
                tracer=tracer,
                prefix=f"interaction.layer_{layer_index:02d}.text_enhancer",
            ).transpose(0, 1)
            if zero_fill:
                text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
            if dump is not None:
                dump(f"interaction.layer_{layer_index}.text_after_enhancer", text_tokens)
        return visual_tokens, text_tokens


class TurboVLA(nn.Module):
    """Shared TurboVLA architecture for LIBERO and RoboTwin."""

    def __init__(self, config: TurboVLAConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.interaction.hidden_dim
        self.action_dim = int(config.action.action_dim)
        self.chunk_size = int(config.action.horizon)
        self.state_dim = int(config.action.state_dim)
        self.num_views = int(config.vision.num_views)
        self.embedding_dumper = EmbeddingDumper(config.embedding_dump)
        # Parity trace context – disabled by default.
        self._parity_tracer: TraceContext | None = None

        self.text_encoder = TurboVLATextEncoder(config.text, hidden_dim=hidden_dim)
        self.vision_encoder = DINOv3VisionEncoder(config.vision)
        self.vision_projection = VisionProjection(
            in_dim=self.vision_encoder.hidden_size,
            out_dim=hidden_dim,
            hidden_dim=max(hidden_dim * 4, self.vision_encoder.hidden_size // 2),
            dropout=config.vision.dropout,
        )

        if config.vision.position_embedding == "learned_patch":
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, 1, hidden_dim))
            self.patch_position_embedding = nn.Parameter(
                torch.zeros(1, self.num_views, self.vision_encoder.num_patches, hidden_dim)
            )
            self.patch_position_scale = nn.Parameter(
                torch.full((1, self.num_views, 1, 1), float(config.vision.position_scale_init))
            )
            nn.init.trunc_normal_(self.patch_position_embedding, std=config.vision.position_init_std)
        else:
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, hidden_dim))
            self.register_parameter("patch_position_embedding", None)
            self.register_parameter("patch_position_scale", None)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

        self.vision_language_interaction = VisionLanguageInteraction(config.interaction)
        self.action_head = TurboVLAActionHead(
            config=config.action,
            hidden_dim=hidden_dim,
            nheads=config.interaction.nheads,
            dim_feedforward=config.interaction.dim_feedforward,
        )

    def _normalize_samples(self, samples: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(samples, Mapping):
            if "dinov3" not in samples:
                raise ValueError("samples mapping must contain 'dinov3'")
            pixel_values = samples["dinov3"]
        else:
            pixel_values = samples
        if pixel_values.ndim == 6:
            pixel_values = pixel_values[:, -1]
        if pixel_values.ndim != 5:
            raise ValueError(f"samples must be [B,V,3,H,W] or [B,T,V,3,H,W], got {tuple(pixel_values.shape)}")
        return pixel_values

    def _position_visual_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.config.vision.position_embedding == "learned_patch":
            if tokens.shape[2] != self.patch_position_embedding.shape[2]:
                raise ValueError(
                    f"configured patch position length {self.patch_position_embedding.shape[2]} "
                    f"does not match encoded length {tokens.shape[2]}"
                )
            position = self.patch_position_embedding.to(device=tokens.device, dtype=tokens.dtype)
            scale = self.patch_position_scale.to(device=tokens.device, dtype=tokens.dtype)
            view = self.view_embedding.to(device=tokens.device, dtype=tokens.dtype)
            return tokens + scale * position + view
        view = self.view_embedding[:, :, None, :].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + view

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        dump = self.embedding_dumper.record if self.embedding_dumper.active else None
        tracer = self._parity_tracer
        tokens = self.vision_encoder(pixel_values, dump=dump, tracer=tracer)
        tokens = tokens.to(dtype=self.vision_projection.skip.weight.dtype)
        tokens = self.vision_projection(tokens, tracer=tracer)
        if dump is not None:
            dump("vision.projected_tokens", tokens)
        if tracer is not None and tracer.active:
            tracer.tensor("vision_projection.view_embedding", self.view_embedding, layout="1,V,D", operation="parameter", required_level="op")
            tracer.tensor("vision_projection.position.before_add", tokens, layout="B,V,N,D", operation="identity", required_level="exhaustive")
        tokens = self._position_visual_tokens(tokens)
        if dump is not None:
            dump("vision.positioned_tokens", tokens)
        if tracer is not None and tracer.active:
            tracer.tensor("vision_projection.position.after_add", tokens, layout="B,V,N,D", operation="position_add", required_level="op")
            tracer.tensor("vision_projection.before_flatten", tokens, layout="B,V,N,D", operation="identity", required_level="exhaustive")
        tokens = tokens.flatten(1, 2)
        if dump is not None:
            dump("vision.flattened_tokens", tokens)
        if tracer is not None and tracer.active:
            tracer.tensor("vision_projection.flattened", tokens, layout="B,VxN,D", operation="flatten")
        return tokens

    def encode_condition(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        pixel_values = self._normalize_samples(samples)
        device = pixel_values.device
        precision_context = nullcontext()
        if self.config.interaction.compute_precision == "bf16_autocast" and device.type == "cuda":
            precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with precision_context:
            dump = self.embedding_dumper.record if self.embedding_dumper.active else None
            tracer = self._parity_tracer
            if dump is not None:
                dump("vision.pixel_values", pixel_values)
            text_tokens, text_key_padding_mask, text_self_attention_masks = self.text_encoder(
                instructions,
                device=device,
                dump=dump,
                tracer=tracer,
            )
            if text_tokens.shape[0] != pixel_values.shape[0]:
                raise ValueError("instruction batch size does not match image batch size")
            visual_tokens = self.encode_vision(pixel_values)
            visual_tokens, text_tokens = self.vision_language_interaction(
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
                text_self_attention_masks=text_self_attention_masks,
                dump=dump,
                tracer=tracer,
            )
            condition = torch.cat([visual_tokens, text_tokens], dim=1)
            if dump is not None:
                dump("condition.vision_language_tokens", condition)
            if tracer is not None and tracer.active:
                tracer.tensor("condition.visual_tokens", visual_tokens, layout="B,VxN,D", operation="split")
                tracer.tensor("condition.text_tokens", text_tokens, layout="B,N,D", operation="split")
                tracer.tensor("condition.concat_axis", tensor_axis(1, condition.device), layout="", operation="constant", required_level="exhaustive")
                tracer.tensor("condition.concatenated", condition, layout="B,VxN+N,D", operation="concat")
            return condition

    def forward(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        tracer = self._parity_tracer
        # EmbeddingDumper is retained only for explicit legacy inspection.
        self.embedding_dumper.begin(self)
        try:
            condition = self.encode_condition(instructions, samples)
            action_dtype = self.action_head.decoder.action_queries.weight.dtype
            dump = self.embedding_dumper.record if self.embedding_dumper.active else None
            if tracer is not None and tracer.active:
                tracer.tensor("state_projection.input_raw", state, layout="B,D", operation="input")
            actions = self.action_head(
                condition.to(dtype=action_dtype),
                state.to(dtype=action_dtype),
                dump=dump,
                tracer=tracer,
            )
        except Exception as error:
            self.embedding_dumper.finish(error=f"{type(error).__name__}: {error}")
            raise
        self.embedding_dumper.finish()
        return actions

    def set_parity_tracer(self, tracer: TraceContext | None) -> None:
        """Attach a parity trace context. Pass None to disable."""
        self._parity_tracer = tracer

    # Transitional read-only names used only by legacy checkpoint initialization.
    @property
    def dinov3(self):
        return self.vision_encoder.backbone

    @property
    def text_proj(self):
        return self.text_encoder.text_projection

    @property
    def vision_proj(self):
        return self.vision_projection

    @property
    def feature_enhancer(self):
        return self.vision_language_interaction

    @property
    def state_proj(self):
        return self.action_head.state_projection

    @property
    def action_policy(self):
        return self.action_head.decoder


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_turbovla(args: TurboVLAConfig | Mapping[str, Any] | Any) -> TurboVLA:
    if isinstance(args, TurboVLAConfig):
        config = args
    elif isinstance(args, Mapping):
        config = TurboVLAConfig.from_mapping(args)
    else:
        config = TurboVLAConfig(
            text=TextEncoderConfig(
                model_name_or_path=_arg(args, "bert_path", "bert-base-uncased"),
                max_length=int(_arg(args, "max_text_len", 256)),
                padding_length=_arg(args, "text_padding_length", None),
                padding_length_by_instruction=dict(_arg(args, "text_padding_length_by_instruction", {})),
                sub_sentence_present=bool(_arg(args, "sub_sentence_present", True)),
                frozen=bool(_arg(args, "freeze_text_encoder", True)),
                force_eval_when_frozen=True,
                zero_padded_tokens=bool(_arg(args, "zero_padded_text", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "text_attention_implementation", None),
            ),
            vision=VisionEncoderConfig(
                model_name_or_path=_arg(args, "dinov3_path", _arg(args, "LOCAL_DINOV3_PATH", "")),
                image_size=int(_arg(args, "image_size", _arg(args, "expected_image_size", 256))),
                num_views=int(_arg(args, "num_views", 2)),
                position_embedding=str(_arg(args, "position_embedding", "view")),
                encode_views_separately=bool(_arg(args, "encode_views_separately", True)),
                frozen=bool(_arg(args, "freeze_vision_encoder", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "vision_attention_implementation", None),
                compute_precision=str(_arg(args, "dinov3_precision", "bf16_autocast")),
                dropout=float(_arg(args, "vision_dropout", 0.1)),
            ),
            interaction=InteractionConfig(
                hidden_dim=int(_arg(args, "hidden_dim", 256)),
                nheads=int(_arg(args, "nheads", 8)),
                num_layers=int(_arg(args, "vla_feature_enhancer_layers", 6)),
                dim_feedforward=int(_arg(args, "dim_feedforward", 2048)),
                enhancer_inner_dim=int(_arg(args, "enhancer_inner_dim", 1024)),
                text_dropout=float(_arg(args, "text_dropout", 0.0)),
                fusion_dropout=float(_arg(args, "fusion_dropout", 0.0)),
                fusion_droppath=float(_arg(args, "fusion_droppath", 0.1)),
                padding_strategy=str(_arg(args, "padding_strategy", "key_padding_mask")),
                residual_style=str(_arg(args, "residual_style", "normalized")),
                attention_backend=str(_arg(args, "attention_backend", "manual")),
                compute_precision=str(_arg(args, "interaction_precision", "fp32")),
            ),
            action=ActionHeadConfig(
                action_dim=int(_arg(args, "action_dim", 7)),
                state_dim=int(_arg(args, "state_dim", 8)),
                horizon=int(_arg(args, "chunk_size", _arg(args, "action_horizon", 12))),
                num_state_tokens=int(_arg(args, "num_state_tokens", 2)),
                num_layers=int(_arg(args, "act_num_layers", 3)),
                mlp_hidden_dim=int(_arg(args, "act_mlp_hidden_dim", 512)),
                state_hidden_dim=int(_arg(args, "act_state_hidden_dim", 256)),
                dropout=float(_arg(args, "act_dropout", 0.1)),
            ),
            embedding_dump=EmbeddingDumpConfig(
                enabled=bool(_arg(args, "dump_embeddings", False)),
                output_dir=str(_arg(args, "embedding_dump_dir", "embedding_dumps")),
                max_dumps=int(_arg(args, "embedding_dump_max", 1)),
                every_n_forwards=int(_arg(args, "embedding_dump_every", 1)),
                rank_zero_only=bool(_arg(args, "embedding_dump_rank_zero_only", True)),
                print_summary=bool(_arg(args, "embedding_dump_print_summary", True)),
                preview_values=int(_arg(args, "embedding_dump_preview_values", 8)),
            ),
        )
    return TurboVLA(config)

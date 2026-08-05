from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class TextEncoderConfig:
    model_name_or_path: str = "bert-base-uncased"
    max_length: int = 256
    padding_length: int | None = None
    padding_length_by_instruction: dict[str, int] = field(default_factory=dict)
    sub_sentence_present: bool = True
    frozen: bool = True
    force_eval_when_frozen: bool = True
    zero_padded_tokens: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None


@dataclass
class VisionEncoderConfig:
    model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size: int = 256
    num_views: int = 2
    position_embedding: str = "view"
    encode_views_separately: bool = True
    frozen: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None
    compute_precision: str = "bf16_autocast"
    position_init_std: float = 0.01
    position_scale_init: float = 0.01
    dropout: float = 0.1


@dataclass
class InteractionConfig:
    hidden_dim: int = 256
    nheads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    enhancer_inner_dim: int = 1024
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    padding_strategy: str = "key_padding_mask"
    residual_style: str = "normalized"
    attention_backend: str = "manual"
    compute_precision: str = "fp32"


@dataclass
class ActionHeadConfig:
    action_dim: int = 7
    state_dim: int = 8
    horizon: int = 12
    num_state_tokens: int = 2
    num_layers: int = 3
    mlp_hidden_dim: int = 512
    state_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class EmbeddingDumpConfig:
    enabled: bool = False
    output_dir: str = "embedding_dumps"
    max_dumps: int = 1
    every_n_forwards: int = 1
    rank_zero_only: bool = True
    print_summary: bool = True
    preview_values: int = 8
    capture_module_io: bool = True
    save_pt: bool = True
    save_raw: bool = True


@dataclass
class TurboVLAConfig:
    name: str = "TurboVLA"
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    action: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    embedding_dump: EmbeddingDumpConfig = field(default_factory=EmbeddingDumpConfig)

    def __post_init__(self) -> None:
        if self.name != "TurboVLA":
            raise ValueError(f"model name must be 'TurboVLA', got {self.name!r}")
        if self.vision.num_views < 1:
            raise ValueError("vision.num_views must be positive")
        if self.vision.position_embedding not in {"view", "learned_patch"}:
            raise ValueError("vision.position_embedding must be 'view' or 'learned_patch'")
        if self.vision.compute_precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError("vision.compute_precision must be fp32, bf16, or bf16_autocast")
        if self.text.padding_length is not None:
            if self.text.padding_length < 1:
                raise ValueError("text.padding_length must be positive")
            if self.text.padding_length > self.text.max_length:
                raise ValueError("text.padding_length cannot exceed text.max_length")
        for instruction, length in self.text.padding_length_by_instruction.items():
            if not instruction:
                raise ValueError("text.padding_length_by_instruction cannot contain an empty instruction")
            if length < 1 or length > self.text.max_length:
                raise ValueError(f"invalid text padding length {length} for instruction {instruction!r}")
        if self.interaction.hidden_dim % self.interaction.nheads != 0:
            raise ValueError("interaction.hidden_dim must be divisible by interaction.nheads")
        if self.interaction.padding_strategy not in {"key_padding_mask", "zero_fill"}:
            raise ValueError("interaction.padding_strategy must be key_padding_mask or zero_fill")
        if self.interaction.residual_style not in {"normalized", "pre_norm"}:
            raise ValueError("interaction.residual_style must be normalized or pre_norm")
        if self.interaction.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("interaction.attention_backend must be manual or sdpa")
        if self.interaction.compute_precision not in {"fp32", "bf16_autocast"}:
            raise ValueError("interaction.compute_precision must be fp32 or bf16_autocast")
        if self.action.action_dim < 1 or self.action.state_dim < 1 or self.action.horizon < 1:
            raise ValueError("action dimensions and horizon must be positive")
        if self.embedding_dump.max_dumps < 1:
            raise ValueError("embedding_dump.max_dumps must be positive")
        if self.embedding_dump.every_n_forwards < 1:
            raise ValueError("embedding_dump.every_n_forwards must be positive")
        if self.embedding_dump.preview_values < 0:
            raise ValueError("embedding_dump.preview_values cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TurboVLAConfig":
        data = dict(payload)
        return cls(
            name=str(data.get("name", "TurboVLA")),
            text=TextEncoderConfig(**dict(data.get("text", {}))),
            vision=VisionEncoderConfig(**dict(data.get("vision", {}))),
            interaction=InteractionConfig(**dict(data.get("interaction", {}))),
            action=ActionHeadConfig(**dict(data.get("action", {}))),
            embedding_dump=EmbeddingDumpConfig(**dict(data.get("embedding_dump", {}))),
        )

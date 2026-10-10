"""Exact TurboVLA LIBERO reference under the upstream-supported runtime."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel


PORT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PORT_ROOT.parents[1]
CHECKPOINT_PATH = SOURCE_ROOT / "pretrained" / "TurboVLA" / "checkpoints" / "libero" / "object.pth"
BERT_PATH = SOURCE_ROOT / "pretrained" / "bert-base-uncased"

TEXT_OUTPUT_LENGTH = 21
IMAGE_SIZE = 256
PATCH_SIZE = 16


def dinov3_vitb16_config() -> DINOv3ViTConfig:
    """Return the fixed ViT-B/16 architecture used by the released checkpoint."""
    return DINOv3ViTConfig(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_register_tokens=4,
        hidden_act="gelu",
        attention_dropout=0.0,
        layer_norm_eps=1e-5,
        rope_theta=100.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        use_gated_mlp=False,
    )


def _source_modules():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from turbovla.models.configuration import TurboVLAConfig
    from turbovla.models.turbovla import TurboVLA, build_turbovla
    import turbovla.models.vision_encoder as vision_encoder

    return TurboVLAConfig, TurboVLA, build_turbovla, vision_encoder


@contextmanager
def _instantiate_dinov3_from_config(vision_encoder) -> Iterator[None]:
    """Avoid an HF gated download: release checkpoint strictly supplies every tensor."""
    original_loader = vision_encoder._load_pretrained_model

    def build_backbone(_config):
        return DINOv3ViTModel(dinov3_vitb16_config())

    vision_encoder._load_pretrained_model = build_backbone
    try:
        yield
    finally:
        vision_encoder._load_pretrained_model = original_loader


def load_object_reference(
    device: str | torch.device = "cpu",
    checkpoint_path: Path = CHECKPOINT_PATH,
    bert_path: Path = BERT_PATH,
    dino_attention_implementation: str | None = None,
    bert_attention_implementation: str | None = None,
) -> nn.Module:
    """Strict-load the released LIBERO Object policy without version drift.

    TurboVLA's checkpoint includes all DINOv3 and BERT weights. The local
    BERT directory supplies tokenizer/config construction; DINOv3 is created
    from its fixed architecture and immediately fully overwritten by the
    release state dict. This retains the upstream `hidden_states[-1]` path.
    """
    checkpoint_path = Path(checkpoint_path)
    bert_path = Path(bert_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not bert_path.is_dir():
        raise FileNotFoundError(bert_path)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_payload = payload.get("model_config")
    state_dict = payload.get("model_state_dict")
    if not isinstance(config_payload, dict) or not isinstance(state_dict, dict):
        raise RuntimeError("expected a released TurboVLA checkpoint with model_config and model_state_dict")

    TurboVLAConfig, _TurboVLA, build_turbovla, vision_encoder = _source_modules()
    config = TurboVLAConfig.from_mapping(config_payload)
    config.text.model_name_or_path = str(bert_path)
    config.text.local_files_only = True
    config.text.attention_implementation = bert_attention_implementation
    config.vision.model_name_or_path = "local-dinov3-vitb16-from-release-state"
    config.vision.local_files_only = True
    config.vision.compute_precision = "fp32"

    with _instantiate_dinov3_from_config(vision_encoder):
        model = build_turbovla(config)
    model.load_state_dict(state_dict, strict=True)
    if dino_attention_implementation is not None:
        model.vision_encoder.backbone.config._attn_implementation = dino_attention_implementation
    model.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model


class OneViewDINOv3(nn.Module):
    """One original TurboVLA DINOv3 view, retaining issue #4's feature choice."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.vision_encoder.backbone
        self.prefix_tokens = int(model.vision_encoder.prefix_tokens)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        # Issue #4: do not replace this with last_hidden_state.
        patch_tokens = outputs.hidden_states[-1][:, self.prefix_tokens :, :]
        return patch_tokens.reshape(1, 256, 768)


class StaticBert(nn.Module):
    """BERT invocation with host-generated TurboVLA special-token masks."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.bert = model.text_encoder.bert

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        text_self_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=text_self_attention_mask,
            position_ids=position_ids,
        ).last_hidden_state


class StaticPolicyCore(nn.Module):
    """TurboVLA after DINO/BERT, with the original fixed 21-token policy layout."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.vision_projection = model.vision_projection
        self.view_embedding = model.view_embedding
        self.vision_language_interaction = model.vision_language_interaction
        self.text_projection = model.text_encoder.text_projection
        self.action_head = model.action_head

    def forward(
        self,
        vision_view0: torch.Tensor,
        vision_view1: torch.Tensor,
        bert_hidden_padded: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_self_attention_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        vision = torch.stack((vision_view0, vision_view1), dim=1)
        visual_tokens = self.vision_projection(vision)
        visual_tokens = visual_tokens + self.view_embedding[:, :, None, :].to(
            device=visual_tokens.device, dtype=visual_tokens.dtype
        )
        visual_tokens = visual_tokens.flatten(1, 2)
        text_tokens = self.text_projection(bert_hidden_padded)
        visual_tokens, text_tokens = self.vision_language_interaction(
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
            text_key_padding_mask=text_key_padding_mask,
            text_self_attention_masks=text_self_attention_mask,
        )
        return self.action_head(torch.cat((visual_tokens, text_tokens), dim=1), state)


def prepare_static_text_inputs(model: nn.Module, instruction: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Reproduce TurboVLA's per-instruction BERT length and 21-token padding."""
    text_encoder = model.text_encoder
    configured_length = int(text_encoder.config.padding_length or TEXT_OUTPUT_LENGTH)
    group_length = int(text_encoder.config.padding_length_by_instruction.get(instruction, configured_length))
    tokenized, group_self_mask, position_ids = text_encoder._tokenize_group([instruction], device, group_length)

    text_key_padding_mask = torch.ones((1, configured_length), dtype=torch.bool, device=device)
    text_key_padding_mask[:, :group_length] = ~tokenized.attention_mask.bool()
    text_self_attention_mask = torch.eye(configured_length, dtype=torch.bool, device=device).unsqueeze(0)
    text_self_attention_mask[:, :group_length, :group_length] = group_self_mask
    return {
        "input_ids": tokenized.input_ids,
        "token_type_ids": tokenized.token_type_ids,
        "bert_attention_mask": group_self_mask,
        "position_ids": position_ids,
        "text_key_padding_mask": text_key_padding_mask,
        "text_self_attention_mask": text_self_attention_mask,
        "group_length": torch.tensor(group_length, device=device),
    }


def pad_bert_hidden(bert_hidden: torch.Tensor, output_length: int = TEXT_OUTPUT_LENGTH) -> torch.Tensor:
    if bert_hidden.ndim != 3 or bert_hidden.shape[0] != 1 or bert_hidden.shape[1] > output_length:
        raise ValueError(f"expected BERT hidden [1,L,768] where L<={output_length}, got {tuple(bert_hidden.shape)}")
    padded = bert_hidden.new_zeros((1, output_length, bert_hidden.shape[-1]))
    padded[:, : bert_hidden.shape[1]] = bert_hidden
    return padded
